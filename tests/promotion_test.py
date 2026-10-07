"""Historical average gate, immutable originals and exact cache/resume provenance."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import my_board_engine as engine
import torch

from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig, save_model
from kingdom_ai.checkpoint import _file_digest, export_training_model
from kingdom_ai.encoding import encode_state
from kingdom_ai.evaluation import EvaluationResult, evaluate_models
from kingdom_ai.metrics import compact_metric_row, summarize_metrics
import kingdom_ai.loop as loop
from kingdom_ai.promotion import PromotionLeague
from kingdom_ai.training import GameData, TrainingSample


REPOSITORY = Path(__file__).resolve().parents[1]
OPTIONS = dict(simulations=1, c_puct=1.5, opening_moves=2, opening_temperature=1.0,
               workers=1, backend="legacy", leaf_batch_size=8, reuse_tree=True,
               tactical_checks=False)


def result(wins, games=100):
    black = min(wins, games // 2)
    return EvaluationResult(games, wins, games - wins, black, wins - black,
                            games * 2, {"TwoPasses": games})


def pass_game(*args, **kwargs):
    state, samples = engine.State(), []
    for _ in range(2):
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[81] = 1
        samples.append(TrainingSample(encoded.features, encoded.legal_mask, policy,
            -1.0 if encoded.to_play == engine.Cell.Black else 1.0, encoded.to_play))
        state.pass_turn()
    return GameData(samples, state.result.winner, state.result.reason, (81, 81))


class PromotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="promotion_test_", dir=REPOSITORY / "runs")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.archive = self.root / "past-version"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(26)
            self.candidate = PolicyValueNet(channels=4, residual_blocks=0)
            self.reference = PolicyValueNet(channels=4, residual_blocks=0)
        for version in (177, 106):
            folder = self.archive / f"champion_{version}v"
            folder.mkdir(parents=True)
            save_model(self.reference, folder / "best.pt")
        self.originals = self.snapshot()

    def snapshot(self):
        return {str(path): (_file_digest(path), path.stat().st_mtime_ns)
                for path in self.archive.rglob("*") if path.is_file()}

    def league(self, **kwargs):
        return PromotionLeague(str(self.archive), **kwargs)

    def trainer(self, **kwargs):
        config = TrainingConfig(games_per_iteration=2, simulations=1, replay_capacity=32,
            batch_size=4, train_steps_per_iteration=1, evaluation_games=40, evaluation_simulations=1,
            promotion_threshold=0.55, promotion_archive_dir=str(self.archive), **kwargs)
        return Trainer(config, self.candidate)

    def run_cycle(self, trainer, results):
        with patch.object(loop, "collect_puct_game", side_effect=pass_game), \
                patch.object(loop, "evaluate_models", side_effect=results) as evaluate:
            metrics = trainer.run_iteration()
        return metrics, evaluate

    def test_gate_below_twenty_two_of_forty_skips_entire_archive(self):
        trainer = self.trainer()
        metrics, evaluate = self.run_cycle(trainer, [result(21, 40)])
        self.assertFalse(metrics["promoted"])
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(metrics["promotion_league"]["reason"], "head_to_head_rejected")
        self.assertIsNone(trainer.promotion_league.reference_cache)

    def test_gate_equal_fifty_five_percent_promotes_only_on_higher_average(self):
        trainer = self.trainer()
        metrics, evaluate = self.run_cycle(trainer, [result(22, 40), result(60), result(60),
                                                    result(65), result(58)])
        self.assertTrue(metrics["promoted"])
        self.assertEqual(trainer.champion_version, 1)
        report = metrics["promotion_league"]
        self.assertEqual(report["actual_games_played"], 400)
        self.assertEqual(report["candidate_mean_win_rate"], 0.615)
        self.assertEqual(report["reference_mean_win_rate"], 0.6)
        self.assertEqual(evaluate.call_count, 5)
        # A worse result against one opponent does not veto a higher mean.
        self.assertEqual(report["results"][1]["candidate"]["wins"], 58)
        self.assertEqual(self.originals, self.snapshot())
        self.assertEqual(loop._model_digest(trainer.model), loop._model_digest(trainer.champion))

    def test_tied_and_lower_averages_do_not_replace_champion(self):
        for wins in ((51, 61), (50, 61)):
            trainer = self.trainer()
            before = loop._model_digest(trainer.champion)
            metrics, _ = self.run_cycle(trainer, [result(22, 40), result(52), result(60),
                                                result(wins[0]), result(wins[1])])
            self.assertFalse(metrics["promoted"])
            self.assertEqual(loop._model_digest(trainer.champion), before)
            self.assertEqual(trainer.champion_version, 0)

    def test_same_opponents_seeds_colors_and_search_options_for_both_actors(self):
        league = self.league()
        evaluator = Mock(return_value=result(50))
        league.compare(self.candidate, self.reference, evaluator=evaluator, **OPTIONS)
        calls = evaluator.call_args_list
        self.assertEqual([call.kwargs["seed"] for call in calls], [148, 219, 148, 219])
        self.assertTrue(all(call.kwargs == {**OPTIONS, "games": 100, "seed": seed}
            for call, seed in zip(calls, (148, 219, 148, 219))))
        self.assertIs(calls[0].args[0], self.reference)
        self.assertIs(calls[2].args[0], self.candidate)
        self.assertIs(calls[0].args[1], calls[2].args[1])

    def test_unchanged_reference_cache_survives_save_and_resume(self):
        trainer = self.trainer()
        self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        path = self.root / "latest.pt"
        trainer.save_checkpoint(path)
        restored = Trainer.load_checkpoint(path)
        self.assertEqual(restored.promotion_league.state_dict(), trainer.promotion_league.state_dict())
        metrics, evaluate = self.run_cycle(restored, [result(22, 40)] + [result(50)] * 2)
        self.assertEqual(evaluate.call_count, 3)
        self.assertTrue(metrics["promotion_league"]["reference_cache_hit"])
        self.assertEqual(metrics["promotion_league"]["actual_games_played"], 200)
        self.assertEqual(self.originals, self.snapshot())

    def test_reference_weight_or_settings_changes_invalidate_cache(self):
        league = self.league()
        evaluate = Mock(return_value=result(50))
        league.compare(self.candidate, self.reference, evaluator=evaluate, **OPTIONS)
        for changed_options in ({**OPTIONS, "workers": 2}, OPTIONS):
            evaluate.reset_mock()
            report = league.compare(self.candidate, self.reference, evaluator=evaluate, **changed_options)
            self.assertFalse(report["reference_cache_hit"])
            self.assertEqual(evaluate.call_count, 4)
        with torch.no_grad():
            next(self.reference.parameters()).add_(0.01)
        evaluate.reset_mock()
        self.assertFalse(league.compare(self.candidate, self.reference, evaluator=evaluate,
                                       **OPTIONS)["reference_cache_hit"])
        self.assertEqual(evaluate.call_count, 4)

    def test_archive_changes_or_missing_models_fail_before_learning(self):
        trainer = self.trainer()
        (self.archive / "champion_177v" / "best.pt").unlink()
        with patch.object(loop, "collect_puct_game") as collect, self.assertRaises(OSError):
            trainer.run_iteration()
        collect.assert_not_called()
        self.assertEqual(trainer.training_steps, 0)
        self.assertTrue(trainer._at_boundary)

    def test_new_archive_versions_require_explicit_reconfigure(self):
        trainer = self.trainer()
        folder = self.archive / "champion_128v"
        folder.mkdir()
        save_model(self.reference, folder / "best.pt")
        with self.assertRaisesRegex(ValueError, "changed"):
            trainer.run_iteration()
        trainer.reconfigure(promotion_archive_dir=str(self.archive))
        self.assertEqual([row["version"] for row in trainer.promotion_league.opponents], [106, 128, 177])

    def test_empty_duplicate_or_incomplete_archive_is_not_silently_ignored(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            PromotionLeague(str(empty))
        duplicate = self.archive / "champion_0106v"
        duplicate.mkdir()
        save_model(self.reference, duplicate / "best.pt")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.league()

    def test_failed_archive_comparison_never_promotes_or_saves_partial_iteration(self):
        trainer = self.trainer()
        before = loop._model_digest(trainer.champion)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_cycle(trainer, [result(22, 40), result(50), RuntimeError("injected")])
        self.assertEqual(loop._model_digest(trainer.champion), before)
        self.assertIsNone(trainer.promotion_league.reference_cache)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            trainer.save_checkpoint(self.root / "partial.pt")

    def test_invalid_config_and_cache_are_rejected(self):
        for changes in ({"promotion_archive_games": 1}, {"promotion_archive_games": 99},
                        {"promotion_archive_games": True}, {"promotion_archive_dir": ""},
                        {"promotion_archive_dir": 7}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**changes)
        league = self.league()
        league.compare(self.candidate, self.reference, evaluator=Mock(return_value=result(50)), **OPTIONS)
        state = league.state_dict()
        state["reference_cache"]["results"][0]["evaluation"]["wins"] += 1
        with self.assertRaises(ValueError):
            self.league(state=state)

    def test_version_eight_migrates_without_enabling_archive_or_changing_threshold(self):
        trainer = Trainer(TrainingConfig(promotion_threshold=0.6), self.candidate)
        path = self.root / "legacy.pt"
        trainer.save_checkpoint(path)
        state = torch.load(path, weights_only=True)
        state["checkpoint_version"] = 8
        state.pop("promotion_league")
        for name in loop._PROMOTION_CONFIG_KEYS:
            state["config"].pop(name)
        torch.save(state, path)
        restored = Trainer.load_checkpoint(path)
        self.assertIsNone(restored.config.promotion_archive_dir)
        self.assertEqual(restored.config.promotion_threshold, 0.6)
        self.assertEqual(loop._model_digest(restored.model), loop._model_digest(trainer.model))
        self.assertTrue(torch.equal(restored.generator.get_state(), trainer.generator.get_state()))

    def test_invalid_reconfigure_is_atomic_and_preparation_does_not_consume_rng(self):
        trainer = Trainer(TrainingConfig(), self.candidate)
        python_rng, cpu_rng = random.getstate(), torch.get_rng_state().clone()
        private_rng = trainer.generator.get_state().clone()
        original_config = trainer.config
        with self.assertRaises(OSError):
            trainer.reconfigure(promotion_archive_dir=str(self.root / "absent"))
        self.assertEqual(trainer.config, original_config)
        self.assertIsNone(trainer.promotion_league.directory)
        trainer.reconfigure(promotion_archive_dir=str(self.archive), promotion_threshold=0.55)
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(private_rng, trainer.generator.get_state()))

    def test_version_nine_candidate_export_still_works(self):
        trainer = self.trainer()
        path = self.root / "latest.pt"
        trainer.save_checkpoint(path)
        metadata = export_training_model(path, self.root / "export.pt")
        self.assertEqual(metadata["source_checkpoint_version"], 9)
        self.assertEqual(metadata["weights_sha256"], loop._model_digest(trainer.model))

    def test_compact_logs_and_analysis_keep_separate_primary_and_archive_results(self):
        trainer = self.trainer()
        first, _ = self.run_cycle(trainer, [result(21, 40)])
        second, _ = self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        summary = summarize_metrics([compact_metric_row(first), compact_metric_row(second)])
        league = summary["promotion_league"]
        self.assertEqual(league["enabled_iterations"], 2)
        self.assertEqual(league["head_to_head_passes"], 1)
        self.assertEqual(league["evaluated_iterations"], 1)
        self.assertEqual(league["actual_games_played"], 400)
        self.assertEqual(league["promotions"], 0)
        self.assertEqual(league["latest_comparison"]["iteration"], 2)
        self.assertEqual(league["latest_comparison"]["candidate_mean_win_rate"], 0.5)
        self.assertEqual(summary["evaluation"]["games"], 80)

    def test_actual_analysis_cli_reads_new_comparison_log_and_cache_counts(self):
        trainer = self.trainer()
        metrics_path = self.root / "metrics.jsonl"
        def evaluate(*args, **kwargs):
            return result(22, 40) if kwargs["games"] == 40 else result(50)
        with patch.object(loop, "collect_puct_game", side_effect=pass_game), \
                patch.object(loop, "evaluate_models", side_effect=evaluate):
            trainer.run(2, metrics_path=metrics_path)
        specification = importlib.util.spec_from_file_location(
            "promotion_analysis_cli", REPOSITORY / "examples" / "analyze_training.py")
        program = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(program)
        output = io.StringIO()
        with patch("sys.argv", ["analyze_training.py", str(metrics_path)]), redirect_stdout(output):
            self.assertEqual(program.main(), 0)
        self.assertIn("1차 통과 2회", output.getvalue())
        self.assertIn("실제 추가 대국 600판", output.getvalue())
        self.assertIn("현 챔피언 결과 재사용 1회", output.getvalue())
        self.assertIn("champion_106v: 후보 50.0%, 현 챔피언 50.0%", output.getvalue())

    def test_real_two_game_archive_duels_use_existing_cpp_batched_evaluator(self):
        # Only a tiny wiring smoke, not a 100-game strength experiment.
        league = self.league(games=2)
        options = {**OPTIONS, "backend": "batched_cpp", "workers": 2,
                   "opening_moves": 0, "leaf_batch_size": 2}
        before = loop._model_digest(self.reference)
        report = league.compare(self.reference, self.reference, evaluator=evaluate_models, **options)
        self.assertFalse(report["passed"])
        self.assertEqual(report["candidate_mean_win_rate"], 0.5)
        self.assertEqual(report["reference_mean_win_rate"], 0.5)
        self.assertEqual(report["actual_games_played"], 8)
        self.assertEqual(loop._model_digest(self.reference), before)
        self.assertEqual(self.originals, self.snapshot())


if __name__ == "__main__":
    unittest.main()
