"""Head-to-head-only promotion and compatibility with retired archive state."""

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

    def test_gate_equal_fifty_five_percent_promotes_without_archive_duels(self):
        trainer = self.trainer()
        metrics, evaluate = self.run_cycle(trainer, [result(22, 40), result(60), result(60),
                                                    result(65), result(58)])
        self.assertTrue(metrics["promoted"])
        self.assertEqual(trainer.champion_version, 1)
        report = metrics["promotion_league"]
        self.assertEqual(report["actual_games_played"], 0)
        self.assertFalse(report["enabled"])
        self.assertFalse(report["evaluated"])
        self.assertEqual(report["rule"], "head_to_head_only")
        self.assertEqual(evaluate.call_count, 1)
        self.assertNotIn("results", report)
        self.assertEqual(self.originals, self.snapshot())
        self.assertEqual(loop._model_digest(trainer.model), loop._model_digest(trainer.champion))

    def test_unused_archive_results_cannot_veto_current_champion_gate(self):
        for wins in ((51, 61), (50, 61)):
            trainer = self.trainer()
            before = loop._model_digest(trainer.champion)
            metrics, _ = self.run_cycle(trainer, [result(22, 40), result(52), result(60),
                                                result(wins[0]), result(wins[1])])
            self.assertTrue(metrics["promoted"])
            self.assertEqual(loop._model_digest(trainer.champion), loop._model_digest(trainer.model))
            self.assertEqual(trainer.champion_version, 1)

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

    def test_disabled_archive_survives_save_and_resume(self):
        trainer = self.trainer()
        self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        path = self.root / "latest.pt"
        trainer.save_checkpoint(path)
        restored = Trainer.load_checkpoint(path)
        self.assertEqual(restored.promotion_league.state_dict(), trainer.promotion_league.state_dict())
        metrics, evaluate = self.run_cycle(restored, [result(22, 40)] + [result(50)] * 2)
        self.assertEqual(evaluate.call_count, 1)
        self.assertIsNone(restored.promotion_league.reference_cache)
        self.assertEqual(metrics["promotion_league"]["actual_games_played"], 0)
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

    def test_archive_changes_or_missing_models_do_not_block_learning(self):
        trainer = self.trainer()
        (self.archive / "champion_177v" / "best.pt").unlink()
        metrics, evaluate = self.run_cycle(trainer, [result(22, 40)])
        self.assertEqual(evaluate.call_count, 1)
        self.assertTrue(metrics["promoted"])
        self.assertEqual(trainer.training_steps, 1)
        self.assertTrue(trainer._at_boundary)

    def test_new_archive_versions_are_ignored_and_cannot_be_reactivated(self):
        trainer = self.trainer()
        folder = self.archive / "champion_128v"
        folder.mkdir()
        save_model(self.reference, folder / "best.pt")
        metrics, evaluate = self.run_cycle(trainer, [result(22, 40)])
        self.assertTrue(metrics["promoted"])
        self.assertEqual(evaluate.call_count, 1)
        trainer.reconfigure(promotion_archive_dir=str(self.archive))
        self.assertIsNone(trainer.config.promotion_archive_dir)
        self.assertEqual(trainer.promotion_league.opponents, [])

    def test_legacy_active_checkpoint_preserves_training_and_discards_archive_cache(self):
        trainer = self.trainer()
        self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        source = self.root / "latest.pt"
        trainer.save_checkpoint(source)
        old = torch.load(source, weights_only=True)
        # Emulate the active archive fields/cache stored before retirement.
        league = self.league()
        league.compare(self.candidate, self.reference, evaluator=Mock(return_value=result(50)), **OPTIONS)
        old["config"]["promotion_archive_dir"] = str(self.archive)
        old["promotion_league"] = league.state_dict()
        torch.save(old, source)
        original_bytes = source.read_bytes()
        added = self.archive / "champion_185v"
        added.mkdir()
        save_model(self.reference, added / "best.pt")
        archive_snapshot = self.snapshot()
        with patch.object(PromotionLeague, "_discover", side_effect=AssertionError("No archive reads")), \
             patch("kingdom_ai.promotion.load_model", side_effect=AssertionError("No old model loads")), \
             patch("kingdom_ai.promotion._file_digest", side_effect=AssertionError("No archive hashes")):
            restored = Trainer.load_checkpoint(source)
        self.assertEqual(restored.promotion_league.opponents, [])
        self.assertIsNone(restored.promotion_league.reference_cache)
        prepared = self.root / "refreshed.pt"
        restored.save_checkpoint(prepared)
        new = torch.load(prepared, weights_only=True)

        def same(left, right):
            self.assertEqual(type(left), type(right))
            if isinstance(left, torch.Tensor):
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.shape, right.shape)
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    same(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for before, after in zip(left, right):
                    same(before, after)
            else:
                self.assertEqual(left, right)

        for key in old.keys() - {"promotion_league", "config"}:
            same(old[key], new[key])
        for key in old["config"].keys() - {"promotion_archive_dir"}:
            same(old["config"][key], new["config"][key])
        self.assertIsNone(new["config"]["promotion_archive_dir"])
        self.assertEqual(new["promotion_league"], {"version": 1, "opponents": [], "reference_cache": None})
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual(self.snapshot(), archive_snapshot)
        # No old/new pool can require further games or veto the primary result.
        metrics, evaluate = self.run_cycle(restored, [result(22, 40)] + [result(50)] * 6)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(metrics["promotion_league"]["opponent_count"], 0)
        self.assertEqual(metrics["promotion_league"]["actual_games_played"], 0)

    def test_removed_or_replaced_pinned_opponents_do_not_block_resume(self):
        trainer = self.trainer()
        source = self.root / "latest.pt"
        trainer.save_checkpoint(source)
        state = torch.load(source, weights_only=True)
        state["config"]["promotion_archive_dir"] = str(self.archive)
        state["promotion_league"] = self.league().state_dict()
        torch.save(state, source)
        added = self.archive / "champion_185v"
        added.mkdir()
        save_model(self.reference, added / "best.pt")
        old_path = self.archive / "champion_106v" / "best.pt"
        save_model(self.candidate, old_path)
        self.assertIsNone(Trainer.load_checkpoint(source).config.promotion_archive_dir)
        old_path.unlink()
        self.assertIsNone(Trainer.load_checkpoint(source, refresh_promotion_archive=True).config.promotion_archive_dir)

    def test_legacy_refresh_argument_is_a_noop_but_keeps_strict_type_validation(self):
        trainer = self.trainer()
        self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        source = self.root / "latest.pt"
        trainer.save_checkpoint(source)
        restored = Trainer.load_checkpoint(source, refresh_promotion_archive=True)
        self.assertEqual(restored.promotion_league.state_dict(), trainer.promotion_league.state_dict())
        with self.assertRaises(ValueError):
            Trainer.load_checkpoint(source, refresh_promotion_archive=1)
        disabled = Trainer(TrainingConfig(), self.candidate)
        disabled.save_checkpoint(source)
        self.assertIsNone(Trainer.load_checkpoint(source, refresh_promotion_archive=True).config.promotion_archive_dir)

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

    def test_archive_comparison_is_never_called_and_primary_failure_is_atomic(self):
        trainer = self.trainer()
        before = loop._model_digest(trainer.champion)
        with patch.object(PromotionLeague, "compare", side_effect=AssertionError("No archive games")), \
             patch.object(PromotionLeague, "check_unchanged", side_effect=AssertionError("No archive checks")):
            metrics, evaluate = self.run_cycle(trainer, [result(22, 40)])
        self.assertTrue(metrics["promoted"])
        self.assertEqual(evaluate.call_count, 1)
        trainer = self.trainer()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.run_cycle(trainer, [RuntimeError("injected")])
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
        state.pop("capture_defense_replay")
        for name in loop._CAPTURE_DEFENSE_CONFIG_KEYS:
            state["config"].pop(name)
        state.pop("promotion_league")
        for name in loop._PROMOTION_CONFIG_KEYS:
            state["config"].pop(name)
        torch.save(state, path)
        restored = Trainer.load_checkpoint(path)
        self.assertIsNone(restored.config.promotion_archive_dir)
        self.assertEqual(restored.config.promotion_threshold, 0.6)
        self.assertEqual(loop._model_digest(restored.model), loop._model_digest(trainer.model))
        self.assertTrue(torch.equal(restored.generator.get_state(), trainer.generator.get_state()))

    def test_active_v9_and_v10_resume_with_missing_archive_and_custom_threshold(self):
        trainer = Trainer(TrainingConfig(promotion_threshold=0.6), self.candidate)
        path = self.root / "legacy-active.pt"
        trainer.save_checkpoint(path)
        base = torch.load(path, weights_only=True)
        league = self.league()
        league.compare(self.candidate, self.reference, evaluator=Mock(return_value=result(50)), **OPTIONS)
        for version in (9, 10):
            with self.subTest(version=version):
                payload = torch.load(path, weights_only=True) if version == 9 else base
                payload["checkpoint_version"] = version
                payload["config"]["promotion_archive_dir"] = str(self.root / "missing-archive")
                payload["promotion_league"] = league.state_dict()
                if version == 9:
                    payload.pop("capture_defense_replay")
                    for key in loop._CAPTURE_DEFENSE_CONFIG_KEYS:
                        payload["config"].pop(key)
                torch.save(payload, path)
                original = path.read_bytes()
                with patch.object(PromotionLeague, "_discover", side_effect=AssertionError("No discovery")), \
                     patch("kingdom_ai.promotion.load_model", side_effect=AssertionError("No old models")), \
                     patch("kingdom_ai.promotion._file_digest", side_effect=AssertionError("No hashing")):
                    restored = Trainer.load_checkpoint(path, refresh_promotion_archive=True)
                self.assertEqual(restored.config.promotion_threshold, 0.6)
                self.assertIsNone(restored.config.promotion_archive_dir)
                self.assertEqual(restored.promotion_league.state_dict(),
                                 {"version": 1, "opponents": [], "reference_cache": None})
                self.assertEqual(loop._model_digest(restored.model), loop._model_digest(trainer.model))
                self.assertTrue(torch.equal(restored.generator.get_state(), trainer.generator.get_state()))
                self.assertEqual(path.read_bytes(), original)

    def test_invalid_reconfigure_is_atomic_and_preparation_does_not_consume_rng(self):
        trainer = Trainer(TrainingConfig(), self.candidate)
        python_rng, cpu_rng = random.getstate(), torch.get_rng_state().clone()
        private_rng = trainer.generator.get_state().clone()
        original_config = trainer.config
        with self.assertRaises(ValueError):
            trainer.reconfigure(promotion_threshold=0.4)
        self.assertEqual(trainer.config, original_config)
        self.assertIsNone(trainer.promotion_league.directory)
        trainer.reconfigure(promotion_archive_dir=str(self.root / "absent"))
        trainer.reconfigure(promotion_archive_dir=str(self.archive), promotion_threshold=0.55)
        self.assertIsNone(trainer.config.promotion_archive_dir)
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(private_rng, trainer.generator.get_state()))

    def test_version_nine_candidate_export_still_works(self):
        trainer = self.trainer()
        path = self.root / "latest.pt"
        trainer.save_checkpoint(path)
        metadata = export_training_model(path, self.root / "export.pt")
        self.assertEqual(metadata["source_checkpoint_version"], 10)
        self.assertEqual(metadata["weights_sha256"], loop._model_digest(trainer.model))

    def test_compact_logs_and_analysis_report_zero_archive_games(self):
        trainer = self.trainer()
        first, _ = self.run_cycle(trainer, [result(21, 40)])
        second, _ = self.run_cycle(trainer, [result(22, 40)] + [result(50)] * 4)
        summary = summarize_metrics([compact_metric_row(first), compact_metric_row(second)])
        league = summary["promotion_league"]
        self.assertEqual(league["enabled_iterations"], 0)
        self.assertEqual(league["head_to_head_passes"], 0)
        self.assertEqual(league["evaluated_iterations"], 0)
        self.assertEqual(league["actual_games_played"], 0)
        self.assertEqual(league["promotions"], 0)
        self.assertIsNone(league["latest_comparison"])
        self.assertEqual(summary["evaluation"]["promotions"], 1)
        self.assertEqual(summary["evaluation"]["games"], 80)

    def test_actual_analysis_cli_reports_only_primary_duels(self):
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
        self.assertNotIn("과거 버전 승격 평가", output.getvalue())
        self.assertNotIn("champion_106v", output.getvalue())
        self.assertIn("승격 2/2", output.getvalue())

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
