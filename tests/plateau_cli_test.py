"""Actor, defense and changed-budget CLI wiring without running real training."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kingdom_ai import Trainer, TrainingConfig, load_model


REPOSITORY = Path(__file__).resolve().parents[1]
SPECIFICATION = importlib.util.spec_from_file_location(
    "kingdom_plateau_train_cli", REPOSITORY / "examples" / "train.py")
PROGRAM = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(PROGRAM)


class RecordingTrainer:
    """Exercise the real CLI/parser/callback but never generate games or update weights."""

    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.iteration = 515
        self.self_play_games = 515 * self.config.games_per_iteration
        self.evaluation_workers = 12
        self.evaluation_backend = "batched_cpp"
        self.evaluation_leaf_batch_size = 8
        self.evaluation_reuse_tree = True
        self.events = []
        self.overrides = None
        self.save_error = None

    def reconfigure(self, **overrides):
        self.config = replace(self.config, **overrides)
        self.overrides = overrides
        self.events.append(("reconfigure", overrides))

    def save_checkpoint(self, path):
        self.events.append(("checkpoint", Path(path)))
        if self.save_error is not None:
            raise self.save_error

    def export_champion(self, path):
        self.events.append(("champion", Path(path)))

    def export_learner(self, path):
        self.events.append(("learner", Path(path)))

    def run(self, iterations, *, checkpoint_path, metrics_path, on_iteration, collect_metrics):
        self.events.append(("run", iterations, Path(metrics_path), collect_metrics))
        self.save_checkpoint(checkpoint_path)
        # A fabricated completed metrics row checks export behavior even when
        # promotion fails. No Trainer.run_iteration(), game or optimizer runs.
        row = {
            "iteration": 516, "self_play_games": self.self_play_games,
            "replay_size": 16, "loss": 1.0,
            "evaluation": {"wins": 0, "games": 2, "win_rate": 0.0},
            "promoted": False, "self_play_backend": "cpu",
            "self_play_games_per_second": 1.0, "iteration_games_per_second": 0.5,
            "self_play_positions_per_second": 2.0, "self_play_compute_seconds": 1.0,
            "replay_store_seconds": 0.1, "training_seconds": 0.2,
            "evaluation_seconds": 0.3, "checkpoint_seconds": 0.1,
            "elapsed_with_checkpoint_seconds": 1.7,
        }
        on_iteration(row)


class PlateauCLITest(unittest.TestCase):
    def setUp(self):
        # Workspace fixtures also work under Windows restricted temp permissions.
        fixture_root = REPOSITORY / "runs"
        fixture_root.mkdir(exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="plateau_cli_test_", dir=fixture_root)
        self.addCleanup(self.workspace.cleanup)
        self.path = Path(self.workspace.name)
        self.output = self.path / "new_output"
        self.source = self.path / "source" / "latest.pt"
        self.trainer = RecordingTrainer()

    def invoke(self, *arguments, trainer=None, constructor=None):
        trainer = trainer or self.trainer
        output, error = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["train.py", "--output", str(self.output), *map(str, arguments)]), \
                patch.object(PROGRAM, "Trainer") as trainer_type, \
                patch.object(PROGRAM, "PolicyValueNet"), \
                patch.object(PROGRAM.torch, "set_num_threads"), \
                redirect_stdout(output), redirect_stderr(error):
            trainer_type.load_checkpoint.return_value = trainer

            def create(config, **kwargs):
                trainer.config = config
                if constructor is not None:
                    constructor(config, kwargs)
                return trainer

            trainer_type.side_effect = create
            try:
                code = PROGRAM.main()
            except SystemExit as exception:
                code = exception.code
            return code, output.getvalue(), error.getvalue(), trainer_type

    def test_default_actor_prepare_exports_champion_and_candidate_after_checkpoint(self):
        code, output, error, _ = self.invoke("--prepare-only")
        self.assertEqual(code, 0, error)
        self.assertEqual(self.trainer.config.self_play_model, "champion")
        self.assertFalse(self.trainer.config.online_tactics_include_loss)
        self.assertEqual(self.trainer.events, [
            ("checkpoint", self.output / "latest.pt"),
            ("champion", self.output / "best.pt"),
            ("learner", self.output / "candidate.pt"),
        ])
        self.assertIn("자료 생성 모델 champion", output)
        self.assertIn("학습 후보 파일 (승격과 별개)", output)
        self.assertIn("추가 학습은 시작하지 않았습니다", output)

    def test_new_online_training_defaults_to_twenty_ply_cap_with_existing_budgets(self):
        code, output, error, _ = self.invoke("--prepare-only", "--online-tactics")
        self.assertEqual(code, 0, error)
        config = self.trainer.config
        self.assertEqual(config.online_tactics_max_depth, 20)
        self.assertEqual(config.online_tactics_min_proof_depth, 3)
        self.assertEqual(config.online_tactics_max_nodes, 2000000)
        self.assertEqual(config.online_tactics_time_limit_ms, 2000)
        self.assertEqual(config.online_tactics_generation_seconds, 30.0)
        self.assertIn('"online_tactics_max_depth": 20', output)

    def test_resume_preserves_saved_nine_ply_cap_until_explicit_reconfigure(self):
        self.trainer.config = replace(self.trainer.config, online_tactics=True,
                                      online_tactics_max_depth=9)
        code, output, error, _ = self.invoke("--resume", self.source, "--prepare-only")
        self.assertEqual(code, 0, error)
        self.assertIsNone(self.trainer.overrides)
        self.assertEqual(self.trainer.config.online_tactics_max_depth, 9)
        self.assertIn('"online_tactics_max_depth": 9', output)

    def test_resume_reconfigure_forwards_only_explicit_twenty_ply_cap(self):
        self.trainer.config = replace(self.trainer.config, online_tactics=True,
                                      online_tactics_max_depth=9)
        saved_config = self.trainer.config
        saved_counters = (self.trainer.iteration, self.trainer.self_play_games)
        code, _, error, trainer_type = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only",
            "--online-tactics-max-depth", "20")
        self.assertEqual(code, 0, error)
        trainer_type.load_checkpoint.assert_called_once()
        self.assertEqual(self.trainer.overrides, {"online_tactics_max_depth": 20})
        self.assertEqual(self.trainer.config, replace(saved_config, online_tactics_max_depth=20))
        self.assertEqual((self.trainer.iteration, self.trainer.self_play_games), saved_counters)
        self.assertFalse(any(event[0] == "run" for event in self.trainer.events))

    def test_explicit_learner_defense_and_five_percent_are_config_fields(self):
        code, output, error, _ = self.invoke(
            "--prepare-only", "--self-play-model", "learner", "--online-tactics",
            "--online-tactics-include-loss", "--online-tactics-fraction", "0.05",
            "--train-steps", "256", "--batch-size", "512")
        self.assertEqual(code, 0, error)
        config = self.trainer.config
        self.assertEqual(config.self_play_model, "learner")
        self.assertTrue(config.online_tactics_include_loss)
        self.assertEqual(config.online_tactics_fraction, 0.05)
        self.assertEqual(config.train_steps_per_iteration, 256)
        self.assertIn("자료 생성 모델 learner", output)

    def test_resume_reconfigure_passes_changed_updates_actor_and_loss_toggle(self):
        code, _, error, trainer_type = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only",
            "--self-play-model", "learner", "--train-steps", "256",
            "--learning-rate", "3e-5", "--online-tactics-fraction", "0.05",
            "--online-tactics-include-loss")
        self.assertEqual(code, 0, error)
        self.assertEqual(self.trainer.overrides, {
            "self_play_model": "learner", "train_steps_per_iteration": 256,
            "learning_rate": 3e-5, "online_tactics_fraction": 0.05,
            "online_tactics_include_loss": True,
        })
        trainer_type.load_checkpoint.assert_called_once()
        self.assertEqual(self.trainer.iteration, 515)
        self.assertFalse(any(event[0] == "run" for event in self.trainer.events))

    def test_resume_without_flags_preserves_saved_actor_and_defense_setting(self):
        self.trainer.config = replace(self.trainer.config, self_play_model="learner",
                                      online_tactics_include_loss=True,
                                      train_steps_per_iteration=256)
        code, _, error, _ = self.invoke("--resume", self.source, "--prepare-only")
        self.assertEqual(code, 0, error)
        self.assertIsNone(self.trainer.overrides)
        self.assertEqual(self.trainer.config.self_play_model, "learner")
        self.assertTrue(self.trainer.config.online_tactics_include_loss)
        self.assertEqual(self.trainer.config.train_steps_per_iteration, 256)

    def test_resume_doubles_only_future_updates_from_256_to_512(self):
        self.trainer.config = replace(
            self.trainer.config, train_steps_per_iteration=256,
            replay_capacity=131072, games_per_iteration=2048, batch_size=512,
            learning_rate=3e-5, self_play_model="learner", online_tactics=True,
            online_tactics_fraction=0.05, online_tactics_max_depth=20,
            online_tactics_include_loss=True)
        saved_config = self.trainer.config
        self.trainer.iteration = 575
        self.trainer.self_play_games = 575 * saved_config.games_per_iteration
        code, _, error, _ = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only", "--train-steps", "512")
        self.assertEqual(code, 0, error)
        self.assertEqual(self.trainer.overrides, {"train_steps_per_iteration": 512})
        self.assertEqual(self.trainer.config, replace(saved_config, train_steps_per_iteration=512))
        self.assertEqual((self.trainer.iteration, self.trainer.self_play_games), (575, 1177600))
        self.assertFalse(any(event[0] == "run" for event in self.trainer.events))

    def test_resume_forwards_fifty_five_percent_and_historical_pool_without_learning(self):
        code, output, error, _ = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only",
            "--promotion-threshold", "0.55", "--promotion-archive-dir", self.path / "past-version",
            "--promotion-archive-games", "100")
        self.assertEqual(code, 0, error)
        self.assertEqual(self.trainer.config.promotion_threshold, 0.55)
        self.assertEqual(self.trainer.config.promotion_archive_games, 100)
        self.assertEqual(self.trainer.config.promotion_archive_dir, str((self.path / "past-version").resolve()))
        self.assertIn("동률 유지", output)
        self.assertFalse(any(event[0] == "run" for event in self.trainer.events))

    def test_cli_never_writes_into_past_version(self):
        protected = self.path / "past-version" / "new-run"
        code, _, error, trainer_type = self.invoke("--output", protected, "--prepare-only")
        self.assertEqual(code, 2)
        self.assertIn("읽기 전용", error)
        trainer_type.assert_not_called()
        self.assertFalse(protected.exists())

    def test_explicit_archive_refresh_is_forwarded_only_in_a_separate_reconfigure_run(self):
        self.trainer.config = replace(self.trainer.config,
            promotion_archive_dir=str(self.path / "past-version"))
        saved_config = self.trainer.config
        code, output, error, trainer_type = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only", "--refresh-promotion-archive")
        self.assertEqual(code, 0, error)
        self.assertTrue(trainer_type.load_checkpoint.call_args.kwargs["refresh_promotion_archive"])
        self.assertEqual(self.trainer.config, saved_config)
        self.assertIn("상대 목록을 갱신", output)
        self.assertFalse(any(event[0] == "run" for event in self.trainer.events))

    def test_archive_refresh_requires_resume_and_explicit_reconfiguration(self):
        for options in ((), ("--resume", self.source), ("--reconfigure",)):
            with self.subTest(options=options):
                code, _, error, trainer_type = self.invoke("--refresh-promotion-archive", *options)
                self.assertEqual(code, 2)
                self.assertIn("--resume", error)
                trainer_type.assert_not_called()

    def test_explicit_negative_loss_flag_restores_win_only_control(self):
        self.trainer.config = replace(self.trainer.config, online_tactics_include_loss=True)
        code, _, error, _ = self.invoke(
            "--resume", self.source, "--reconfigure", "--prepare-only",
            "--self-play-model", "champion", "--no-online-tactics-include-loss")
        self.assertEqual(code, 0, error)
        self.assertFalse(self.trainer.config.online_tactics_include_loss)
        self.assertEqual(self.trainer.config.self_play_model, "champion")

    def test_resume_config_override_requires_explicit_reconfigure(self):
        for option in (("--self-play-model", "learner"), ("--train-steps", "256"),
                       ("--online-tactics-include-loss",),
                       ("--online-tactics-max-depth", "20")):
            with self.subTest(option=option):
                code, _, error, trainer_type = self.invoke("--resume", self.source, *option)
                self.assertEqual(code, 2)
                self.assertIn("변경할 수 없습니다", error)
                trainer_type.load_checkpoint.assert_not_called()

    def test_existing_candidate_alone_is_not_overwritten(self):
        self.output.mkdir()
        candidate = self.output / "candidate.pt"
        candidate.write_bytes(b"preserve original candidate")
        for options in ((), ("--resume", self.source),
                        ("--resume", self.source, "--reconfigure")):
            with self.subTest(options=options):
                code, _, _, trainer_type = self.invoke("--prepare-only", *options)
                self.assertEqual(code, 2)
                trainer_type.assert_not_called()
                trainer_type.load_checkpoint.assert_not_called()
                self.assertEqual(candidate.read_bytes(), b"preserve original candidate")

    def test_completed_iteration_exports_candidate_even_without_promotion(self):
        code, output, error, _ = self.invoke("--resume", self.source, "--iterations", "1")
        self.assertEqual(code, 0, error)
        self.assertEqual([event[0] for event in self.trainer.events],
                         ["run", "checkpoint", "champion", "learner"])
        self.assertEqual(self.trainer.events[0][-1], False)
        self.assertIn("승격 False", output)

    def test_initial_checkpoint_failure_does_not_export_either_model(self):
        self.trainer.save_error = OSError("injected initial save failure")
        with self.assertRaisesRegex(OSError, "injected initial save failure"):
            self.invoke("--prepare-only")
        self.assertEqual(self.trainer.events, [("checkpoint", self.output / "latest.pt")])

    def test_reconfigure_requires_separate_output_from_source(self):
        self.source.parent.mkdir()
        self.source.write_bytes(b"checkpoint sentinel")
        code, _, error, trainer_type = self.invoke(
            "--output", self.source.parent, "--resume", self.source,
            "--reconfigure", "--prepare-only", "--train-steps", "256")
        self.assertEqual(code, 2)
        self.assertIn("새 --output", error)
        trainer_type.load_checkpoint.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"checkpoint sentinel")

    def test_invalid_actor_or_budget_is_rejected_before_constructing_trainer(self):
        for options in (("--self-play-model", "invalid"), ("--train-steps", "0")):
            with self.subTest(options=options):
                code, _, _, trainer_type = self.invoke(*options)
                self.assertEqual(code, 2)
                trainer_type.assert_not_called()

    def test_real_prepare_and_budget_resume_persist_without_any_learning(self):
        """Only create tiny empty CPU boundary checkpoints; run no real cycle."""
        first = self.path / "actual_prepare"
        changed = self.path / "actual_changed"
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        output = io.StringIO()
        arguments = [
            "train.py", "--device", "cpu", "--prepare-only", "--channels", "4",
            "--residual-blocks", "0", "--games-per-iteration", "2",
            "--simulations", "1", "--eval-games", "2", "--eval-simulations", "1",
            "--replay-capacity", "16", "--train-steps", "2", "--batch-size", "4",
            "--output", str(first),
        ]
        with patch("sys.argv", arguments), redirect_stdout(output):
            self.assertEqual(PROGRAM.main(), 0)
        source = first / "latest.pt"
        before = source.read_bytes()
        for name in ("best.pt", "candidate.pt"):
            exported = load_model(first / name)
            self.assertEqual(exported.model_config, {"channels": 4, "residual_blocks": 0})
        with patch("sys.argv", [
                "train.py", "--resume", str(source), "--device", "cpu", "--reconfigure",
                "--prepare-only", "--self-play-model", "learner", "--train-steps", "256",
                "--online-tactics-include-loss", "--online-tactics-fraction", "0.05",
                "--learning-rate", "3e-5", "--output", str(changed),
        ]), redirect_stdout(output):
            self.assertEqual(PROGRAM.main(), 0)
        restored = Trainer.load_checkpoint(changed / "latest.pt")
        self.assertEqual((restored.iteration, restored.self_play_games, restored.training_steps,
                          restored.normal_training_steps), (0, 0, 0, 0))
        self.assertEqual(restored.config.self_play_model, "learner")
        self.assertTrue(restored.config.online_tactics_include_loss)
        self.assertEqual(restored.config.online_tactics_fraction, 0.05)
        self.assertEqual(restored.config.train_steps_per_iteration, 256)
        self.assertEqual(restored.optimizer.param_groups[0]["lr"], 3e-5)
        self.assertEqual(restored.training_budget_history,
                         [{"start_iteration": 0, "train_steps_per_iteration": 256}])
        self.assertEqual(len(restored.replay), 0)
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse((first / "metrics.jsonl").exists())
        self.assertFalse((changed / "metrics.jsonl").exists())
        self.assertTrue((changed / "candidate.pt").is_file())

    def test_real_nine_to_twenty_depth_prepare_preserves_all_training_state(self):
        """Save tiny CPU checkpoints only; no games, solver or Adam update."""
        initial = self.path / "depth_nine"
        resumed = self.path / "depth_nine_resumed"
        changed = self.path / "depth_twenty"
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        output = io.StringIO()
        with patch("sys.argv", [
                "train.py", "--device", "cpu", "--prepare-only", "--channels", "4",
                "--residual-blocks", "0", "--games-per-iteration", "2",
                "--simulations", "1", "--eval-games", "2", "--eval-simulations", "1",
                "--replay-capacity", "16", "--train-steps", "2", "--batch-size", "4",
                "--online-tactics", "--online-tactics-max-depth", "9",
                "--output", str(initial),
        ]), redirect_stdout(output):
            self.assertEqual(PROGRAM.main(), 0)
        source = initial / "latest.pt"
        source_bytes = source.read_bytes()
        saved = torch.load(source, weights_only=True)
        with patch("sys.argv", [
                "train.py", "--resume", str(source), "--device", "cpu",
                "--prepare-only", "--output", str(resumed),
        ]), redirect_stdout(output):
            self.assertEqual(PROGRAM.main(), 0)
        self.assertEqual(Trainer.load_checkpoint(resumed / "latest.pt")
                         .config.online_tactics_max_depth, 9)
        with patch("sys.argv", [
                "train.py", "--resume", str(source), "--device", "cpu", "--reconfigure",
                "--prepare-only", "--online-tactics-max-depth", "20",
                "--output", str(changed),
        ]), redirect_stdout(output):
            self.assertEqual(PROGRAM.main(), 0)
        restored = Trainer.load_checkpoint(changed / "latest.pt")
        self.assertEqual(restored.config.online_tactics_max_depth, 20)
        self.assertEqual(restored.config, replace(TrainingConfig(**saved["config"]),
                                                 online_tactics_max_depth=20))
        changed_payload = torch.load(changed / "latest.pt", weights_only=True)

        def assert_same_state(left, right):
            if isinstance(left, torch.Tensor):
                self.assertIsInstance(right, torch.Tensor)
                self.assertEqual(left.dtype, right.dtype)
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    assert_same_state(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(type(left), type(right))
                self.assertEqual(len(left), len(right))
                for before, after in zip(left, right):
                    assert_same_state(before, after)
            else:
                self.assertEqual(left, right)

        for field in ("progress", "training_budget_history", "optimizer", "generator_state",
                      "rng", "model", "champion", "replay", "online_tactical_replay", "runtime"):
            with self.subTest(field=field):
                assert_same_state(saved[field], changed_payload[field])
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertFalse(any((folder / "metrics.jsonl").exists()
                             for folder in (initial, resumed, changed)))


if __name__ == "__main__":
    unittest.main()
