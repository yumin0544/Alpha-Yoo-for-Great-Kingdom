"""Completed reinforcement iterations, exact CPU resume and checkpoint recovery."""

from dataclasses import asdict, replace
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai import (
    ACTION_SIZE, PASS_ACTION, GameData, PolicyValueNet,
    TrainingSample, encode_state, load_model,
)
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.loop import Trainer, TrainingConfig


loop = importlib.import_module(Trainer.__module__)


def pass_game():
    """Small completed game with genuine pre-move observations and labels."""
    state = engine.State()
    positions = []
    while not state.result.finished():
        encoded = encode_state(state)
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        policy[PASS_ACTION] = 1
        positions.append((encoded, policy))
        state.pass_turn()
    return GameData([
        TrainingSample(encoded.features, encoded.legal_mask, policy,
                       1.0 if encoded.to_play == state.result.winner else -1.0,
                       encoded.to_play)
        for encoded, policy in positions
    ], state.result.winner, state.result.reason)


def evaluation(wins):
    return EvaluationResult(games=2, wins=wins, losses=2 - wins,
                            wins_as_black=int(wins == 2), wins_as_white=int(wins >= 1),
                            total_plies=4, endings={"TwoPasses": 2})


class ReinforcementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kingdom-rl-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.config = TrainingConfig(
            games_per_iteration=2, simulations=2, c_puct=1.5,
            dirichlet_alpha=0.3, dirichlet_epsilon=0.25, temperature=1.0,
            replay_capacity=64, batch_size=4, train_steps_per_iteration=2,
            learning_rate=0.001, weight_decay=0.0001,
            evaluation_games=2, evaluation_simulations=2,
            evaluation_opening_moves=2, evaluation_opening_temperature=1.0,
            promotion_threshold=0.55, seed=914,
        )

    def trainer(self, evaluation_workers=1):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(127)
            model = PolicyValueNet(channels=8, residual_blocks=1)
        return Trainer(config=self.config, model=model, device="cpu",
                       evaluation_workers=evaluation_workers)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                with self.subTest(key=key):
                    self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def trainer_state(self, trainer):
        return copy.deepcopy({
            "model": trainer.model.state_dict(),
            "champion": trainer.champion.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "replay": trainer.replay.state_dict(),
            "generator": trainer.generator.get_state(),
            "iteration": trainer.iteration,
            "self_play_games": trainer.self_play_games,
            "training_steps": trainer.training_steps,
            "champion_version": trainer.champion_version,
        })

    def test_actual_selfplay_training_evaluation_and_exact_cpu_resume(self):
        initial_python = random.getstate()
        initial_torch = torch.get_rng_state().clone()
        uninterrupted = self.trainer(evaluation_workers=2)
        uninterrupted.run(2)
        expected = self.trainer_state(uninterrupted)
        next_python = [random.random() for _ in range(5)]
        next_torch = torch.rand(5)

        random.setstate(initial_python)
        torch.set_rng_state(initial_torch)
        interrupted = self.trainer(evaluation_workers=2)
        checkpoint = self.path / "resume.pt"
        interrupted.run(1, checkpoint_path=checkpoint)
        resumed = Trainer.load_checkpoint(checkpoint, device="cpu")
        self.assertEqual(resumed.evaluation_workers, 2)
        self.assertEqual(asdict(resumed.config), asdict(self.config))
        self.assert_tree_equal(self.trainer_state(interrupted), self.trainer_state(resumed))
        resumed.run(1)

        self.assertEqual(resumed.iteration, 2)
        self.assertEqual(resumed.self_play_games, 4)
        self.assertEqual(resumed.training_steps, 4)
        self.assertGreater(len(resumed.replay), 0)
        self.assertLessEqual(len(resumed.replay), self.config.replay_capacity)
        self.assert_tree_equal(expected, self.trainer_state(resumed))
        self.assertEqual(next_python, [random.random() for _ in range(5)])
        torch.testing.assert_close(next_torch, torch.rand(5), rtol=0, atol=0)

    def test_promotion_rejection_and_selfplay_uses_champion(self):
        # A perfect score equals the threshold: promotion uses >=, not >.
        self.config = replace(self.config, promotion_threshold=1.0)
        trainer = self.trainer()
        starting_champion = copy.deepcopy(trainer.champion.state_dict())
        actors = []

        def generate(model, *args, **kwargs):
            actors.append(model)
            return pass_game()

        with patch.object(loop, "collect_puct_game", side_effect=generate), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            rejected = trainer.run_iteration()
        self.assertIsInstance(rejected, dict)
        self.assertEqual(trainer.champion_version, 0)
        self.assert_tree_equal(starting_champion, trainer.champion.state_dict())
        self.assertTrue(any(not torch.equal(value, trainer.model.state_dict()[name])
                            for name, value in starting_champion.items()))
        self.assertTrue(all(actor is trainer.champion for actor in actors))

        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(2)):
            promoted = trainer.run_iteration()
        self.assertIsInstance(promoted, dict)
        self.assertEqual(trainer.champion_version, 1)
        self.assert_tree_equal(trainer.model.state_dict(), trainer.champion.state_dict())
        self.assertEqual(trainer.iteration, 2)
        self.assertEqual(trainer.training_steps, 4)

    def test_iteration_metrics_expose_replay_cost_and_data_ratios(self):
        trainer = self.trainer(evaluation_workers=3)
        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)) as evaluator:
            metrics = trainer.run_iteration()

        self.assertEqual(metrics["generated_samples"], 4)
        self.assertEqual(metrics["self_play_winners"], {"Black": 0, "White": 2})
        self.assertEqual(metrics["mean_self_play_plies"], 2)
        self.assertEqual(metrics["p95_self_play_plies"], 2)
        self.assertEqual(metrics["max_self_play_plies"], 2)
        self.assertEqual(metrics["games_per_iteration"], 2)
        self.assertEqual(metrics["replay_capacity"], 64)
        self.assertEqual(metrics["retained_new_samples"], 4)
        self.assertEqual(metrics["retained_new_sample_ratio"], 1.0)
        self.assertEqual(metrics["training_samples_drawn"], 8)
        self.assertEqual(metrics["training_draws_per_generated_sample"], 2.0)
        self.assertEqual(metrics["training_draws_per_replay_sample"], 2.0)
        self.assertEqual(metrics["evaluation_workers"], 3)
        self.assertEqual(evaluator.call_args.kwargs["workers"], 3)
        self.assertGreaterEqual(metrics["replay_store_seconds"], 0.0)
        self.assertGreater(metrics["self_play_positions_per_second"], 0.0)
        self.assertGreaterEqual(metrics["replay_store_positions_per_second"], 0.0)

    def test_evaluation_workers_are_checkpointed_and_validated(self):
        for value in (0, -1, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.trainer(evaluation_workers=value)

        trainer = self.trainer(evaluation_workers=4)
        checkpoint = self.path / "workers.pt"
        trainer.save_checkpoint(checkpoint)
        self.assertEqual(Trainer.load_checkpoint(checkpoint).evaluation_workers, 4)
        self.assertEqual(
            Trainer.load_checkpoint(checkpoint, evaluation_workers=2).evaluation_workers,
            2,
        )
        with self.assertRaises(ValueError):
            Trainer.load_checkpoint(checkpoint, evaluation_workers=0)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["runtime"]["evaluation_workers"] = 0
        broken = self.path / "broken-workers.pt"
        torch.save(payload, broken)
        with self.assertRaisesRegex(ValueError, "runtime"):
            Trainer.load_checkpoint(broken)

    def test_distinct_selfplay_seeds_and_requested_noise_budget(self):
        trainer = self.trainer()
        seeds = []

        def generate(model, options=None, temperature=1.0, seed=42):
            self.assertIs(model, trainer.champion)
            self.assertEqual(options.simulations, self.config.simulations)
            self.assertEqual(options.c_puct, self.config.c_puct)
            self.assertEqual(options.dirichlet_alpha, self.config.dirichlet_alpha)
            self.assertEqual(options.dirichlet_epsilon, self.config.dirichlet_epsilon)
            self.assertEqual(options.time_limit_ms, 0)
            self.assertEqual(options.seed, seed)
            self.assertEqual(temperature, self.config.temperature)
            seeds.append(seed)
            return pass_game()

        with patch.object(loop, "collect_puct_game", side_effect=generate), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            trainer.run(2)
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(type(seed) is int and 0 <= seed < 2 ** 64 for seed in seeds))

    def test_run_counts_additional_iterations_writes_jsonl_and_saves_before_callback(self):
        trainer = self.trainer()
        checkpoint = self.path / "nested" / "latest.pt"
        metrics_path = self.path / "nested" / "metrics.jsonl"
        observations = []

        def after_iteration(metrics):
            restored = Trainer.load_checkpoint(checkpoint)
            observations.append((restored.iteration, copy.deepcopy(metrics)))
            self.assertIsInstance(metrics, dict)
            self.assertEqual(restored.iteration, trainer.iteration)
            self.assertNotIn("checkpoint_seconds", restored._last_metrics)

        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            trainer.run(1, checkpoint_path=checkpoint, metrics_path=metrics_path,
                        on_iteration=after_iteration)
            trainer.run(1, checkpoint_path=checkpoint, metrics_path=metrics_path,
                        on_iteration=after_iteration)
        self.assertEqual([item[0] for item in observations], [1, 2])
        self.assertEqual(trainer.iteration, 2)
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows, [item[1] for item in observations])
        for row in rows:
            self.assertTrue(row["checkpoint_written"])
            self.assertGreaterEqual(row["initial_checkpoint_seconds"], 0.0)
            self.assertGreaterEqual(row["checkpoint_seconds"], 0.0)
            self.assertGreater(row["checkpoint_bytes"], 0)
            self.assertAlmostEqual(
                row["elapsed_with_checkpoint_seconds"],
                row["elapsed_seconds"] + row["initial_checkpoint_seconds"]
                + row["checkpoint_seconds"],
            )
        self.assertEqual(Trainer.load_checkpoint(checkpoint).iteration, 2)

    def test_streamed_run_avoids_accumulating_metrics_and_keeps_logs_and_callbacks(self):
        trainer = self.trainer()
        metrics_path = self.path / "streamed.jsonl"
        seen = []

        def record(row):
            seen.append(row["iteration"])
            row["iteration"] = -1  # Callback receives an independent copy.

        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            result = trainer.run(3, metrics_path=metrics_path, on_iteration=record,
                                 collect_metrics=False)
        self.assertEqual(result, [])
        self.assertEqual(trainer.iteration, 3)
        self.assertEqual(seen, [1, 2, 3])
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["iteration"] for row in rows], seen)
        self.assertTrue(all(row["checkpoint_written"] is False for row in rows))
        checkpoint = self.path / "streamed.pt"
        trainer.save_checkpoint(checkpoint)
        self.assertEqual(Trainer.load_checkpoint(checkpoint).iteration, 3)

    def test_atomic_save_failure_keeps_previous_checkpoint(self):
        trainer = self.trainer()
        checkpoint = self.path / "latest.pt"
        trainer.save_checkpoint(checkpoint)
        previous = checkpoint.read_bytes()

        def fail_save(payload, file, *args, **kwargs):
            if hasattr(file, "write"):
                file.write(b"incomplete checkpoint")
            else:
                Path(file).write_bytes(b"incomplete checkpoint")
            raise OSError("simulated disk failure")

        with patch("torch.save", side_effect=fail_save), \
                self.assertRaisesRegex(OSError, "simulated disk failure"):
            trainer.save_checkpoint(checkpoint)
        self.assertEqual(checkpoint.read_bytes(), previous)
        restored = Trainer.load_checkpoint(checkpoint)
        self.assert_tree_equal(self.trainer_state(trainer), self.trainer_state(restored))
        self.assertEqual(sorted(path.name for path in self.path.iterdir()), ["latest.pt"])

    def test_failed_iteration_requires_recovery_from_previous_completed_checkpoint(self):
        trainer = self.trainer()
        checkpoint = self.path / "latest.pt"
        trainer.save_checkpoint(checkpoint)
        previous = torch.load(checkpoint, map_location="cpu", weights_only=True)
        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "train_step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                trainer.run(1, checkpoint_path=checkpoint)
        # run() saves the initial boundary again. ZIP archive names can differ
        # between atomic temporary paths, while the resumable state stays equal.
        self.assert_tree_equal(torch.load(checkpoint, map_location="cpu", weights_only=True),
                               previous)
        with self.assertRaises((ValueError, RuntimeError)):
            trainer.save_checkpoint(self.path / "partial.pt")
        self.assertFalse((self.path / "partial.pt").exists())
        restored = Trainer.load_checkpoint(checkpoint)
        self.assertEqual(restored.iteration, 0)
        self.assertEqual(restored.training_steps, 0)
        self.assertEqual(restored.self_play_games, 0)
        self.assertEqual(len(restored.replay), 0)
        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            restored.run(1)
        self.assertEqual(restored.iteration, 1)

    def test_strict_checkpoint_schema_rejects_missing_and_extra_fields(self):
        trainer = self.trainer()
        checkpoint = self.path / "source.pt"
        trainer.save_checkpoint(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertIsInstance(payload, dict)
        broken = self.path / "broken.pt"
        for field in payload:
            damaged = dict(payload)
            damaged.pop(field)
            torch.save(damaged, broken)
            with self.subTest(missing=field), self.assertRaises((ValueError, TypeError)):
                Trainer.load_checkpoint(broken)
        damaged = dict(payload, unsupported_field=True)
        torch.save(damaged, broken)
        with self.assertRaises((ValueError, TypeError)):
            Trainer.load_checkpoint(broken)

    def test_corrupt_checkpoint_components_are_rejected_without_changing_global_rng(self):
        trainer = self.trainer()
        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            trainer.run(1)
        source = self.path / "valid.pt"
        trainer.save_checkpoint(source)
        original = torch.load(source, map_location="cpu", weights_only=True)
        variants = []

        damaged = copy.deepcopy(original)
        damaged["schema"]["format_version"] = True
        variants.append(("boolean schema version", damaged))
        damaged = copy.deepcopy(original)
        damaged["config"]["simulations"] = 0
        variants.append(("invalid search budget", damaged))
        damaged = copy.deepcopy(original)
        damaged["progress"]["training_steps"] += 1
        variants.append(("inconsistent progress", damaged))
        for model_key in ("model", "champion"):
            damaged = copy.deepcopy(original)
            name = next(iter(damaged[model_key]))
            damaged[model_key][name].flatten()[0] = float("nan")
            variants.append((f"nonfinite {model_key}", damaged))
        damaged = copy.deepcopy(original)
        name = next(iter(damaged["model"]))
        damaged["model"][name] = damaged["model"][name].flatten()
        variants.append(("incorrect model shape", damaged))
        damaged = copy.deepcopy(original)
        first = next(iter(damaged["optimizer"]["state"].values()))
        first["exp_avg_sq"].flatten()[0] = -1
        variants.append(("negative Adam second moment", damaged))
        damaged = copy.deepcopy(original)
        first = next(iter(damaged["optimizer"]["state"].values()))
        first["step"] = torch.tensor(1.5)
        variants.append(("incorrect Adam step", damaged))
        damaged = copy.deepcopy(original)
        damaged["replay"]["policy"][0, PASS_ACTION] = 0
        variants.append(("unnormalized replay target", damaged))
        damaged = copy.deepcopy(original)
        damaged["generator_state"] = torch.zeros(1, dtype=torch.uint8)
        variants.append(("invalid private RNG", damaged))
        damaged = copy.deepcopy(original)
        damaged["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
        variants.append(("invalid global Torch RNG", damaged))
        damaged = copy.deepcopy(original)
        damaged["rng"]["python"] = (0, (), None)
        variants.append(("invalid global Python RNG", damaged))
        damaged = copy.deepcopy(original)
        damaged["last_metrics"]["iteration"] += 1
        variants.append(("mismatched metrics", damaged))

        broken = self.path / "broken.pt"
        for description, damaged in variants:
            torch.save(damaged, broken)
            python_before = random.getstate()
            torch_before = torch.get_rng_state().clone()
            with self.subTest(component=description), self.assertRaises((ValueError, TypeError)):
                Trainer.load_checkpoint(broken)
            self.assertEqual(random.getstate(), python_before)
            torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    def test_invalid_configuration_and_run_arguments_fail_before_work(self):
        options = {
            "games_per_iteration": (0, True, 1.5),
            "simulations": (0, -1, True),
            "replay_capacity": (0, -1),
            "batch_size": (0, False),
            "train_steps_per_iteration": (0, -1),
            "learning_rate": (0, float("nan")),
            "weight_decay": (-1, float("inf")),
            "c_puct": (0, float("nan")),
            "dirichlet_alpha": (0, -1),
            "dirichlet_epsilon": (-0.1, 1.1),
            "temperature": (-1, float("nan")),
            "evaluation_games": (1, 3),
            "evaluation_simulations": (0, True),
            "evaluation_opening_moves": (-1, True),
            "evaluation_opening_temperature": (-1, float("inf")),
            "promotion_threshold": (0.49, 1.01),
            "seed": (-1, 2 ** 64, True),
        }
        for name, values in options.items():
            for value in values:
                with self.subTest(option=name, value=value), self.assertRaises((ValueError, TypeError)):
                    TrainingConfig(**{name: value})
        trainer = self.trainer()
        with patch.object(loop, "collect_puct_game") as generate:
            for iterations in (-1, True, 1.0):
                with self.subTest(iterations=iterations), self.assertRaises((ValueError, TypeError)):
                    trainer.run(iterations)
            with self.assertRaises(TypeError):
                trainer.run(1, on_iteration="not callable")
            with self.assertRaises(TypeError):
                trainer.run(1, collect_metrics=1)
            self.assertEqual(trainer.run(0), [])
            generate.assert_not_called()

    def test_champion_export_uses_existing_inference_schema(self):
        trainer = self.trainer()
        with patch.object(loop, "collect_puct_game", return_value=pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            trainer.run(1)
        target = self.path / "models" / "champion.pt"
        trainer.export_champion(target)
        restored = load_model(target)
        self.assert_tree_equal(trainer.champion.state_dict(), restored.state_dict())
        self.assertEqual(restored.model_config, trainer.champion.model_config)
        self.assertFalse(restored.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
