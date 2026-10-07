"""Evaluation algorithm settings survive checkpoints without changing old resumes."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig
import kingdom_ai.loop as loop


class BatchedRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kingdom-batched-runtime-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "latest.pt"
        self.config = TrainingConfig(
            games_per_iteration=2, simulations=2, replay_capacity=128, batch_size=2,
            train_steps_per_iteration=1, evaluation_games=2, evaluation_simulations=4,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(57)
            self.model = PolicyValueNet(channels=4, residual_blocks=0)

    def trainer(self, **settings):
        return Trainer(self.config, self.model, evaluation_workers=2, **settings)

    def test_runtime_round_trip_and_explicit_overrides(self):
        trainer = self.trainer(evaluation_backend="batched_cpp", evaluation_leaf_batch_size=4,
                               evaluation_reuse_tree=False)
        trainer.save_checkpoint(self.path)
        self.assertEqual(torch.load(self.path, weights_only=True)["checkpoint_version"], 9)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.evaluation_workers, 2)
        self.assertEqual(restored.evaluation_backend, "batched_cpp")
        self.assertEqual(restored.evaluation_leaf_batch_size, 4)
        self.assertFalse(restored.evaluation_reuse_tree)
        overridden = Trainer.load_checkpoint(
            self.path, evaluation_workers=3, evaluation_backend="legacy",
            evaluation_leaf_batch_size=8, evaluation_reuse_tree=True,
        )
        self.assertEqual(overridden.evaluation_workers, 3)
        self.assertEqual(overridden.evaluation_backend, "legacy")
        self.assertEqual(overridden.evaluation_leaf_batch_size, 8)
        self.assertTrue(overridden.evaluation_reuse_tree)

    def test_version_four_keeps_original_algorithm_and_worker_count(self):
        self.trainer().save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload["checkpoint_version"] = 4
        payload.pop("promotion_league")
        for key in loop._PROMOTION_CONFIG_KEYS:
            payload["config"].pop(key)
        payload.pop("training_budget_history")
        payload["progress"].pop("normal_training_steps")
        for key in loop._ADAPTIVE_CONFIG_KEYS:
            payload["config"].pop(key)
        payload.pop("online_tactical_replay")
        for key in loop._ONLINE_CONFIG_KEYS:
            payload["config"].pop(key)
        payload["progress"].pop("tactical_training_steps")
        payload["runtime"] = {"evaluation_workers": 12}
        torch.save(payload, self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.evaluation_workers, 12)
        self.assertEqual(restored.evaluation_backend, "legacy")
        self.assertEqual(restored.evaluation_leaf_batch_size, 8)
        self.assertTrue(restored.evaluation_reuse_tree)
        selected = Trainer.load_checkpoint(self.path, evaluation_backend="batched_cpp")
        self.assertEqual(selected.evaluation_workers, 12)
        self.assertEqual(selected.evaluation_backend, "batched_cpp")

    def test_bad_runtime_is_rejected_even_with_valid_override(self):
        self.trainer().save_checkpoint(self.path)
        original = torch.load(self.path, weights_only=True)
        for name, value in (("evaluation_backend", "unknown"),
                            ("evaluation_leaf_batch_size", 0),
                            ("evaluation_leaf_batch_size", True),
                            ("evaluation_reuse_tree", 1)):
            payload = deepcopy(original)
            payload["runtime"][name] = value
            torch.save(payload, self.path)
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "runtime"):
                Trainer.load_checkpoint(self.path, evaluation_backend="batched_cpp")

    def test_bad_constructor_and_override_settings(self):
        self.trainer().save_checkpoint(self.path)
        for settings in ({"evaluation_backend": "unknown"}, {"evaluation_backend": 1},
                         {"evaluation_leaf_batch_size": 0}, {"evaluation_leaf_batch_size": True},
                         {"evaluation_reuse_tree": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.trainer(**settings)
            with self.subTest(override=settings), self.assertRaises(ValueError):
                Trainer.load_checkpoint(self.path, **settings)

    def test_actual_iteration_uses_batched_path_and_logs_diagnostics(self):
        trainer = self.trainer(evaluation_backend="batched_cpp", evaluation_leaf_batch_size=4)
        metrics = trainer.run_iteration()
        self.assertEqual(metrics["evaluation_backend"], "batched_cpp")
        self.assertEqual(metrics["evaluation_leaf_batch_size"], 4)
        self.assertTrue(metrics["evaluation_reuse_tree"])
        diagnostics = metrics["evaluation_diagnostics"]
        self.assertEqual(diagnostics["backend"], "batched_cpp")
        self.assertGreater(diagnostics["candidate_inference"]["network_evaluations"], 0)
        trainer.save_checkpoint(self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.iteration, trainer.iteration)
        self.assertEqual(restored._last_metrics, metrics)

    def test_version_five_defaults_extra_updates_to_zero(self):
        self.trainer().save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload["checkpoint_version"] = 5
        payload.pop("promotion_league")
        for key in loop._PROMOTION_CONFIG_KEYS:
            payload["config"].pop(key)
        payload.pop("training_budget_history")
        payload["progress"].pop("normal_training_steps")
        for key in loop._ADAPTIVE_CONFIG_KEYS:
            payload["config"].pop(key)
        payload.pop("online_tactical_replay")
        for key in loop._ONLINE_CONFIG_KEYS:
            payload["config"].pop(key)
        payload["progress"].pop("tactical_training_steps")
        torch.save(payload, self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.tactical_training_steps, 0)
        restored.save_checkpoint(self.path)
        self.assertEqual(torch.load(self.path, weights_only=True)["checkpoint_version"], 9)

    def test_extra_updates_round_trip_without_inventing_games_or_promoting(self):
        trainer = self.trainer()
        trainer.run_iteration()
        champion = deepcopy(trainer.champion.state_dict())
        batch = trainer.replay.sample(2, generator=trainer.generator)
        trainer.train_tactical_batch(batch)
        self.assertEqual((trainer.iteration, trainer.self_play_games,
                          trainer.training_steps, trainer.tactical_training_steps), (1, 2, 2, 1))
        self.assertTrue(trainer._last_metrics["tactical_finetuning_since_evaluation"])
        for name, value in champion.items():
            self.assertTrue(torch.equal(value, trainer.champion.state_dict()[name]))
        trainer.save_checkpoint(self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.tactical_training_steps, 1)
        self.assertEqual(restored.training_steps, 2)
        restored.run_iteration()
        self.assertEqual(restored.training_steps, 3)
        self.assertEqual(restored.tactical_training_steps, 1)
        self.assertEqual(restored.iteration, 2)

    def test_failed_extra_update_cannot_be_saved(self):
        trainer = self.trainer()
        with patch("kingdom_ai.loop.train_step", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                trainer.train_tactical_batch(None)
        self.assertEqual(trainer.training_steps, 0)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            trainer.save_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
