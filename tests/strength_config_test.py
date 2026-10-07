"""Legacy checkpoint migration and state-preserving strength configuration."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

import kingdom_ai.loop as loop
from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.training import GameData, TrainingSample
from kingdom_ai.encoding import ACTION_SIZE, PASS_ACTION, encode_state


def pass_game(*args, **kwargs):
    state = engine.State()
    samples = []
    for _ in range(2):
        observation = encode_state(state)
        policy = torch.zeros(ACTION_SIZE)
        policy[PASS_ACTION] = 1
        samples.append(TrainingSample(observation.features, observation.legal_mask,
                                      policy, -1.0 if state.to_play == engine.Cell.Black else 1.0,
                                      state.to_play))
        state.pass_turn()
    return GameData(samples, state.result.winner, state.result.reason)


def equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal(test, a, b)
    else:
        test.assertEqual(left, right)


class StrengthConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.config = TrainingConfig(games_per_iteration=2, simulations=2,
                                     train_steps_per_iteration=2, batch_size=4,
                                     replay_capacity=16, evaluation_games=2,
                                     evaluation_simulations=2)
        self.trainer = Trainer(self.config, PolicyValueNet(channels=8, residual_blocks=1))
        self.evaluation = EvaluationResult(2, 0, 2, 0, 0, 4, {"TwoPasses": 2})

    def run_one(self, trainer):
        with patch.object(loop, "collect_puct_game", side_effect=pass_game), \
                patch.object(loop, "evaluate_models", return_value=self.evaluation):
            trainer.run_iteration()

    def test_version_two_restores_original_behavior_then_upgrades(self):
        self.run_one(self.trainer)
        original = self.path / "v3.pt"
        self.trainer.save_checkpoint(original)
        payload = torch.load(original, weights_only=True)
        payload["checkpoint_version"] = 2
        payload["progress"].pop("tactical_training_steps")
        payload.pop("runtime")
        for key in loop._STRENGTH_CONFIG_KEYS:
            payload["config"].pop(key)
        legacy = self.path / "v2.pt"
        torch.save(payload, legacy)
        restored = Trainer.load_checkpoint(legacy)
        self.assertEqual(asdict(restored.config), asdict(self.trainer.config))
        equal(self, restored.model.state_dict(), self.trainer.model.state_dict())
        equal(self, restored.optimizer.state_dict(), self.trainer.optimizer.state_dict())
        equal(self, restored.replay.state_dict(), self.trainer.replay.state_dict())
        self.assertTrue(torch.equal(restored.generator.get_state(), self.trainer.generator.get_state()))
        restored.save_checkpoint(self.path / "upgraded.pt")
        self.assertEqual(torch.load(self.path / "upgraded.pt", weights_only=True)["checkpoint_version"], 6)

    def test_version_three_defaults_workers_and_upgrades_runtime_settings(self):
        source = self.path / "v4.pt"
        self.trainer.save_checkpoint(source)
        payload = torch.load(source, weights_only=True)
        payload["checkpoint_version"] = 3
        payload["progress"].pop("tactical_training_steps")
        payload.pop("runtime")
        legacy = self.path / "v3.pt"
        torch.save(payload, legacy)

        restored = Trainer.load_checkpoint(legacy)
        self.assertEqual(restored.evaluation_workers, 1)
        overridden = Trainer.load_checkpoint(legacy, evaluation_workers=12)
        self.assertEqual(overridden.evaluation_workers, 12)
        upgraded = self.path / "upgraded-v4.pt"
        overridden.save_checkpoint(upgraded)
        upgraded_payload = torch.load(upgraded, weights_only=True)
        self.assertEqual(upgraded_payload["checkpoint_version"], 6)
        self.assertEqual(upgraded_payload["runtime"], {
            "evaluation_workers": 12, "evaluation_backend": "legacy",
            "evaluation_leaf_batch_size": 8, "evaluation_reuse_tree": True,
        })

    def test_reconfiguration_preserves_progress_weights_replay_rng_and_adam_moments(self):
        self.run_one(self.trainer)
        before_model = deepcopy(self.trainer.model.state_dict())
        before_optimizer = deepcopy(self.trainer.optimizer.state_dict())
        before_replay = self.trainer.replay.state_dict()
        before_rng = self.trainer.generator.get_state()
        self.trainer.reconfigure(simulations=128, augment_symmetries=True,
                                 temperature_moves=8, final_temperature=0.25,
                                 learning_rate=0.0003, evaluation_games=40,
                                 evaluation_simulations=128, promotion_threshold=0.6)
        equal(self, self.trainer.model.state_dict(), before_model)
        equal(self, self.trainer.optimizer.state_dict()["state"], before_optimizer["state"])
        equal(self, self.trainer.replay.state_dict(), before_replay)
        self.assertTrue(torch.equal(before_rng, self.trainer.generator.get_state()))
        self.assertEqual((self.trainer.iteration, self.trainer.self_play_games,
                          self.trainer.training_steps), (1, 2, 2))
        self.assertEqual(self.trainer.optimizer.param_groups[0]["lr"], 0.0003)
        target = self.path / "changed.pt"
        self.trainer.save_checkpoint(target)
        restored = Trainer.load_checkpoint(target)
        self.assertEqual(asdict(restored.config), asdict(self.trainer.config))
        equal(self, restored.optimizer.state_dict(), self.trainer.optimizer.state_dict())

    def test_invalid_reconfiguration_is_atomic(self):
        original = asdict(self.trainer.config)
        optimizer = deepcopy(self.trainer.optimizer.state_dict())
        for settings in ({"games_per_iteration": 3}, {"train_steps_per_iteration": 3},
                         {"replay_capacity": 32}, {"seed": 0}, {"unknown": True},
                         {"temperature_moves": -1}, {"learning_rate": float("nan")},
                         {"self_play_tactical_checks": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.trainer.reconfigure(**settings)
            self.assertEqual(asdict(self.trainer.config), original)
            equal(self, self.trainer.optimizer.state_dict(), optimizer)
        self.trainer._at_boundary = False
        with self.assertRaises(RuntimeError):
            self.trainer.reconfigure(simulations=8)

    def test_augmented_training_matches_checkpoint_resume(self):
        self.trainer.reconfigure(augment_symmetries=True, temperature_moves=0,
                                 final_temperature=0.25)
        self.run_one(self.trainer)
        checkpoint = self.path / "middle.pt"
        self.trainer.save_checkpoint(checkpoint)
        self.run_one(self.trainer)
        restored = Trainer.load_checkpoint(checkpoint)
        self.run_one(restored)
        equal(self, self.trainer.model.state_dict(), restored.model.state_dict())
        equal(self, self.trainer.optimizer.state_dict(), restored.optimizer.state_dict())
        equal(self, self.trainer.replay.state_dict(), restored.replay.state_dict())
        self.assertTrue(torch.equal(self.trainer.generator.get_state(), restored.generator.get_state()))

    def test_strict_strength_configuration_types(self):
        for settings in ({"augment_symmetries": 1}, {"temperature_moves": True},
                         {"final_temperature": -0.1}, {"self_play_tactical_checks": 1},
                         {"self_play_backend": "cuda", "self_play_fpu_reduction": True},
                         {"self_play_backend": "cuda", "self_play_fpu_reduction": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TrainingConfig(**settings)


if __name__ == "__main__":
    unittest.main()
