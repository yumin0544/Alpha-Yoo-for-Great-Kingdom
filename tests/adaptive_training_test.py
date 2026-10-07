"""Frozen latest actors and honest resume counters across update-budget changes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

import kingdom_ai.loop as loop
from kingdom_ai.checkpoint import load_model
from kingdom_ai.encoding import encode_state
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.loop import Trainer, TrainingConfig
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.tactical_training import CertifiedTacticalSample
from kingdom_ai.training import GameData, TrainingSample


def pass_game():
    state, samples = engine.State(), []
    for _ in range(2):
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[81] = 1
        samples.append(TrainingSample(encoded.features, encoded.legal_mask, policy,
                                     -1.0 if encoded.to_play == engine.Cell.Black else 1.0,
                                     encoded.to_play))
        state.pass_turn()
    return GameData(samples, state.result.winner, state.result.reason, (81, 81))


REJECTED = EvaluationResult(2, 0, 2, 0, 0, 4, {"TwoPasses": 2})


class AdaptiveTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kingdom-adaptive-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "latest.pt"

    def trainer(self, **changes):
        config = TrainingConfig(games_per_iteration=2, simulations=2, replay_capacity=32,
                                batch_size=4, train_steps_per_iteration=2,
                                evaluation_games=2, evaluation_simulations=2)
        return Trainer(replace(config, **changes), PolicyValueNet(channels=4, residual_blocks=0))

    def run_one(self, trainer):
        with patch.object(loop, "collect_puct_game", side_effect=lambda *a, **k: pass_game()), \
                patch.object(loop, "evaluate_models", return_value=REJECTED):
            return trainer.run_iteration()

    def test_latest_actor_is_frozen_across_games_and_not_reverted_by_rejection(self):
        trainer = self.trainer(self_play_model="learner")
        with torch.no_grad():
            next(trainer.model.parameters()).add_(0.1)
        original_champion = loop._model_digest(trainer.champion)
        actors, hashes = [], []
        def collect(actor, **kwargs):
            self.assertIsNot(actor, trainer.model)
            self.assertIsNot(actor, trainer.champion)
            self.assertFalse(actor.training)
            actors.append(actor)
            hashes.append(loop._model_digest(actor))
            return pass_game()
        starting_learner = loop._model_digest(trainer.model)
        with patch.object(loop, "collect_puct_game", side_effect=collect), \
                patch.object(loop, "evaluate_models", return_value=REJECTED):
            first = trainer.run_iteration()
            updated_learner = loop._model_digest(trainer.model)
            second = trainer.run_iteration()
        self.assertIs(actors[0], actors[1])
        self.assertIs(actors[2], actors[3])
        self.assertIsNot(actors[0], actors[2])
        self.assertEqual(hashes, [starting_learner] * 2 + [updated_learner] * 2)
        self.assertNotEqual(starting_learner, updated_learner)
        self.assertEqual(first["self_play_actor"]["weights_sha256"], starting_learner)
        self.assertEqual(second["self_play_actor"]["training_steps"], 2)
        self.assertEqual(loop._model_digest(trainer.champion), original_champion)
        self.assertEqual(trainer.champion_version, 0)

    def test_v7_migration_preserves_adam_and_allows_future_budget_change(self):
        trainer = self.trainer()
        self.run_one(trainer)
        trainer.train_tactical_batch(trainer.replay.sample(2, generator=trainer.generator))
        trainer.save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload["checkpoint_version"] = 7
        payload.pop("training_budget_history")
        payload["progress"].pop("normal_training_steps")
        payload["last_metrics"].pop("normal_training_steps")
        for name in loop._ADAPTIVE_CONFIG_KEYS:
            payload["config"].pop(name)
        # Legacy proof-replay version 1 is also accepted.
        payload["online_tactical_replay"]["version"] = 1
        payload["online_tactical_replay"].pop("policy_enabled")
        torch.save(payload, self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.config.self_play_model, "champion")
        self.assertFalse(restored.config.online_tactics_include_loss)
        self.assertEqual(restored.normal_training_steps, 2)
        self.assertEqual(restored.tactical_training_steps, 1)
        before_moments = deepcopy(restored.optimizer.state_dict()["state"])
        before_rng = restored.generator.get_state().clone()
        restored.reconfigure(train_steps_per_iteration=4, self_play_model="learner")
        self.assertEqual(restored.training_steps, 3)
        self.assertTrue(torch.equal(before_rng, restored.generator.get_state()))
        for index, state in before_moments.items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, restored.optimizer.state_dict()["state"][index][key]))
        restored.save_checkpoint(self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.run_one(restored)
        self.assertEqual((restored.iteration, restored.normal_training_steps,
                          restored.tactical_training_steps, restored.training_steps), (2, 6, 1, 7))
        self.assertEqual(restored.training_budget_history,
                         [{"start_iteration": 0, "train_steps_per_iteration": 2},
                          {"start_iteration": 1, "train_steps_per_iteration": 4}])
        restored.save_checkpoint(self.path)
        self.assertEqual(Trainer.load_checkpoint(self.path).training_steps, 7)

    def test_repeated_boundary_change_and_extra_updates_resume_exactly(self):
        trainer = self.trainer()
        trainer.reconfigure(train_steps_per_iteration=3)
        trainer.reconfigure(train_steps_per_iteration=4)
        self.assertEqual(len(trainer.training_budget_history), 1)
        self.run_one(trainer)
        trainer.reconfigure(train_steps_per_iteration=1, augment_symmetries=True)
        trainer.save_checkpoint(self.path)
        self.run_one(trainer)
        resumed = Trainer.load_checkpoint(self.path)
        self.run_one(resumed)
        self.assertEqual((resumed.iteration, resumed.training_steps), (2, 5))
        for name, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[name]))
        self.assertTrue(torch.equal(trainer.generator.get_state(), resumed.generator.get_state()))

    def test_invalid_budget_ledgers_fail_without_global_rng_changes(self):
        trainer = self.trainer()
        self.run_one(trainer)
        trainer.save_checkpoint(self.path)
        source = torch.load(self.path, weights_only=True)
        histories = [[], [{"start_iteration": 1, "train_steps_per_iteration": 2}],
                     [{"start_iteration": 0, "train_steps_per_iteration": 3}],
                     [{"start_iteration": 0, "train_steps_per_iteration": 2},
                      {"start_iteration": 0, "train_steps_per_iteration": 2}],
                     [{"start_iteration": False, "train_steps_per_iteration": 2}],
                     [{"start_iteration": 0, "train_steps_per_iteration": 2},
                      {"start_iteration": 2, "train_steps_per_iteration": 2}]]
        for history in histories:
            payload = deepcopy(source)
            payload["training_budget_history"] = history
            torch.save(payload, self.path)
            python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
            with self.subTest(history=history), self.assertRaises(ValueError):
                Trainer.load_checkpoint(self.path)
            self.assertEqual(python_rng, random.getstate())
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_learner_export_is_distinct_from_rejected_champion(self):
        trainer = self.trainer()
        self.run_one(trainer)
        candidate = self.path.with_name("candidate.pt")
        trainer.export_learner(candidate)
        exported = load_model(candidate)
        self.assertEqual(loop._model_digest(exported), loop._model_digest(trainer.model))
        self.assertNotEqual(loop._model_digest(exported), loop._model_digest(trainer.champion))

    def test_new_configuration_types_and_failed_reconfigure_are_atomic(self):
        for changes in ({"self_play_model": "latest"}, {"self_play_model": True},
                        {"online_tactics_include_loss": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**changes)
        trainer = self.trainer()
        before = deepcopy(trainer.training_budget_history)
        with self.assertRaises(ValueError):
            trainer.reconfigure(train_steps_per_iteration=4, unknown=True)
        self.assertEqual(before, trainer.training_budget_history)
        self.assertEqual(trainer.config.train_steps_per_iteration, 2)

    def test_five_percent_loss_rows_use_value_only_in_regular_cycle_and_resume(self):
        trainer = self.trainer(batch_size=512, online_tactics=True,
                               online_tactics_include_loss=True, online_tactics_fraction=0.05)
        encoded = encode_state(engine.State())
        policy = encoded.legal_mask.float() / int(encoded.legal_mask.sum())
        row = CertifiedTacticalSample("forced-loss", "forced-loss", "atari_defense",
            TrainingSample(encoded.features, encoded.legal_mask, policy, -1.0, encoded.to_play),
            {"outcome": "LOSS", "winning_actions": [], "proof_depth": 3,
             "actor": "Black"}, policy_enabled=False)
        generation = {"records": [{"id": "forced-loss", "training_label": True,
                                   "outcome": "LOSS"}]}
        with patch("kingdom_ai.tactical_training.collect_certified_samples",
                   return_value=([row], generation)) as collector:
            metrics = self.run_one(trainer)
        self.assertTrue(collector.call_args.kwargs["include_loss"])
        self.assertEqual(metrics["normal_training_samples_drawn"], 486 * 2)
        self.assertEqual(metrics["tactical_training_samples_drawn"], 26 * 2)
        self.assertEqual(metrics["training_draw_counts"]["teacher_loss_rows"], 52)
        self.assertEqual(metrics["teacher_policy_loss"], 0.0)
        self.assertEqual(metrics["teacher_policy_rows"], 0)
        self.assertAlmostEqual(metrics["online_tactics"]["actual_tactical_fraction"], 26 / 512)
        certificate = trainer.online_tactical_replay.state_dict()["certificates"][0]
        self.assertEqual(certificate["proof"]["actor"], "Black")
        self.assertEqual(certificate["proof"]["self_play_actor"]["role"], "champion")
        trainer.save_checkpoint(self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertFalse(bool(restored.online_tactical_replay.sample(
            10, generator=restored.generator).policy_enabled.any()))

    def test_category_mse_is_row_weighted_not_average_of_nonempty_batches(self):
        trainer = self.trainer(online_tactics=True, online_tactics_fraction=0.5)
        encoded = encode_state(engine.State())
        policy = torch.zeros(82)
        policy[0] = 1
        trainer.online_tactical_replay.extend([CertifiedTacticalSample("win", "win", "defense",
            TrainingSample(encoded.features, encoded.legal_mask, policy, 1.0, encoded.to_play),
            {"outcome": "WIN", "winning_actions": [0], "proof_depth": 3})])
        updates = [dict(loss=1, policy_loss=0, value_loss=1,
                        teacher_win_value_loss=4, teacher_win_rows=2,
                        teacher_loss_value_loss=0, teacher_loss_rows=0),
                   dict(loss=1, policy_loss=0, value_loss=1,
                        teacher_win_value_loss=0, teacher_win_rows=0,
                        teacher_loss_value_loss=9, teacher_loss_rows=2)]
        with patch("kingdom_ai.tactical_training.collect_certified_samples",
                   return_value=([], {"records": []})), \
                patch("kingdom_ai.training.train_mixed_step", side_effect=updates):
            metrics = self.run_one(trainer)
        self.assertEqual(metrics["teacher_win_value_loss"], 4)
        self.assertEqual(metrics["teacher_loss_value_loss"], 9)
        self.assertEqual(metrics["training_draw_counts"]["teacher_win_rows"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_actual_cuda_latest_actor_cycle_and_changed_budget_resume(self):
        trainer = Trainer(TrainingConfig(games_per_iteration=2, simulations=2,
            self_play_backend="cuda", self_play_model="learner", self_play_batch_size=2,
            replay_capacity=256, batch_size=4, train_steps_per_iteration=1,
            evaluation_games=2, evaluation_simulations=2),
            PolicyValueNet(channels=4, residual_blocks=0), device="cuda")
        before = loop._model_digest(trainer.model)
        champion = loop._model_digest(trainer.champion)
        with patch.object(loop, "evaluate_models", return_value=REJECTED):
            first = trainer.run_iteration()
            trainer.reconfigure(train_steps_per_iteration=2)
            trainer.save_checkpoint(self.path)
            restored = Trainer.load_checkpoint(self.path, device="cuda")
            latest = loop._model_digest(restored.model)
            second = restored.run_iteration()
        self.assertEqual(first["self_play_actor"]["weights_sha256"], before)
        self.assertEqual(second["self_play_actor"]["weights_sha256"], latest)
        self.assertEqual(loop._model_digest(restored.champion), champion)
        self.assertEqual((restored.self_play_games, restored.training_steps), (4, 3))


if __name__ == "__main__":
    unittest.main()
