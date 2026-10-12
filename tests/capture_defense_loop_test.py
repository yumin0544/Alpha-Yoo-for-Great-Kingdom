"""Capture-action teacher integration, fixed batch budget and v9 migration."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig
from kingdom_ai.capture_defense import CertifiedDefenseSample
from kingdom_ai.encoding import encode_state
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.metrics import compact_metric_row
from kingdom_ai.tactical_training import CertifiedTacticalSample
from kingdom_ai.training import GameData, TrainingSample
import kingdom_ai.loop as loop


def game():
    state, samples = engine.State(), []
    for _ in range(2):
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[81] = 1
        samples.append(TrainingSample(encoded.features, encoded.legal_mask, policy,
            -1.0 if encoded.to_play == engine.Cell.Black else 1.0, encoded.to_play))
        state.pass_turn()
    return GameData(samples, state.result.winner, state.result.reason, (81, 81))


def proof():
    encoded = encode_state(engine.State())
    policy = torch.zeros(82)
    policy[1] = 1
    return CertifiedTacticalSample("proof", "proof", "mock",
        TrainingSample(encoded.features, encoded.legal_mask, policy, 1.0, encoded.to_play),
        {"outcome": "WIN", "winning_actions": [1], "proof_depth": 3})


def defense():
    encoded = encode_state(engine.State())
    bad = torch.zeros(82, dtype=torch.bool)
    bad[0] = True
    return CertifiedDefenseSample("bad-action", "defense", encoded.features,
        encoded.legal_mask, bad,
        {"outcome": "ACTION_LOSS", "losing_actions": [0], "proof_depth": 4,
         "successor_outcome": "WIN", "target_kind": "negative_policy_only"})


class CaptureDefenseLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="capture_loop_",
            dir=Path(__file__).resolve().parents[1] / "runs")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "latest.pt"

    def trainer(self, **overrides):
        config = TrainingConfig(games_per_iteration=2, simulations=1, replay_capacity=16,
            batch_size=8, train_steps_per_iteration=2, evaluation_games=2,
            evaluation_simulations=1, online_tactics=True,
            online_tactics_capture_defense=True, online_tactics_fraction=0.5,
            online_tactics_max_depth=30, online_tactics_replay_capacity=8,
            online_tactics_defense_replay_capacity=8, **overrides)
        return Trainer(config, PolicyValueNet(channels=4, residual_blocks=0))

    def run_one(self, trainer, *, proofs=True, defenses=True):
        proof_rows, defense_rows = ([proof()] if proofs else []), ([defense()] if defenses else [])
        proof_report = {"records": [{"id": row.case_id, "outcome": "WIN",
            "proof_depth": 3, "training_label": True} for row in proof_rows]}
        defense_report = {"records": [{"id": row.case_id, "outcome": "ACTION_LOSS",
            "proof_depth": 4, "training_label": True} for row in defense_rows]}
        with patch.object(trainer, "_collect_games", return_value=iter((game(), game()))), \
             patch("kingdom_ai.tactical_training.collect_certified_samples",
                   return_value=(proof_rows, proof_report)), \
             patch("kingdom_ai.capture_mining.collect_capture_defense_samples",
                   return_value=(defense_rows, defense_report)) as collect, \
             patch.object(loop, "evaluate_models", return_value=EvaluationResult(
                 2, 0, 2, 0, 0, 4, {"TwoPasses": 2})):
            row = trainer.run_iteration()
        return row, collect

    def test_fixed_teacher_budget_splits_proof_and_avoidance_without_extra_updates(self):
        trainer = self.trainer()
        metrics, collect = self.run_one(trainer)
        online = metrics["online_tactics"]
        self.assertEqual((online["replay_rows_per_batch"], online["proof_rows_per_batch"],
                          online["defense_rows_per_batch"]), (4, 2, 2))
        self.assertEqual((trainer.training_steps, trainer.normal_training_steps,
                          trainer.tactical_training_steps, trainer.self_play_games), (2, 2, 0, 2))
        self.assertEqual(metrics["training_draw_counts"]["defense_rows"], 4)
        self.assertEqual(metrics["training_draw_counts"]["teacher_win_rows"], 4)
        self.assertEqual(metrics["training_draw_counts"]["teacher_loss_rows"], 0)
        self.assertEqual(collect.call_args.args[1].max_depth, 30)
        self.assertEqual(collect.call_args.kwargs["generation_seconds"], 15)
        self.assertIn("self_play_actor", trainer.capture_defense_replay.state_dict()["certificates"][0]["proof"])

    def test_defense_only_and_proof_only_reassign_reserved_teacher_rows(self):
        for proofs, defenses, expected in ((False, True, (0, 4)), (True, False, (4, 0)),
                                           (False, False, (0, 0))):
            with self.subTest(proofs=proofs, defenses=defenses):
                row, _ = self.run_one(self.trainer(), proofs=proofs, defenses=defenses)
                self.assertEqual((row["online_tactics"]["proof_rows_per_batch"],
                                  row["online_tactics"]["defense_rows_per_batch"]), expected)
                self.assertEqual(row["online_tactics"]["replay_rows_per_batch"], 8 - sum(expected))

    def test_exact_resume_preserves_new_buffer_rng_optimizer_and_mixed_learning(self):
        trainer = self.trainer()
        self.run_one(trainer)
        trainer.save_checkpoint(self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertEqual(restored.config, trainer.config)
        a = trainer.capture_defense_replay.sample(8, generator=trainer.generator)
        b = restored.capture_defense_replay.sample(8, generator=restored.generator)
        self.assertTrue(torch.equal(a.losing_mask, b.losing_mask))
        self.assertTrue(torch.equal(a.features, b.features))
        before = trainer.generator.get_state().clone()
        expected, _ = self.run_one(trainer, proofs=False, defenses=False)
        actual, _ = self.run_one(restored, proofs=False, defenses=False)
        self.assertTrue(torch.equal(trainer.generator.get_state(), restored.generator.get_state()))
        for key, tensor in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.model.state_dict()[key]))
        for key in ("loss", "policy_loss", "value_loss", "defense_policy_loss"):
            self.assertEqual(expected[key], actual[key])
        self.assertFalse(torch.equal(before, trainer.generator.get_state()))

    def test_old_v9_restores_disabled_empty_without_changing_prior_settings(self):
        trainer = Trainer(TrainingConfig(online_tactics=True, online_tactics_max_depth=20),
                          PolicyValueNet(channels=4, residual_blocks=0))
        trainer.save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload["checkpoint_version"] = 9
        payload.pop("capture_defense_replay")
        for field in loop._CAPTURE_DEFENSE_CONFIG_KEYS:
            payload["config"].pop(field)
        torch.save(payload, self.path)
        restored = Trainer.load_checkpoint(self.path)
        self.assertFalse(restored.config.online_tactics_capture_defense)
        self.assertEqual(restored.config.online_tactics_max_depth, 20)
        self.assertEqual(len(restored.capture_defense_replay), 0)
        for key, value in payload["config"].items():
            self.assertEqual(asdict(restored.config)[key], value)

    def test_corrupt_v10_defense_field_fails_before_restoring_global_rng(self):
        trainer = self.trainer()
        trainer.capture_defense_replay.extend([defense()])
        trainer.save_checkpoint(self.path)
        payload = torch.load(self.path, weights_only=True)
        payload["capture_defense_replay"]["certificates"][0]["proof"]["successor_outcome"] = "UNKNOWN"
        torch.save(payload, self.path)
        before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            Trainer.load_checkpoint(self.path)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_configuration_validation_and_nonempty_resize_are_atomic(self):
        for config in ({"online_tactics_capture_defense": True},
                       {"online_tactics_capture_defense": 1},
                       {"online_tactics_defense_fraction": 1},
                       {"online_tactics_defense_fraction": float("nan")},
                       {"online_tactics_defense_replay_capacity": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TrainingConfig(**config)
        trainer = self.trainer()
        trainer.capture_defense_replay.extend([defense()])
        old_config = trainer.config
        with self.assertRaises(ValueError):
            trainer.reconfigure(online_tactics_defense_replay_capacity=16)
        self.assertEqual(trainer.config, old_config)
        trainer.reconfigure(online_tactics_capture_defense=False)
        self.assertEqual(len(trainer.capture_defense_replay), 1)

    def test_compact_logs_strip_defense_histories_without_mutating_checkpoint_audit(self):
        trainer = self.trainer()
        row, _ = self.run_one(trainer)
        before = deepcopy(row)
        compact = compact_metric_row(row)
        self.assertNotIn("records", compact["online_tactics"]["capture_defense"])
        self.assertEqual(compact["online_tactics"]["capture_defense"]["record_summary"]["accepted"],
                         {"ACTION_LOSS": 1})
        self.assertEqual(row, before)


if __name__ == "__main__":
    unittest.main()
