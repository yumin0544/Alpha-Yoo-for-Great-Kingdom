"""Bounded post-cycle teacher, honest update counts, and resumable proof replay."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

from kingdom_ai.encoding import action_to_move, encode_state
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.loop import Trainer, TrainingConfig, _ONLINE_CONFIG_KEYS
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.proof_replay import CertifiedTacticalReplay
from kingdom_ai.tactical_positions import user_game_two_positions
from kingdom_ai.tactical_training import CertifiedTacticalSample, collect_certified_samples
from kingdom_ai.training import GameData, TrainingSample

loop = importlib.import_module(Trainer.__module__)


def game(actions=(81, 81)):
    state, positions = engine.State(), []
    for action in actions:
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[action] = 1
        positions.append((encoded, policy))
        assert state.play(action_to_move(action)).accepted()
    assert state.result.finished()
    return GameData([TrainingSample(encoded.features, encoded.legal_mask, policy,
                                   1.0 if encoded.to_play == state.result.winner else -1.0,
                                   encoded.to_play) for encoded, policy in positions],
                    state.result.winner, state.result.reason, tuple(actions))


def teacher(identifier="new-win", depth=3, action=0):
    encoded = encode_state(engine.State())
    policy = torch.zeros(82)
    policy[action] = 1
    return CertifiedTacticalSample(identifier, identifier, "mock_teacher",
        TrainingSample(encoded.features, encoded.legal_mask, policy, 1.0, encoded.to_play),
        {"outcome": "WIN", "winning_actions": [action], "proof_depth": depth})


def evaluation():
    return EvaluationResult(games=2, wins=0, losses=2, wins_as_black=0,
                            wins_as_white=0, total_plies=4, endings={"TwoPasses": 2})


class OnlineTacticsLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def trainer(self, **changes):
        config = TrainingConfig(games_per_iteration=2, simulations=2, replay_capacity=16,
            batch_size=4, train_steps_per_iteration=2, evaluation_games=2,
            evaluation_simulations=2, online_tactics=True, online_tactics_max_cases=2,
            online_tactics_replay_capacity=4, **changes)
        return Trainer(config, model=PolicyValueNet(channels=4, residual_blocks=0))

    def run_mock(self, trainer, rows):
        records = [{"id": row.case_id, "outcome": "WIN", "training_label": True}
                   for row in rows]
        records += [{"id": "unproved", "outcome": "UNKNOWN", "training_label": False},
                    {"id": "forced-loss", "outcome": "LOSS", "training_label": False}]
        report = {"cases_available": len(records), "cases_solved": len(records),
                  "certified_win_samples": len(rows), "excluded_unknown": 1,
                  "excluded_loss": 1, "records": records}
        with patch.object(trainer, "_collect_games", return_value=iter((game(), game()))), \
                patch("kingdom_ai.tactical_training.collect_certified_samples", return_value=(rows, report)), \
                patch.object(loop, "evaluate_models", return_value=evaluation()):
            return trainer.run_iteration()

    def test_new_proofs_mix_without_inventing_extra_adam_updates_or_games(self):
        trainer = self.trainer()
        batches = []
        from kingdom_ai.training import train_mixed_step
        def step(model, optimizer, normal, proof):
            batches.append(torch.cat((normal.policy, proof.policy)).cpu().clone())
            return train_mixed_step(model, optimizer, normal, proof)
        with patch("kingdom_ai.training.train_mixed_step", side_effect=step):
            metrics = self.run_mock(trainer, [teacher(), teacher("shallow", 1)])
        self.assertEqual((trainer.iteration, trainer.self_play_games, trainer.training_steps,
                          trainer.tactical_training_steps, trainer.champion_version), (1, 2, 2, 0, 0))
        self.assertEqual(len(trainer.replay), 4)
        self.assertEqual(len(trainer.online_tactical_replay), 1)
        online = metrics["online_tactics"]
        self.assertEqual((online["added_samples"], online["excluded_shallow"],
                          online["mixed_updates"], online["tactical_rows_per_batch"],
                          online["replay_rows_per_batch"]), (1, 1, 2, 1, 3))
        self.assertFalse(next(row for row in online["records"] if row["id"] == "shallow")["training_label"])
        for policy in batches:
            self.assertEqual(int((policy[:, 0] == 1).sum()), 1)
            self.assertEqual(int((policy[:, 81] == 1).sum()), 3)

    def test_no_new_proof_reuses_old_certificates_or_falls_back_to_normal_replay(self):
        trainer = self.trainer()
        first = self.run_mock(trainer, [])
        self.assertEqual(first["online_tactics"]["mixed_updates"], 0)
        trainer.online_tactical_replay.extend([teacher("existing")])
        second = self.run_mock(trainer, [])
        self.assertEqual(second["online_tactics"]["added_samples"], 0)
        self.assertEqual(second["online_tactics"]["mixed_updates"], 2)
        self.assertEqual(trainer.training_steps, 4)

    def test_file_log_is_compact_but_callback_return_and_checkpoint_keep_full_records(self):
        trainer = self.trainer()
        completed = self.run_mock(trainer, [teacher()])
        seen = []
        fixture_root = Path(__file__).resolve().parents[1] / "runs"
        with tempfile.TemporaryDirectory(prefix="compact_log_test_", dir=fixture_root) as directory:
            checkpoint, log = Path(directory) / "latest.pt", Path(directory) / "metrics.jsonl"
            with patch.object(trainer, "run_iteration", return_value=deepcopy(completed)):
                returned = trainer.run(1, checkpoint_path=checkpoint, metrics_path=log,
                                       on_iteration=seen.append)
            written = json.loads(log.read_text(encoding="utf-8"))
            self.assertNotIn("records", written["online_tactics"])
            self.assertEqual(written["online_tactics"]["record_summary"]["records"], 3)
            self.assertEqual(returned[0]["online_tactics"]["records"], completed["online_tactics"]["records"])
            self.assertEqual(seen[0]["online_tactics"]["records"], completed["online_tactics"]["records"])
            restored = Trainer.load_checkpoint(checkpoint)
            self.assertEqual(restored._last_metrics["online_tactics"]["records"],
                             completed["online_tactics"]["records"])
            self.assertEqual(len(restored.online_tactical_replay), 1)

    def test_disabled_teacher_never_mines_solves_or_mixes(self):
        trainer = Trainer(replace(self.trainer().config, online_tactics=False),
                          model=PolicyValueNet(channels=4, residual_blocks=0))
        trainer.online_tactical_replay.extend([teacher("preserved")])
        with patch("kingdom_ai.online_tactics.TacticalPositionMiner", side_effect=AssertionError("disabled")), \
                patch("kingdom_ai.tactical_training.collect_certified_samples", side_effect=AssertionError("disabled")), \
                patch.object(trainer, "_collect_games", return_value=iter((game(), game()))), \
                patch.object(loop, "evaluate_models", return_value=evaluation()):
            metrics = trainer.run_iteration()
        self.assertFalse(metrics["online_tactics"]["enabled"])
        self.assertEqual(metrics["online_tactics"]["mixed_updates"], 0)
        self.assertEqual(len(trainer.online_tactical_replay), 1)

    def test_v7_roundtrip_preserves_proof_ring_and_seeded_sampling(self):
        trainer = self.trainer()
        self.run_mock(trainer, [teacher()])
        trainer.online_tactical_replay.extend([teacher(str(i), action=i) for i in range(8)])
        with tempfile.TemporaryDirectory(prefix="kingdom-online-") as directory:
            path = Path(directory) / "latest.pt"
            trainer.save_checkpoint(path)
            restored = Trainer.load_checkpoint(path)
        self.assertEqual(restored.config, trainer.config)
        self.assertEqual(len(restored.online_tactical_replay), 4)
        self.assertEqual(restored.training_steps, 2)
        self.assertEqual(restored.tactical_training_steps, 0)
        a = trainer.online_tactical_replay.sample(32, generator=trainer.generator)
        b = restored.online_tactical_replay.sample(32, generator=restored.generator)
        self.assertTrue(torch.equal(a.features, b.features))
        self.assertTrue(torch.equal(a.policy, b.policy))
        self.assertEqual(restored.online_tactical_replay.state_dict()["certificates"],
                         trainer.online_tactical_replay.state_dict()["certificates"])

    def test_v6_migrates_disabled_empty_without_losing_extra_step_count(self):
        trainer = self.trainer()
        self.run_mock(trainer, [])
        batch = trainer.replay.sample(2, generator=trainer.generator)
        trainer.train_tactical_batch(batch)
        with tempfile.TemporaryDirectory(prefix="kingdom-online-v6-") as directory:
            path = Path(directory) / "latest.pt"
            trainer.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            payload["checkpoint_version"] = 6
            payload.pop("promotion_league")
            for key in loop._PROMOTION_CONFIG_KEYS:
                payload["config"].pop(key)
            payload.pop("training_budget_history")
            payload["progress"].pop("normal_training_steps")
            for key in loop._ADAPTIVE_CONFIG_KEYS:
                payload["config"].pop(key)
            payload.pop("online_tactical_replay")
            for key in _ONLINE_CONFIG_KEYS:
                payload["config"].pop(key)
            torch.save(payload, path)
            restored = Trainer.load_checkpoint(path)
        self.assertFalse(restored.config.online_tactics)
        self.assertEqual(len(restored.online_tactical_replay), 0)
        self.assertEqual((restored.training_steps, restored.tactical_training_steps), (3, 1))

    def test_failed_proof_phase_cannot_save_partial_cycle(self):
        trainer = self.trainer()
        with patch.object(trainer, "_collect_games", return_value=iter((game(), game()))), \
                patch("kingdom_ai.tactical_training.collect_certified_samples", side_effect=RuntimeError("solver failed")):
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                trainer.run_iteration()
        self.assertEqual(trainer.training_steps, 0)
        with self.assertRaises(RuntimeError):
            trainer.save_checkpoint("not-written.pt")

    def test_teacher_checkpoint_rejects_unknown_loss_and_mismatched_policy(self):
        replay = CertifiedTacticalReplay(4)
        replay.extend([teacher()])
        state = replay.state_dict()
        for outcome in ("UNKNOWN", "LOSS"):
            damaged = deepcopy(state)
            damaged["certificates"][0]["proof"]["outcome"] = outcome
            with self.assertRaises(ValueError):
                CertifiedTacticalReplay.from_state_dict(damaged)
        damaged = deepcopy(state)
        damaged["certificates"][0]["proof"]["winning_actions"] = [1]
        with self.assertRaises(ValueError):
            CertifiedTacticalReplay.from_state_dict(damaged)
        damaged = deepcopy(state)
        damaged["replay"]["value"][0] = -1
        with self.assertRaises(ValueError):
            CertifiedTacticalReplay.from_state_dict(damaged)
        damaged = deepcopy(state)
        damaged["certificates"].clear()
        with self.assertRaises(ValueError):
            CertifiedTacticalReplay.from_state_dict(damaged)

    def test_configuration_limits_and_nonempty_resize_are_validated(self):
        for changes in ({"online_tactics": 1}, {"online_tactics_max_depth": 257},
                        {"online_tactics_fraction": 1}, {"online_tactics_max_cases": 0},
                        {"online_tactics_time_limit_ms": 2 ** 31},
                        {"online_tactics_max_nodes": 2 ** 64},
                        {"online_tactics_generation_seconds": float("nan")},
                        {"online_tactics": True, "batch_size": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**changes)
        trainer = self.trainer()
        trainer.reconfigure(online_tactics_replay_capacity=8)
        self.assertEqual(trainer.online_tactical_replay.capacity, 8)
        trainer.online_tactical_replay.extend([teacher()])
        with self.assertRaises(ValueError):
            trainer.reconfigure(online_tactics_replay_capacity=16)
        self.assertEqual(trainer.online_tactical_replay.capacity, 8)

    def test_remaining_global_budget_is_passed_to_each_solver(self):
        seen = []
        options = engine.TacticalSolverOptions(max_depth=20, max_nodes=2000000, time_limit_ms=2000)
        _, report = collect_certified_samples(
            [{"id": "deadline"}], options, load_position=lambda case: engine.State(),
            solver=lambda state, bounded: (seen.append(
                (bounded.max_depth, bounded.max_nodes, bounded.time_limit_ms))
                or SimpleNamespace(outcome="UNKNOWN", winning_moves=[])),
            generation_seconds=1.0, bound_remaining_time=True)
        self.assertEqual(report["excluded_unknown"], 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], (20, 2000000))
        self.assertTrue(0 < seen[0][2] <= 1000)
        self.assertEqual(options.time_limit_ms, 2000)

    def test_real_twenty_ply_cap_honors_tiny_node_budget_without_inventing_targets(self):
        state = engine.State()
        before = (state.board.cells, state.ownership, state.to_play, state.consecutive_passes)
        options = engine.TacticalSolverOptions(max_depth=20, max_nodes=1, time_limit_ms=100)
        samples, report = collect_certified_samples(
            [{"id": "depth_twenty_bounded"}], options, load_position=lambda case: state,
            generation_seconds=0.2, bound_remaining_time=True, include_loss=True)
        self.assertEqual(samples, [])
        self.assertEqual(report["excluded_unknown"], 1)
        self.assertEqual(report["cases_solved"], 1)
        record = report["records"][0]
        self.assertEqual(record["outcome"], "UNKNOWN")
        self.assertFalse(record["training_label"])
        self.assertTrue(record["budget_exhausted"])
        self.assertLessEqual(record["nodes"], 1)
        self.assertEqual((options.max_depth, options.max_nodes, options.time_limit_ms), (20, 1, 100))
        self.assertEqual((state.board.cells, state.ownership, state.to_play, state.consecutive_passes),
                         before)

    def test_real_nine_ply_proof_is_used_by_regular_iteration(self):
        # Exact recorded self-play fixture, not a tensor-to-board reconstruction.
        cases = user_game_two_positions()
        case = next(row for row in cases if row["id"] == "user_game_2_after_ply_17")
        # Reuse the complete public game record from its final prefix plus last move.
        history = cases[-1]["position"]["history"] + [[6, 2]]
        actions = tuple((row - 1) * 9 + col - 1 for row, col in history)
        data = game(actions)
        # Keep the original exact nine-ply proof budget explicit; the new
        # default cap is not a promise that every position reaches depth 20.
        # This checks the exact proof-to-learning path, not machine throughput.
        # The proof takes ~1.8s alone on the recorded machine and may exceed
        # the production 2s limit under a combined suite. Keep the node cap and
        # exact WIN/depth/PV assertions; only this test gets a generous deadline.
        # A separate tiny-budget test verifies correct UNKNOWN on exhaustion.
        trainer = self.trainer(online_tactics_max_depth=9, online_tactics_time_limit_ms=10000,
                               online_tactics_generation_seconds=15.0)
        with patch.object(trainer, "_collect_games", return_value=iter((data, data))), \
                patch("kingdom_ai.online_tactics.TacticalPositionMiner.cases", return_value=[case]), \
                patch.object(loop, "evaluate_models", return_value=evaluation()):
            metrics = trainer.run_iteration()
        self.assertEqual(metrics["online_tactics"]["added_samples"], 1)
        self.assertEqual(metrics["online_tactics"]["records"][0]["proof_depth"], 9)
        self.assertEqual(metrics["online_tactics"]["records"][0]["pv_end_reason"], "Capture")
        self.assertEqual(metrics["online_tactics"]["mixed_updates"], 2)
        self.assertEqual((trainer.training_steps, trainer.tactical_training_steps), (2, 0))


if __name__ == "__main__":
    unittest.main()
