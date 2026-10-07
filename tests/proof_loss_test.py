"""Certified defensive WIN and forced LOSS supervision have distinct losses."""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import my_board_engine as engine
import torch

from kingdom_ai.encoding import encode_state, move_to_action
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.proof_replay import CertifiedTacticalReplay
from kingdom_ai.tactical_positions import _exercise, curriculum_positions, load_position
from kingdom_ai.tactical_training import (
    CertifiedTacticalSample, collect_certified_samples, split_tactical_samples, tactical_metrics,
)
from kingdom_ai.training import (
    ProofTrainingBatch, TrainingSample, augment_proof_batch, make_batch,
    mixed_policy_value_loss, train_mixed_step, train_step,
)


def proof(identifier, *, win=True):
    state = engine.State()
    encoded = encode_state(state)
    policy = torch.zeros(82)
    action = move_to_action(state.legal_moves()[0])
    if win:
        policy[action] = 1
    else:
        policy[encoded.legal_mask] = 1 / int(encoded.legal_mask.sum())
    target = TrainingSample(encoded.features, encoded.legal_mask, policy,
                            1.0 if win else -1.0, encoded.to_play)
    return CertifiedTacticalSample(identifier, identifier, "defense_test", target,
        {"outcome": "WIN" if win else "LOSS", "proof_depth": 3,
         "winning_actions": [action] if win else []}, win)


class ProofLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_forced_capture_loss_has_no_policy_imitation_label(self):
        # White cannot save two separate one-liberty chains in a single move.
        case = _exercise("forced_double_atari_loss", "atari_defense",
                         [(1, 2), (1, 3), (1, 5)], [(1, 1), (1, 4)], to_play="white")
        options = engine.TacticalSolverOptions(max_depth=3, max_nodes=100000, time_limit_ms=1000)
        rows, report = collect_certified_samples([case], options, include_loss=True)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.proof["outcome"], "LOSS")
        self.assertEqual(row.proof["proof_depth"], 2)
        self.assertEqual(row.proof["pv_end_reason"], "Capture")
        self.assertEqual(row.sample.to_play, engine.Cell.White)
        self.assertEqual(row.sample.value, -1)
        self.assertFalse(row.policy_enabled)
        self.assertEqual(row.proof["winning_actions"], [])
        self.assertEqual(row.proof["target_kind"], "value_only")
        self.assertEqual(row.proof["position"], case["position"])
        self.assertEqual(row.proof["source"], case["source"])
        self.assertEqual(row.proof["rules"]["suicide_rule"], "Loses")
        self.assertEqual(row.proof["actor"], "White")
        self.assertEqual(report["certified_loss_samples"], 1)
        self.assertEqual(report["excluded_loss"], 0)
        self.assertAlmostEqual(row.sample.policy.sum().item(), 1)
        # Returned audit data must not alias caller-owned case dictionaries.
        case["position"]["board"][0] = "........."
        self.assertNotEqual(row.proof["position"], case["position"])

    def test_real_countercapture_is_a_proved_defensive_win_not_an_escape_guess(self):
        case = next(row for row in curriculum_positions() if row["id"] == "counter_capture")
        rows, report = collect_certified_samples([case], engine.TacticalSolverOptions(
            max_depth=3, max_nodes=100000, time_limit_ms=1000), include_loss=True)
        self.assertEqual(report["certified_win_samples"], 1)
        self.assertTrue(rows[0].policy_enabled)
        state = load_position(case)
        for action in rows[0].proof["winning_actions"]:
            from kingdom_ai.encoding import action_to_move
            child = state.copy()
            self.assertTrue(child.play(action_to_move(action)).accepted())
            self.assertEqual(child.result.reason, engine.EndReason.Capture)
            self.assertEqual(child.result.winner, state.to_play)

    def test_unknown_never_becomes_a_value_target_and_loss_is_explicit_opt_in(self):
        cases = [{"id": name} for name in ("WIN", "LOSS", "UNKNOWN")]
        def generate(include_loss):
            outcomes = iter(("WIN", "LOSS", "UNKNOWN"))
            def solver(state, options):
                outcome = next(outcomes)
                return SimpleNamespace(outcome=outcome, proof_depth=3,
                    winning_moves=[state.legal_moves()[0]] if outcome == "WIN" else [])
            return collect_certified_samples(cases, object(), include_loss=include_loss,
                load_position=lambda case: engine.State(), solver=solver)
        wins, _ = generate(False)
        mixed, report = generate(True)
        self.assertEqual(len(wins), 1)
        self.assertEqual([row.sample.value for row in mixed], [1, -1])
        self.assertEqual(report["excluded_unknown"], 1)
        self.assertFalse(report["records"][-1]["training_label"])

    def test_loss_logits_get_zero_gradient_but_value_is_trained(self):
        logits = torch.zeros((3, 82), requires_grad=True)
        values = torch.zeros(3, requires_grad=True)
        policy = torch.full((3, 82), 1 / 82)
        targets = torch.tensor([1.0, 1.0, -1.0])
        legal = torch.ones((3, 82), dtype=torch.bool)
        enabled = torch.tensor([True, True, False])
        losses = mixed_policy_value_loss(logits, values, policy, targets, legal,
                                        normal_rows=1, policy_enabled=enabled)
        self.assertAlmostEqual(losses["policy_loss"].item(), 2 / 3 * math.log(82), places=6)
        self.assertAlmostEqual(losses["teacher_policy_loss"].item(), math.log(82) / 2, places=6)
        self.assertEqual(losses["teacher_loss_value_loss"].item(), 1)
        losses["loss"].backward()
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(82)))
        self.assertGreater(values.grad[2].item(), 0)
        # A different legal placeholder cannot alter the total loss.
        alternative = policy.clone()
        alternative[2] = 0
        alternative[2, 10] = 1
        changed = mixed_policy_value_loss(logits.detach(), values.detach(), alternative,
            targets, legal, normal_rows=1, policy_enabled=enabled)
        self.assertEqual(changed["loss"].item(), losses["loss"].item())

    def test_proof_only_loss_update_has_zero_direct_policy_head_gradient(self):
        row = proof("loss", win=False)
        batch = make_batch([row.sample])
        teacher = ProofTrainingBatch(batch.features, batch.legal_mask, batch.policy,
                                    batch.value, torch.tensor([False]))
        model = PolicyValueNet(channels=4, residual_blocks=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        metrics = train_mixed_step(model, optimizer, None, teacher)
        self.assertEqual(metrics["teacher_loss_rows"], 1)
        self.assertEqual(metrics["teacher_policy_loss"], 0)
        self.assertEqual(metrics["teacher_black_rows"], 1)
        for parameter in model.policy_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))
        self.assertTrue(any(parameter.grad is not None and bool((parameter.grad != 0).any())
                            for parameter in model.value_head.parameters()))
        with self.assertRaisesRegex(ValueError, "train_mixed_step"):
            train_step(model, optimizer, teacher)

    def test_replay_ring_mask_and_seeded_sampling_roundtrip(self):
        replay = CertifiedTacticalReplay(4)
        replay.extend([proof(str(index), win=index % 2 == 0) for index in range(13)])
        state = replay.state_dict()
        self.assertEqual(state["version"], 2)
        self.assertTrue(torch.equal(state["policy_enabled"], state["replay"]["value"] == 1))
        restored = CertifiedTacticalReplay.from_state_dict(state)
        a = replay.sample(40, generator=torch.Generator().manual_seed(104))
        b = restored.sample(40, generator=torch.Generator().manual_seed(104))
        self.assertTrue(torch.equal(a.features, b.features))
        self.assertTrue(torch.equal(a.value, b.value))
        self.assertTrue(torch.equal(a.policy_enabled, b.policy_enabled))
        self.assertTrue(torch.equal(a.policy_enabled, a.value == 1))
        augmented = augment_proof_batch(a, generator=torch.Generator().manual_seed(81))
        self.assertTrue(torch.equal(augmented.policy_enabled, a.policy_enabled))
        self.assertTrue(torch.equal(augmented.value, a.value))
        state["policy_enabled"][0] = ~state["policy_enabled"][0]
        with self.assertRaises(ValueError):
            CertifiedTacticalReplay.from_state_dict(state)

    def test_legacy_194_wins_import_without_losing_ring_order(self):
        replay = CertifiedTacticalReplay(194)
        replay.extend([proof(str(index)) for index in range(200)])
        legacy = replay.state_dict()
        legacy["version"] = 1
        legacy.pop("policy_enabled")
        restored = CertifiedTacticalReplay.from_state_dict(legacy)
        self.assertEqual(len(restored), 194)
        self.assertEqual(restored.state_dict()["replay"]["next_index"], 6)
        self.assertEqual(restored.state_dict()["certificates"], legacy["certificates"])
        self.assertTrue(bool(restored.state_dict()["policy_enabled"].all()))
        left = replay.sample(512, generator=torch.Generator().manual_seed(106))
        right = restored.sample(512, generator=torch.Generator().manual_seed(106))
        self.assertTrue(torch.equal(left.policy, right.policy))

    def test_corrupt_loss_and_unknown_certificates_fail_closed(self):
        replay = CertifiedTacticalReplay(2)
        replay.extend([proof("loss", win=False)])
        baseline = replay.state_dict()
        for field, value in (("outcome", "UNKNOWN"), ("winning_actions", [0])):
            damaged = deepcopy(baseline)
            damaged["certificates"][0]["proof"][field] = value
            with self.assertRaises(ValueError):
                CertifiedTacticalReplay.from_state_dict(damaged)
        damaged = deepcopy(baseline)
        damaged["version"] = 1
        damaged.pop("policy_enabled")
        with self.assertRaises(ValueError):
            CertifiedTacticalReplay.from_state_dict(damaged)

    def test_loss_heldout_has_value_metrics_but_no_dummy_policy_hits(self):
        rows = [proof("black_loss", win=False)]
        state = engine.State()
        state.place(0, 0)
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[encoded.legal_mask] = 1 / int(encoded.legal_mask.sum())
        rows.append(CertifiedTacticalSample("white_loss", "white_family", "defense_test",
            TrainingSample(encoded.features, encoded.legal_mask, policy, -1, encoded.to_play),
            {"outcome": "LOSS", "proof_depth": 3, "winning_actions": []}, False))
        train, heldout = split_tactical_samples(rows, allow_loss=True)
        self.assertEqual((len(train), len(heldout)), (1, 1))
        metrics = tactical_metrics(PolicyValueNet(channels=4, residual_blocks=0),
                                   rows, allow_loss=True)
        self.assertEqual(metrics["loss_cases"], 2)
        self.assertEqual(metrics["win_cases"], 0)
        self.assertIsNone(metrics["certified_top1"])
        self.assertIsNone(metrics["certified_top3"])
        self.assertIsNone(metrics["win_value_mse"])
        self.assertIsNotNone(metrics["loss_value_mse"])


if __name__ == "__main__":
    unittest.main()
