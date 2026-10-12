"""Certified negative policy rows never fabricate values or safe UNKNOWN moves."""

from copy import deepcopy
from dataclasses import replace
import math
import unittest

import my_board_engine as engine
import torch

from kingdom_ai.capture_defense import (
    CertifiedDefenseSample, CaptureDefenseReplay, augment_capture_defense_batch,
)
from kingdom_ai.encoding import encode_state
from kingdom_ai.training import (
    CaptureDefenseBatch, ProofTrainingBatch, TrainingSample, make_batch,
    capture_defense_policy_rows, train_mixed_step,
)


def sample(identifier="case", action=0, *, state=None):
    encoded = encode_state(engine.State() if state is None else state)
    losing = torch.zeros(82, dtype=torch.bool)
    losing[action] = True
    return CertifiedDefenseSample(identifier, identifier, encoded.features,
        encoded.legal_mask, losing, {"outcome": "ACTION_LOSS", "losing_actions": [action],
            "proof_depth": 30, "successor_outcome": "WIN", "root_outcome": "UNKNOWN",
            "target_kind": "negative_policy_only", "position": {"history": []}})


def defense_batch(*rows):
    return CaptureDefenseBatch(*(torch.stack([getattr(row, name) for row in rows])
                                 for name in ("features", "legal_mask", "losing_mask")))


def labels(*values):
    encoded = encode_state(engine.State())
    policy = torch.zeros(82)
    policy[0] = 1
    return make_batch([TrainingSample(encoded.features.clone(), encoded.legal_mask.clone(),
                      policy.clone(), value, encoded.to_play) for value in values])


def proof_batch():
    batch = labels(1, -1)
    return ProofTrainingBatch(batch.features, batch.legal_mask, batch.policy, batch.value,
                              torch.tensor([True, False]))


class DirectModel(torch.nn.Module):
    """Independent heads make absence of a defense value gradient observable."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(82))
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        return self.logits.expand(features.shape[0], -1), self.value.expand(features.shape[0])


class CaptureDefenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_negative_policy_loss_matches_bad_probability_and_has_no_illegal_gradient(self):
        logits = torch.zeros((1, 82), requires_grad=True)
        legal = torch.zeros((1, 82), dtype=torch.bool)
        legal[0, :3] = True
        losing = torch.zeros_like(legal)
        losing[0, :2] = True
        loss, probability = capture_defense_policy_rows(logits, legal, losing)
        self.assertAlmostEqual(loss.item(), math.log(3), places=6)
        self.assertAlmostEqual(probability.item(), 2 / 3, places=6)
        loss.sum().backward()
        self.assertTrue(bool((logits.grad[0, :2] > 0).all()))
        self.assertLess(logits.grad[0, 2].item(), 0)
        self.assertTrue(torch.equal(logits.grad[0, 3:], torch.zeros(79)))

    def test_extreme_logits_have_finite_loss_and_gradient_when_bad_probability_rounds_to_one(self):
        logits = torch.zeros((1, 82), requires_grad=True)
        with torch.no_grad():
            logits[0, 0] = 10000
            logits[0, 1] = -10000
        legal = torch.zeros((1, 82), dtype=torch.bool)
        legal[0, :2] = True
        losing = torch.zeros_like(legal)
        losing[0, 0] = True
        loss, probability = capture_defense_policy_rows(logits, legal, losing)
        self.assertEqual(loss.item(), 20000)
        self.assertEqual(probability.item(), 1)
        loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertEqual(logits.grad[0, 0].item(), 1)
        self.assertEqual(logits.grad[0, 1].item(), -1)

    def test_unknown_parent_is_accepted_without_a_parent_value_field(self):
        replay = CaptureDefenseReplay(2)
        replay.extend([sample()])
        batch = replay.sample(1, generator=torch.Generator().manual_seed(31))
        self.assertFalse(hasattr(batch, "value"))
        self.assertEqual(replay.state_dict()["certificates"][0]["proof"]["root_outcome"], "UNKNOWN")

    def test_defense_only_update_lowers_bad_mass_without_training_value_head(self):
        model = DirectModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.value.detach().clone()
        metrics = train_mixed_step(model, optimizer, None, defense_batch=defense_batch(sample()))
        self.assertEqual(metrics["value_loss"], 0)
        self.assertIsNone(model.value.grad)
        self.assertTrue(torch.equal(model.value.detach(), before))
        self.assertLess(model.logits[0].item(), model.logits[1].item())
        self.assertEqual(metrics["teacher_win_rows"], 0)
        self.assertEqual(metrics["teacher_loss_rows"], 0)
        self.assertEqual(metrics["defense_rows"], 1)
        self.assertEqual(metrics["actual_tactical_fraction"], 1)

    def test_normal_proof_and_defense_use_one_total_row_denominator(self):
        model = DirectModel()
        metrics = train_mixed_step(model, torch.optim.SGD(model.parameters(), lr=0),
            labels(1), proof_batch(), defense_batch=defense_batch(sample()))
        legal_count = int(sample().legal_mask.sum())
        negative_loss = math.log(legal_count / (legal_count - 1))
        self.assertAlmostEqual(metrics["policy_loss"],
                               (2 * math.log(legal_count) + negative_loss) / 4, places=6)
        self.assertAlmostEqual(metrics["value_loss"], 3 / 4)
        self.assertAlmostEqual(metrics["defense_policy_loss"], negative_loss, places=6)
        self.assertAlmostEqual(metrics["defense_bad_probability"], 1 / legal_count, places=6)
        self.assertEqual(metrics["normal_rows"], 1)
        self.assertEqual(metrics["teacher_rows"], 2)
        self.assertEqual(metrics["teacher_win_rows"], 1)
        self.assertEqual(metrics["teacher_loss_rows"], 1)
        self.assertEqual(metrics["defense_rows"], 1)
        self.assertEqual(metrics["actual_tactical_fraction"], 3 / 4)
        self.assertEqual(metrics["teacher_black_rows"], 2)

    def test_normal_and_defense_without_proof_are_supported(self):
        model = DirectModel()
        metrics = train_mixed_step(model, torch.optim.SGD(model.parameters(), lr=0),
            labels(-1), defense_batch=defense_batch(sample(), sample("b", 1)))
        self.assertEqual(metrics["teacher_rows"], 0)
        self.assertEqual(metrics["defense_rows"], 2)
        self.assertAlmostEqual(metrics["value_loss"], 1 / 3)
        self.assertAlmostEqual(model.value.grad.item(), 2 / 3, places=6)

    def test_no_defense_path_remains_exactly_the_original_proof_behavior(self):
        from kingdom_ai.training import _train_proof_mixed_step
        first, second = DirectModel(), DirectModel()
        a = train_mixed_step(first, torch.optim.SGD(first.parameters(), lr=0.1),
                             labels(1), proof_batch())
        b = _train_proof_mixed_step(second, torch.optim.SGD(second.parameters(), lr=0.1),
                                    labels(1), proof_batch())
        self.assertEqual(a, b)
        self.assertNotIn("defense_rows", a)
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.equal(first.value, second.value))

    def test_loss_placeholder_is_still_disabled_with_defense_present(self):
        original = proof_batch()
        altered = deepcopy(original)
        altered.policy[1] = 0
        altered.policy[1, 1] = 1
        a, b = DirectModel(), DirectModel()
        left = train_mixed_step(a, torch.optim.SGD(a.parameters(), lr=0.1), None,
            original, defense_batch=defense_batch(sample()))
        right = train_mixed_step(b, torch.optim.SGD(b.parameters(), lr=0.1), None,
            altered, defense_batch=defense_batch(sample()))
        self.assertEqual(left, right)
        self.assertTrue(torch.equal(a.logits, b.logits))

    def test_replay_fifo_physical_ring_and_seeded_sampling_roundtrip(self):
        replay = CaptureDefenseReplay(4)
        replay.extend([sample(str(index), index % 9) for index in range(13)])
        state = replay.state_dict()
        self.assertEqual(state["next_index"], 1)
        self.assertEqual([row["case_id"] for row in state["certificates"]], ["12", "9", "10", "11"])
        restored = CaptureDefenseReplay.from_state_dict(state)
        self.assertEqual(restored.state_dict()["certificates"], state["certificates"])
        for seed in (0, 19, 123456):
            a = replay.sample(32, generator=torch.Generator().manual_seed(seed))
            b = restored.sample(32, generator=torch.Generator().manual_seed(seed))
            for field in ("features", "legal_mask", "losing_mask"):
                self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)))
        replay.extend([sample("next", 5)])
        restored.extend([sample("next", 5)])
        self.assertEqual(replay.state_dict()["certificates"], restored.state_dict()["certificates"])

    def test_empty_and_partial_checkpoint_roundtrip(self):
        for size in (0, 1, 3):
            replay = CaptureDefenseReplay(5)
            replay.extend([sample(str(index), index) for index in range(size)])
            restored = CaptureDefenseReplay.from_state_dict(replay.state_dict())
            self.assertEqual(len(restored), size)
            self.assertEqual(restored.state_dict()["next_index"], size)

    def test_extend_is_atomic_and_checkpoints_samples_do_not_alias_storage(self):
        replay = CaptureDefenseReplay(2)
        row = sample()
        replay.extend([row])
        before = replay.state_dict()
        invalid = replace(sample("invalid"), proof={"outcome": "UNKNOWN"})
        with self.assertRaises(ValueError):
            replay.extend([sample("valid", 1), invalid])
        self.assertEqual(replay.state_dict()["certificates"], before["certificates"])
        row.losing_mask.fill_(False)
        row.proof["outcome"] = "UNKNOWN"
        batch = replay.sample(1, generator=torch.Generator().manual_seed(12))
        self.assertTrue(batch.losing_mask[0, 0])
        batch.losing_mask.fill_(False)
        before["losing_mask"].fill_(False)
        self.assertTrue(replay.state_dict()["losing_mask"][0, 0])
        restored = CaptureDefenseReplay.from_state_dict(replay.state_dict())
        replay._losing_mask.fill_(False)
        self.assertTrue(restored.state_dict()["losing_mask"][0, 0])

    def test_replay_rejects_illegal_empty_all_bad_nonfinite_and_unknown_certificates(self):
        row = sample()
        illegal = row.losing_mask.clone()
        illegal[40] = True  # fixed neutral stone is not legal
        empty = torch.zeros(82, dtype=torch.bool)
        bad_features = row.features.clone()
        bad_features[0, 0, 0] = float("nan")
        for damaged in (
            replace(row, losing_mask=illegal), replace(row, losing_mask=empty),
            replace(row, losing_mask=row.legal_mask.clone()), replace(row, features=bad_features),
            replace(row, proof={**row.proof, "outcome": "UNKNOWN"}),
            replace(row, proof={**row.proof, "successor_outcome": "UNKNOWN"}),
            replace(row, proof={**row.proof, "losing_actions": [1]}),
            replace(row, proof={**row.proof, "proof_depth": True}),
            replace(row, proof={**row.proof, "elapsed_ms": float("inf")}),
            replace(row, proof={**row.proof, "target_kind": "value_only"}),
        ):
            with self.subTest(damaged=damaged.proof), self.assertRaises(ValueError):
                CaptureDefenseReplay(2).extend([damaged])

    def test_checkpoint_rejects_corruption_in_ring_schema_masks_and_metadata(self):
        replay = CaptureDefenseReplay(3)
        replay.extend([sample()])
        original = replay.state_dict()
        mutations = [
            lambda state: state.update(version=True),
            lambda state: state.update(size=True),
            lambda state: state.update(capacity=0),
            lambda state: state.update(next_index=2),
            lambda state: state.update(extra="unexpected"),
            lambda state: state.update(certificates=[]),
            lambda state: state.update(legal_mask=state["legal_mask"].float()),
            lambda state: state["certificates"][0]["proof"].update(outcome="UNKNOWN"),
            lambda state: state["losing_mask"].fill_(False),
            lambda state: state["features"][:, 9].fill_(0),
        ]
        for mutation in mutations:
            damaged = deepcopy(original)
            mutation(damaged)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CaptureDefenseReplay.from_state_dict(damaged)

    def test_augmentation_transforms_both_masks_with_same_d4_and_preserves_pass(self):
        state = engine.State()
        state.place(0, 1)
        row = sample(state=state, action=7)
        row.losing_mask[81] = True
        batch = defense_batch(*[row] * 32)
        generator = torch.Generator().manual_seed(281)
        expected_generator = torch.Generator().manual_seed(281)
        expected_symmetries = torch.randint(8, (32,), generator=expected_generator)
        result = augment_capture_defense_batch(batch, generator=generator)
        for index, symmetry in enumerate(expected_symmetries.tolist()):
            reflection, rotation = symmetry >= 4, symmetry % 4
            features = row.features.flip(-1) if reflection else row.features
            self.assertTrue(torch.equal(result.features[index], torch.rot90(features, rotation, (-2, -1))))
            for name in ("legal_mask", "losing_mask"):
                squares = getattr(row, name)[:81].reshape(9, 9)
                squares = squares.flip(-1) if reflection else squares
                self.assertTrue(torch.equal(getattr(result, name)[index, :81],
                                             torch.rot90(squares, rotation, (-2, -1)).flatten()))
                self.assertEqual(getattr(result, name)[index, 81], getattr(row, name)[81])
        self.assertEqual(set(expected_symmetries.tolist()), set(range(8)))
        self.assertTrue(torch.equal(generator.get_state(), expected_generator.get_state()))
        result.losing_mask.fill_(False)
        self.assertTrue(batch.losing_mask[:, 7].all())

    def test_invalid_sampling_and_loss_masks_fail_closed(self):
        replay = CaptureDefenseReplay(2)
        with self.assertRaises(ValueError):
            replay.sample(1, generator=torch.Generator())
        replay.extend([sample()])
        for size, generator in ((0, torch.Generator()), (True, torch.Generator()), (1, None)):
            with self.assertRaises(ValueError):
                replay.sample(size, generator=generator)
        logits = torch.zeros((1, 82))
        legal = torch.ones((1, 82), dtype=torch.bool)
        for losing in (legal, torch.zeros_like(legal), legal.float()):
            with self.assertRaises(ValueError):
                capture_defense_policy_rows(logits, legal, losing)


if __name__ == "__main__":
    unittest.main()
