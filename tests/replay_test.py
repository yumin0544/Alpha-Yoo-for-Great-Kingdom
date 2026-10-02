"""Replay eviction, target validation, and checkpoint reproducibility tests."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch
import my_board_engine as engine

from kingdom_ai.encoding import ACTION_SIZE, encode_state
from kingdom_ai.replay import ReplayBuffer
from kingdom_ai.training import TrainingSample


def sample(index=0, actor=engine.Cell.Black):
    state = engine.State()
    if actor == engine.Cell.White:
        state.pass_turn()
    encoded = encode_state(state)
    policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    action = index % 40  # Initial-board legal actions, excluding central neutral stone.
    policy[action] = 1
    return TrainingSample(encoded.features, encoded.legal_mask, policy,
                          1.0 if index % 2 == 0 else -1.0, encoded.to_play)


def generator(seed=17):
    return torch.Generator(device="cpu").manual_seed(seed)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assertStateEqual(self, first, second):
        self.assertEqual(set(first), set(second))
        for key, before in first.items():
            with self.subTest(field=key):
                if isinstance(before, torch.Tensor):
                    torch.testing.assert_close(before, second[key], rtol=0, atol=0)
                else:
                    self.assertEqual(before, second[key])

    def assertBatchEqual(self, first, second):
        for name in ("features", "legal_mask", "policy", "value"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name),
                                       rtol=0, atol=0)

    def test_capacity_is_fixed_and_storage_is_bounded(self):
        buffer = ReplayBuffer(3)
        self.assertEqual(buffer.capacity, 3)
        self.assertEqual(len(buffer), 0)
        addresses = {name: getattr(buffer, "_" + name).data_ptr()
                     for name in ("features", "legal_mask", "policy", "value", "to_play")}
        buffer.extend(sample(index) for index in range(20))
        self.assertEqual(len(buffer), 3)
        for name, address in addresses.items():
            tensor = getattr(buffer, "_" + name)
            self.assertEqual(tensor.data_ptr(), address)
            self.assertEqual(tensor.shape[0], 3)
        total_bytes = sum(getattr(buffer, "_" + name).numel()
                          * getattr(buffer, "_" + name).element_size() for name in addresses)
        self.assertEqual(total_bytes, 3 * 3655)
        with self.assertRaises(AttributeError):
            buffer.capacity = 9
        for capacity in (0, -1, True, 3.0, "3"):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                ReplayBuffer(capacity)

    def test_fifo_eviction_keeps_latest_positions_and_ring_order(self):
        buffer = ReplayBuffer(3)
        buffer.extend(sample(index) for index in range(5))
        saved = buffer.state_dict()
        self.assertEqual(saved["next_index"], 2)
        self.assertEqual(saved["policy"].argmax(dim=1).tolist(), [3, 4, 2])
        self.assertEqual(sorted(saved["policy"].argmax(dim=1).tolist()), [2, 3, 4])
        buffer.extend([sample(5)])
        saved = buffer.state_dict()
        self.assertEqual(saved["next_index"], 0)
        self.assertEqual(saved["policy"].argmax(dim=1).tolist(), [3, 4, 5])

    def test_large_append_matches_single_position_appends(self):
        bulk, sequential = ReplayBuffer(3), ReplayBuffer(3)
        initial = sample(20, engine.Cell.White)
        bulk.extend([initial])
        sequential.extend([initial])
        incoming = [sample(index) for index in range(9)]
        bulk.extend(incoming)
        for position in incoming:
            sequential.extend([position])
        self.assertStateEqual(bulk.state_dict(), sequential.state_dict())

    def test_empty_extend_and_empty_serialization_do_not_expose_storage(self):
        buffer = ReplayBuffer(9)
        buffer.extend([])
        saved = buffer.state_dict()
        self.assertEqual(saved["size"], 0)
        self.assertEqual(saved["next_index"], 0)
        for name in ("features", "legal_mask", "policy", "value", "to_play"):
            self.assertEqual(saved[name].numel(), 0)
            self.assertEqual(saved[name].shape[0], 0)
        restored = ReplayBuffer.from_state_dict(saved)
        self.assertStateEqual(saved, restored.state_dict())
        with self.assertRaises(ValueError):
            buffer.sample(1, generator=generator())

    def test_bad_later_entry_cannot_partially_mutate_storage(self):
        buffer = ReplayBuffer(2)
        buffer.extend([sample(9)])
        before = buffer.state_dict()
        invalid = replace(sample(1), value=0.0)
        with self.assertRaises(ValueError):
            buffer.extend([sample(0), invalid])
        self.assertStateEqual(before, buffer.state_dict())
        with self.assertRaises(TypeError):
            buffer.extend([object()])
        self.assertStateEqual(before, buffer.state_dict())

    def test_source_sample_and_returned_batches_cannot_modify_buffer(self):
        source = sample(2)
        source.features.requires_grad_(True)
        source.policy.requires_grad_(True)
        buffer = ReplayBuffer(1)
        buffer.extend([source])
        before = buffer.state_dict()
        with torch.no_grad():
            source.features.fill_(0)
            source.policy.fill_(0)
            source.legal_mask.fill_(False)
        batch = buffer.sample(4, generator=generator())
        self.assertEqual(batch.features.device.type, "cpu")
        self.assertFalse(batch.features.requires_grad)
        self.assertFalse(batch.policy.requires_grad)
        batch.features.fill_(0)
        batch.policy.fill_(0)
        batch.legal_mask.fill_(False)
        batch.value.fill_(0)
        self.assertStateEqual(before, buffer.state_dict())

    def test_replacement_sampling_is_seeded_and_does_not_use_global_rng(self):
        buffer = ReplayBuffer(3)
        buffer.extend(sample(index) for index in range(3))
        global_before = torch.random.get_rng_state().clone()
        first_rng, second_rng = generator(8), generator(8)
        first = buffer.sample(16, generator=first_rng)
        second = buffer.sample(16, generator=second_rng)
        self.assertBatchEqual(first, second)
        self.assertEqual(first.features.shape, (16, 10, 9, 9))
        self.assertEqual(first.legal_mask.dtype, torch.bool)
        self.assertEqual(first.policy.dtype, torch.float32)
        self.assertEqual(first.value.dtype, torch.float32)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_before))
        self.assertFalse(torch.equal(first_rng.get_state(), generator(8).get_state()))
        selected = first.policy.argmax(dim=1).tolist()
        self.assertTrue(set(selected).issubset({0, 1, 2}))
        self.assertLess(len(set(selected)), len(selected))

    def test_sampling_checks_batch_size_and_explicit_generator(self):
        buffer = ReplayBuffer(1)
        buffer.extend([sample()])
        for size in (0, -1, True, 1.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                buffer.sample(size, generator=generator())
        for invalid in (None, 18, "cpu"):
            with self.subTest(generator=invalid), self.assertRaises(ValueError):
                buffer.sample(1, generator=invalid)

    def test_partial_full_and_wrapped_state_restore_identical_sampling(self):
        for capacity, count in ((5, 2), (3, 3), (3, 8)):
            with self.subTest(capacity=capacity, count=count):
                original = ReplayBuffer(capacity)
                original.extend(sample(index, engine.Cell.White if index % 2 else engine.Cell.Black)
                                for index in range(count))
                saved = original.state_dict()
                restored = ReplayBuffer.from_state_dict(saved)
                self.assertStateEqual(saved, restored.state_dict())
                self.assertBatchEqual(original.sample(17, generator=generator(64)),
                                      restored.sample(17, generator=generator(64)))
                original.extend([sample(30)])
                restored.extend([sample(30)])
                self.assertStateEqual(original.state_dict(), restored.state_dict())

    def test_payloads_and_restored_storage_have_no_writable_aliases(self):
        original = ReplayBuffer(2)
        original.extend([sample(1), sample(2)])
        before = original.state_dict()
        payload = original.state_dict()
        restored = ReplayBuffer.from_state_dict(payload)
        payload["features"].zero_()
        payload["policy"].zero_()
        payload["legal_mask"].zero_()
        payload["value"].zero_()
        payload["to_play"].zero_()
        self.assertStateEqual(before, original.state_dict())
        self.assertStateEqual(before, restored.state_dict())
        restored.extend([sample(8)])
        self.assertStateEqual(before, original.state_dict())

    def test_payload_is_compatible_with_weights_only_loading(self):
        original = ReplayBuffer(2)
        original.extend([sample(3, engine.Cell.White)])
        with tempfile.TemporaryDirectory(prefix="kingdom-replay-") as directory:
            path = Path(directory) / "replay.pt"
            torch.save(original.state_dict(), path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored = ReplayBuffer.from_state_dict(payload)
        self.assertStateEqual(original.state_dict(), restored.state_dict())

    def test_invalid_targets_observation_shapes_and_dtypes_are_rejected(self):
        valid = sample()
        malformed = [
            replace(valid, features=valid.features.double()),
            replace(valid, features=torch.zeros((9, 9, 9))),
            replace(valid, legal_mask=valid.legal_mask.float()),
            replace(valid, policy=valid.policy.double()),
            replace(valid, policy=torch.zeros(ACTION_SIZE - 1)),
            replace(valid, value=float("nan")),
            replace(valid, value=0.0),
            replace(valid, value=True),
            replace(valid, to_play=engine.Cell.Neutral),
            replace(valid, to_play=engine.Cell.White),
        ]
        for field, index, value in (
            ("features", (0, 0, 0), float("nan")),
            ("features", (6, 0, 0), -1.0),
            ("features", (0, 0, 0), 0.5),
            ("features", (6, 0, 0), 0.7),
            ("features", (8, slice(None), slice(None)), 1.0),
            ("policy", 0, -1.0),
            ("policy", 40, 1.0),
            ("policy", 0, float("inf")),
        ):
            tensor = getattr(valid, field).clone()
            tensor[index] = value
            malformed.append(replace(valid, **{field: tensor}))
        malformed.append(replace(valid, policy=torch.zeros(ACTION_SIZE)))
        malformed.append(replace(valid, legal_mask=torch.zeros(ACTION_SIZE, dtype=torch.bool)))
        mask = valid.legal_mask.clone()
        mask[0] = False
        malformed.append(replace(valid, legal_mask=mask))
        mask = valid.legal_mask.clone()
        mask[-1] = False
        malformed.append(replace(valid, legal_mask=mask))
        for position in malformed:
            with self.subTest(position=position), self.assertRaises((TypeError, ValueError)):
                ReplayBuffer(1).extend([position])

    def test_malformed_checkpoint_structure_and_contents_are_rejected(self):
        buffer = ReplayBuffer(3)
        buffer.extend([sample(0), sample(1)])
        payload = buffer.state_dict()
        invalid = [None, {}, dict(payload, extra=1),
                   dict(payload, version=True), dict(payload, version=2),
                   dict(payload, capacity=0), dict(payload, capacity=True),
                   dict(payload, size=-1), dict(payload, size=4),
                   dict(payload, size=True), dict(payload, next_index=0),
                   dict(payload, next_index=3), dict(payload, next_index=True),
                   dict(payload, features=payload["features"].double()),
                   dict(payload, to_play=payload["to_play"].long())]
        for field, index, value in (
            ("features", (0, 0, 0, 0), float("nan")),
            ("policy", (0, 40), 1.0),
            ("value", 0, 0.0),
            ("to_play", 0, 0),
        ):
            malformed = dict(payload)
            malformed[field] = payload[field].clone()
            malformed[field][index] = value
            invalid.append(malformed)
        for checkpoint in invalid:
            with self.subTest(checkpoint=checkpoint), self.assertRaises((TypeError, ValueError)):
                ReplayBuffer.from_state_dict(checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
