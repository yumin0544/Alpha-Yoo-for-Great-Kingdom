"""D4 action alignment, engine observation parity and checkpointable randomness."""

import unittest

import my_board_engine as engine
import torch

from kingdom_ai.augmentation import augment_batch
from kingdom_ai.encoding import ACTION_SIZE, PASS_ACTION, encode_state
from kingdom_ai.training import TrainingBatch


def rotate(tensor, symmetry):
    return torch.rot90(tensor if symmetry < 4 else tensor.flip(-1), symmetry % 4,
                       dims=(-2, -1))


def example_batch(size=64):
    features = torch.arange(size * 10 * 81, dtype=torch.float32).reshape(size, 10, 9, 9)
    mask = torch.zeros((size, ACTION_SIZE), dtype=torch.bool)
    mask[:, [0, 3, 17, 40, 73, PASS_ACTION]] = True
    policy = mask.float() / mask.sum(dim=1, keepdim=True)
    values = torch.where(torch.arange(size) % 2 == 0, 1., -1.)
    return TrainingBatch(features, mask, policy, values)


class AugmentationTest(unittest.TestCase):
    def test_all_eight_symmetries_align_planes_mask_policy_and_keep_pass_value(self):
        batch = example_batch()
        generator = torch.Generator().manual_seed(715)
        expected_generator = torch.Generator().manual_seed(715)
        transforms = torch.randint(8, (64,), generator=expected_generator).tolist()
        self.assertEqual(set(transforms), set(range(8)))
        transformed = augment_batch(batch, generator=generator)
        self.assertEqual(transformed.features.shape, batch.features.shape)
        for row, transform in enumerate(transforms):
            torch.testing.assert_close(transformed.features[row], rotate(batch.features[row], transform),
                                       rtol=0, atol=0)
            for name in ("legal_mask", "policy"):
                expected = rotate(getattr(batch, name)[row, :81].reshape(9, 9), transform).flatten()
                torch.testing.assert_close(getattr(transformed, name)[row, :81], expected,
                                           rtol=0, atol=0)
                self.assertEqual(getattr(transformed, name)[row, 81].item(),
                                 getattr(batch, name)[row, 81].item())
        torch.testing.assert_close(transformed.value, batch.value, rtol=0, atol=0)
        self.assertTrue((transformed.policy[~transformed.legal_mask] == 0).all())
        torch.testing.assert_close(transformed.policy.sum(1), torch.ones(64), rtol=0, atol=1e-6)
        torch.testing.assert_close(generator.get_state(), expected_generator.get_state(), rtol=0, atol=0)
        for name in ("features", "legal_mask", "policy", "value"):
            self.assertNotEqual(getattr(transformed, name).data_ptr(), getattr(batch, name).data_ptr())
        original = batch.features.clone()
        transformed.features.zero_()
        torch.testing.assert_close(batch.features, original, rtol=0, atol=0)

    def test_spatial_transforms_match_actual_engine_states_and_completed_houses(self):
        black = ((0, 3), (1, 2), (2, 1), (2, 3), (3, 2))
        white = ((5, 6), (6, 5), (6, 7), (7, 6), (8, 0))
        neutral = (4, 7)

        def encode_board(transform):
            indices = rotate(torch.arange(81).reshape(9, 9), transform)
            inverse = {int(source): divmod(target, 9)
                       for target, source in enumerate(indices.flatten())}
            board = engine.Board(neutral=engine.Position(*inverse[neutral[0] * 9 + neutral[1]]))
            for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
                for row, col in points:
                    self.assertTrue(board.place(engine.Position(*inverse[row * 9 + col]), color))
            return encode_state(engine.State(board, engine.Cell.White))

        source = encode_board(0)
        self.assertGreater(source.features[3].sum().item(), 0)
        self.assertGreater(source.features[4].sum().item(), 0)
        batch = TrainingBatch(source.features.repeat(64, 1, 1, 1),
                              source.legal_mask.repeat(64, 1),
                              source.legal_mask.float().repeat(64, 1) / source.legal_mask.sum(),
                              torch.full((64,), -1.))
        generator = torch.Generator().manual_seed(715)
        transforms = torch.randint(8, (64,), generator=torch.Generator().manual_seed(715)).tolist()
        actual = augment_batch(batch, generator=generator)
        for transform in range(8):
            row = transforms.index(transform)
            expected = encode_board(transform)
            torch.testing.assert_close(actual.features[row], expected.features, rtol=0, atol=0)
            torch.testing.assert_close(actual.legal_mask[row], expected.legal_mask, rtol=0, atol=0)

    def test_saved_private_generator_repeats_without_consuming_global_rng(self):
        batch = example_batch(13)
        generator = torch.Generator().manual_seed(73)
        augment_batch(batch, generator=generator)
        saved = generator.get_state()
        global_before = torch.get_rng_state().clone()
        expected = augment_batch(batch, generator=generator)
        restored = torch.Generator().set_state(saved)
        actual = augment_batch(batch, generator=restored)
        for name in ("features", "legal_mask", "policy", "value"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_before, rtol=0, atol=0)

    def test_invalid_batches_or_generators_fail_before_randomness_is_consumed(self):
        batch = example_batch(1)
        generator = torch.Generator().manual_seed(81)
        before = generator.get_state()
        for damaged in (
            None,
            TrainingBatch(batch.features[:, :9], batch.legal_mask, batch.policy, batch.value),
            TrainingBatch(batch.features, batch.legal_mask[:, :81], batch.policy, batch.value),
            TrainingBatch(batch.features, batch.legal_mask.float(), batch.policy, batch.value),
            TrainingBatch(batch.features, batch.legal_mask, batch.policy, batch.value[:, None]),
        ):
            with self.subTest(batch=type(damaged).__name__), self.assertRaises((ValueError, TypeError)):
                augment_batch(damaged, generator=generator)
        for damaged in (None, 42):
            with self.assertRaises(ValueError):
                augment_batch(batch, generator=damaged)
        torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_matches_cpu_and_preserves_cuda_randomness(self):
        batch = example_batch(17)
        cuda_batch = TrainingBatch(*(getattr(batch, name).cuda()
                                    for name in ("features", "legal_mask", "policy", "value")))
        cuda_before = torch.cuda.get_rng_state_all()
        cpu_result = augment_batch(batch, generator=torch.Generator().manual_seed(98))
        cuda_result = augment_batch(cuda_batch, generator=torch.Generator().manual_seed(98))
        for name in ("features", "legal_mask", "policy", "value"):
            self.assertEqual(getattr(cuda_result, name).device.type, "cuda")
            torch.testing.assert_close(getattr(cuda_result, name).cpu(), getattr(cpu_result, name),
                                       rtol=0, atol=0)
        for expected, actual in zip(cuda_before, torch.cuda.get_rng_state_all()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            augment_batch(cuda_batch, generator=torch.Generator(device="cuda"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
