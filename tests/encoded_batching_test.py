"""Encoded leaf batching, source preservation, queue lifecycle and real CUDA."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event
import unittest

import numpy as np
import torch

import my_board_engine as engine
from kingdom_ai.encoded_batching import EncodedBatchedEvaluator
from kingdom_ai.encoding import ACTION_SIZE, encode_state
from kingdom_ai.model import PolicyValueNet, masked_policy


def encoded_rows(count=3):
    state = engine.State()
    states = [state.copy()]
    for index in range(count - 1):
        state.place(index // 9, index % 9)
        states.append(state.copy())
    encoded = [encode_state(item) for item in states]
    return (np.stack([item.features.numpy() for item in encoded]),
            np.stack([item.legal_mask.numpy() for item in encoded]))


class EncodedBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(429)
            return PolicyValueNet(channels=4, residual_blocks=0)

    def assert_prediction(self, actual, expected, masks, *, rtol=1e-5, atol=1e-6):
        policies, values = actual
        self.assertEqual(policies.shape, masks.shape)
        self.assertEqual(values.shape, (len(masks),))
        self.assertEqual(policies.dtype, np.float64)
        self.assertEqual(values.dtype, np.float64)
        self.assertTrue(policies.flags.c_contiguous)
        self.assertTrue(values.flags.c_contiguous)
        self.assertTrue(np.isfinite(policies).all())
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((policies[~masks] == 0).all())
        np.testing.assert_allclose(policies.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
        self.assertTrue((np.abs(values) <= 1).all())
        np.testing.assert_allclose(policies, expected[0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(values, expected[1], rtol=rtol, atol=atol)

    def direct(self, model, features, masks):
        with torch.inference_mode():
            logits, values = deepcopy(model).eval()(torch.from_numpy(features))
            return masked_policy(logits, torch.from_numpy(masks)).numpy(), values.numpy()

    def test_leaf_batch_matches_direct_and_counts_rows_not_requests(self):
        model = self.make_model()
        features, masks = encoded_rows()
        expected = self.direct(model, features, masks)
        with EncodedBatchedEvaluator(model, max_batch_size=8) as evaluator:
            actual = evaluator(features, masks)
            stats = evaluator.stats
            evaluator.reset_stats()
            self.assertEqual(evaluator.stats["network_evaluations"], 0)
        self.assert_prediction(actual, expected, masks)
        self.assertEqual(stats["network_evaluations"], 3)
        self.assertEqual(stats["submitted_requests"], 1)
        self.assertEqual(stats["inference_batches"], 1)
        self.assertEqual(stats["mean_batch_size"], 3)
        self.assertEqual(stats["mean_request_size"], 3)
        self.assertGreater(stats["batch_seconds"], 0)

    def test_multiple_leaf_requests_coalesce_and_respect_row_cap(self):
        model = self.make_model()
        features, masks = encoded_rows()
        expected = self.direct(model, features, masks)
        barrier = Barrier(3)
        with EncodedBatchedEvaluator(model, max_batch_size=6, max_wait_ms=50) as evaluator:
            def predict():
                barrier.wait(timeout=5)
                return evaluator(features, masks)

            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(predict) for _ in range(3)]
                results = [future.result(timeout=10) for future in futures]
            stats = evaluator.stats
        for actual in results:
            self.assert_prediction(actual, expected, masks)
        self.assertEqual(stats["network_evaluations"], 9)
        self.assertEqual(stats["submitted_requests"], 3)
        self.assertEqual(stats["inference_batches"], 2)
        self.assertEqual(stats["max_observed_batch_size"], 6)
        # A returned array must not alias another caller's response.
        results[0][0][:] = 123
        self.assert_prediction(results[1], expected, masks)

    def test_source_model_modes_weights_gradients_and_global_rng_are_preserved(self):
        model = self.make_model()
        model.train()
        model.policy_head.eval()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
        modes = [module.training for module in model.modules()]
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        contexts = []
        hook = model.register_forward_hook(lambda module, inputs, output:
                                           contexts.append((module is model, module.training,
                                                            torch.is_grad_enabled())))
        try:
            with EncodedBatchedEvaluator(model) as evaluator:
                evaluator(*encoded_rows())
        finally:
            hook.remove()
        self.assertEqual(contexts, [(False, False, False)])
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.125),
                                       rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)

    def test_queued_numpy_inputs_are_owned_snapshots(self):
        model = self.make_model()
        features, masks = encoded_rows()
        expected = self.direct(model, features, masks)
        blocked = Event()
        release = Event()
        accepted = Event()
        calls = []

        def delay_first(module, inputs):
            calls.append(1)
            if len(calls) == 1:
                blocked.set()
                if not release.wait(timeout=5):
                    raise AssertionError("test failed to release inference")

        hook = model.register_forward_pre_hook(delay_first)
        try:
            with EncodedBatchedEvaluator(model, max_batch_size=3) as evaluator:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(evaluator, features.copy(), masks.copy())
                    self.assertTrue(blocked.wait(timeout=5))
                    original_put = evaluator._requests.put

                    def record_put(item, *args, **kwargs):
                        original_put(item, *args, **kwargs)
                        if item is not evaluator._sentinel:
                            accepted.set()

                    evaluator._requests.put = record_put
                    second = pool.submit(evaluator, features, masks)
                    self.assertTrue(accepted.wait(timeout=5))
                    features[:] = 0
                    masks[:] = False
                    release.set()
                    first.result(timeout=5)
                    actual = second.result(timeout=5)
                self.assert_prediction(actual, expected, encoded_rows()[1])
        finally:
            release.set()
            hook.remove()

    def test_fatal_failure_releases_active_carry_queued_and_future_callers(self):
        model = self.make_model()
        barrier = Barrier(4)
        features, masks = encoded_rows()

        def fail(module, inputs):
            raise LookupError("encoded inference failed")

        hook = model.register_forward_pre_hook(fail)
        try:
            with EncodedBatchedEvaluator(model, max_batch_size=4, max_wait_ms=100) as evaluator:
                def predict():
                    barrier.wait(timeout=5)
                    return evaluator(features, masks)

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(predict) for _ in range(4)]
                    for future in futures:
                        with self.assertRaisesRegex(LookupError, "encoded inference failed"):
                            future.result(timeout=10)
                with self.assertRaisesRegex(LookupError, "encoded inference failed"):
                    evaluator(features, masks)
        finally:
            hook.remove()

    def test_invalid_inputs_do_not_poison_worker_and_close_is_idempotent(self):
        features, masks = encoded_rows()
        with EncodedBatchedEvaluator(self.make_model(), max_batch_size=3) as evaluator:
            invalid = [
                (features.tolist(), masks), (features.astype(np.float64), masks),
                (features, masks.astype(np.int8)), (features[:, 0], masks),
                (features, masks[:, :-1]), (features[:0], masks[:0]),
                (np.repeat(features, 2, axis=0), np.repeat(masks, 2, axis=0)),
                (features * np.float32(float("nan")), masks),
                (features + np.float32(2), masks), (features, np.zeros_like(masks)),
            ]
            for bad_features, bad_masks in invalid:
                with self.subTest(shape=getattr(bad_features, "shape", None)), \
                        self.assertRaises((TypeError, ValueError)):
                    evaluator(bad_features, bad_masks)
            evaluator(features, masks)
        evaluator.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            evaluator(features, masks)

    def test_invalid_network_outputs_fail_service(self):
        features, masks = encoded_rows()
        for corruption in ("logits_nan", "value_nan", "value_range", "shape", "integer"):
            with self.subTest(corruption=corruption):
                model = self.make_model()

                def corrupt(module, inputs, output):
                    logits, values = output
                    if corruption == "logits_nan":
                        logits[:] = float("nan")
                    elif corruption == "value_nan":
                        values[:] = float("nan")
                    elif corruption == "value_range":
                        values[:] = 2
                    elif corruption == "shape":
                        values = values.unsqueeze(1)
                    else:
                        logits = logits.to(torch.int32)
                    return logits, values

                hook = model.register_forward_hook(corrupt)
                try:
                    with EncodedBatchedEvaluator(model) as evaluator:
                        for _ in range(2):
                            with self.assertRaises((TypeError, ValueError)):
                                evaluator(features, masks)
                finally:
                    hook.remove()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_uses_cuda_tensors_and_matches_cpu(self):
        model = self.make_model()
        features, masks = encoded_rows()
        expected = self.direct(model, features, masks)
        observed = []
        hook = model.register_forward_hook(lambda module, inputs, output:
                                           observed.append((inputs[0].device.type,
                                                            output[0].device.type)))
        try:
            with EncodedBatchedEvaluator(model, device="cuda", max_batch_size=8) as evaluator:
                actual = evaluator(features, masks)
        finally:
            hook.remove()
        self.assertEqual(observed, [("cuda", "cuda")])
        self.assert_prediction(actual, expected, masks, rtol=2e-4, atol=2e-5)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
