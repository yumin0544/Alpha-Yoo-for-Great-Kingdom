"""Batched neural inference, parallel C++ PUCT, and optional real CUDA checks.

Run with the learning dependencies installed:
    python tests/batching_test.py

These checks verify computation and lifecycle behavior, not speed thresholds.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
from threading import Barrier
import unittest

import torch

import my_board_engine as engine
from kingdom_ai.batching import BatchedEvaluator
from kingdom_ai.encoding import ACTION_SIZE, encode_state, move_to_action, terminal_value
from kingdom_ai.inference import NeuralAgent
from kingdom_ai.model import PolicyValueNet


def state_signature(state):
    score = state.score()
    return (
        state.board.to_string(), state.to_play, state.consecutive_passes,
        state.remaining_stones(engine.Cell.Black),
        state.remaining_stones(engine.Cell.White), tuple(state.ownership),
        score.black, score.white, state.result.winner, state.result.reason,
    )


def sample_states():
    initial = engine.State()
    placed = engine.State()
    for row, col in ((0, 0), (8, 8), (2, 3)):
        if not placed.place(row, col).accepted():
            raise AssertionError("Invalid batching fixture")
    passed = placed.copy()
    if not passed.pass_turn().accepted():
        raise AssertionError("Invalid pass fixture")
    return [initial, placed, passed, engine.State(neutral=None)]


def last_liberty_state(actor):
    """Two 40-stone groups share their final liberty at the center."""
    cells = []
    for row in range(9):
        for col in range(9):
            if col < 4 or (col == 4 and row < 4):
                cells.append(engine.Cell.Black)
            elif col > 4 or (col == 4 and row > 4):
                cells.append(engine.Cell.White)
            else:
                cells.append(engine.Cell.Empty)
    return engine.State(engine.Board(cells), actor)


def terminal_states():
    passed = engine.State()
    passed.pass_turn()
    passed.pass_turn()
    captured = last_liberty_state(engine.Cell.Black)
    if not captured.place(4, 4).accepted():
        raise AssertionError("Invalid capture fixture")
    board = engine.Board(neutral=None)
    for points, color in (
        (((0, 1),), engine.Cell.Black),
        (((0, 0), (0, 2), (1, 0), (1, 2), (2, 1)), engine.Cell.White),
    ):
        for row, col in points:
            if not board.place(engine.Position(row, col), color):
                raise AssertionError("Invalid suicide fixture")
    suicide = engine.State(board, engine.Cell.Black)
    if not suicide.place(1, 1).accepted():
        raise AssertionError("Invalid suicide fixture")
    return [passed, captured, suicide]


class BatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        torch.manual_seed(914)
        return PolicyValueNet(channels=8, residual_blocks=1)

    def assert_prediction(self, state, result, expected=None, *, rtol=1e-5, atol=1e-6):
        policy, value = result
        self.assertIsInstance(policy, list)
        self.assertEqual(len(policy), ACTION_SIZE)
        self.assertTrue(all(math.isfinite(item) and item >= 0 for item in policy))
        self.assertIsInstance(value, float)
        self.assertTrue(math.isfinite(value))
        self.assertLessEqual(abs(value), 1.0)
        policy_tensor = torch.tensor(policy)
        mask = encode_state(state).legal_mask
        self.assertTrue((policy_tensor[~mask] == 0).all())
        self.assertAlmostEqual(sum(policy), 1.0, places=6)
        if expected is not None:
            torch.testing.assert_close(policy_tensor, expected.policy, rtol=rtol, atol=atol)
            self.assertAlmostEqual(value, expected.value, delta=atol + rtol * abs(expected.value))

    def test_single_request_matches_direct_neural_agent_and_preserves_state(self):
        model = self.make_model()
        direct = NeuralAgent(deepcopy(model), device="cpu")
        states = sample_states()
        signatures = [state_signature(state) for state in states]
        with BatchedEvaluator(model, device="cpu", max_batch_size=4, max_wait_ms=0) as evaluator:
            for state in states:
                self.assert_prediction(state, evaluator(state), direct.predict(state))
            stats = evaluator.stats
        self.assertEqual([state_signature(state) for state in states], signatures)
        self.assertEqual(stats["network_evaluations"], len(states))
        self.assertEqual(stats["inference_batches"], len(states))
        self.assertEqual(stats["mean_batch_size"], 1.0)
        self.assertEqual(stats["max_observed_batch_size"], 1)
        self.assertGreater(stats["batch_seconds"], 0)

    def test_parallel_requests_are_batched_and_keep_their_own_results(self):
        model = self.make_model()
        states = sample_states()
        direct = NeuralAgent(deepcopy(model))
        expected = [direct.predict(state) for state in states]
        signatures = [state_signature(state) for state in states]
        barrier = Barrier(len(states))
        seen_batch_sizes = []

        def record_batch(module, inputs, output):
            seen_batch_sizes.append(inputs[0].shape[0])

        hook = model.register_forward_hook(record_batch)
        try:
            with BatchedEvaluator(model, max_batch_size=4, max_wait_ms=500) as evaluator:
                def evaluate(state):
                    barrier.wait(timeout=5)
                    return evaluator(state)

                with ThreadPoolExecutor(max_workers=len(states)) as pool:
                    futures = [pool.submit(evaluate, state) for state in states]
                    actual = [future.result(timeout=10) for future in futures]
                stats = evaluator.stats
        finally:
            hook.remove()
        for state, result, prediction in zip(states, actual, expected):
            self.assert_prediction(state, result, prediction)
        self.assertEqual([state_signature(state) for state in states], signatures)
        self.assertGreater(max(seen_batch_sizes), 1)
        self.assertEqual(sum(seen_batch_sizes), len(states))
        self.assertEqual(stats["network_evaluations"], len(states))
        self.assertEqual(stats["inference_batches"], len(seen_batch_sizes))
        self.assertEqual(stats["max_observed_batch_size"], max(seen_batch_sizes))
        self.assertAlmostEqual(stats["mean_batch_size"], len(states) / len(seen_batch_sizes))

    def test_dedicated_inference_copy_preserves_source_modes_weights_and_gradients(self):
        model = self.make_model()
        model.train()
        model.trunk[0].eval()  # Preserve a mixed mode source too.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        modes = [module.training for module in model.modules()]
        weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        inference_contexts = []

        def record_context(module, inputs, output):
            inference_contexts.append((module is model, module.training,
                                       torch.is_grad_enabled(), inputs[0].device.type))

        hook = model.register_forward_hook(record_context)
        try:
            with BatchedEvaluator(model, device="cpu", max_wait_ms=0) as evaluator:
                self.assertIsNot(evaluator.model, model)
                self.assertFalse(evaluator.model.training)
                self.assert_prediction(engine.State(), evaluator(engine.State()))
        finally:
            hook.remove()
        self.assertEqual(inference_contexts, [(False, False, False, "cpu")])
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
            self.assertEqual(tensor.device.type, "cpu")
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_terminal_states_bypass_the_network_with_exact_current_player_values(self):
        model = self.make_model()

        def reject_forward(module, inputs):
            raise AssertionError("Terminal states must bypass network inference")

        hook = model.register_forward_pre_hook(reject_forward)
        try:
            with BatchedEvaluator(model, max_wait_ms=0) as evaluator:
                for state in terminal_states():
                    signature = state_signature(state)
                    policy, value = evaluator(state)
                    self.assertEqual(policy, [0.0] * ACTION_SIZE)
                    self.assertEqual(value, terminal_value(state))
                    self.assertEqual(state_signature(state), signature)
                self.assertEqual(evaluator.stats["network_evaluations"], 0)
                self.assertEqual(evaluator.stats["inference_batches"], 0)
        finally:
            hook.remove()

    def test_independent_parallel_puct_searches_preserve_states_and_complete_captures(self):
        model = self.make_model()
        # Stable nonzero center prior makes the forced winning continuation
        # independent of tiny numerical differences among batched convolutions.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_head[-1].bias[40] = 2.0
        states = [last_liberty_state(actor) for actor in (
            engine.Cell.Black, engine.Cell.White, engine.Cell.Black, engine.Cell.White,
        )]
        signatures = [state_signature(state) for state in states]
        barrier = Barrier(len(states))
        with BatchedEvaluator(model, max_batch_size=4, max_wait_ms=50) as evaluator:
            def search(state):
                searcher = engine.PUCT(engine.PUCTOptions(simulations=8, seed=971))
                barrier.wait(timeout=5)
                return searcher.search(state, evaluator)

            with ThreadPoolExecutor(max_workers=len(states)) as pool:
                futures = [pool.submit(search, state) for state in states]
                results = [future.result(timeout=15) for future in futures]
            total_evaluations = sum(result.network_evaluations for result in results)
            self.assertEqual(evaluator.stats["network_evaluations"], total_evaluations)
        self.assertEqual([state_signature(state) for state in states], signatures)
        for state, result in zip(states, results):
            self.assertEqual(result.simulations, 8)
            self.assertEqual(sum(item.visits for item in result.moves), 8)
            self.assertTrue(all(state.is_legal(item.move) for item in result.moves))
            self.assertTrue(state.is_legal(result.best_move))
            self.assertEqual(move_to_action(result.best_move), 40)
            actor = state.to_play
            outcome = state.play(result.best_move)
            self.assertTrue(outcome.accepted())
            self.assertTrue(state.result.finished())
            self.assertEqual(state.result.reason, engine.EndReason.Capture)
            self.assertEqual(state.result.winner, actor)

    def test_fatal_model_error_reaches_all_parallel_callers_and_future_requests(self):
        model = self.make_model()

        def fail_forward(module, inputs):
            raise LookupError("test inference failure")

        hook = model.register_forward_pre_hook(fail_forward)
        barrier = Barrier(6)
        try:
            with BatchedEvaluator(model, max_batch_size=2, max_wait_ms=500) as evaluator:
                def evaluate():
                    barrier.wait(timeout=5)
                    return evaluator(engine.State())

                with ThreadPoolExecutor(max_workers=6) as pool:
                    futures = [pool.submit(evaluate) for _ in range(6)]
                    for future in futures:
                        with self.assertRaisesRegex(LookupError, "test inference failure"):
                            future.result(timeout=10)
                with self.assertRaisesRegex(LookupError, "test inference failure"):
                    evaluator(engine.State())
        finally:
            hook.remove()

    def test_nonfinite_network_output_propagates_instead_of_returning_invalid_policy(self):
        for field in ("logits", "values"):
            with self.subTest(field=field):
                model = self.make_model()

                def corrupt_output(module, inputs, output):
                    logits, values = output
                    if field == "logits":
                        logits = logits.clone()
                        logits[:, 0] = float("nan")
                    else:
                        values = values.clone()
                        values[:] = float("nan")
                    return logits, values

                hook = model.register_forward_hook(corrupt_output)
                try:
                    with BatchedEvaluator(model, max_wait_ms=0) as evaluator:
                        with self.assertRaises(ValueError):
                            evaluator(engine.State())
                        with self.assertRaises(ValueError):
                            evaluator(engine.State())
                finally:
                    hook.remove()

    def test_close_is_idempotent_and_rejects_all_new_requests(self):
        evaluator = BatchedEvaluator(self.make_model(), max_wait_ms=0)
        self.assert_prediction(engine.State(), evaluator(engine.State()))
        evaluator.close()
        evaluator.close()
        for state in [engine.State(), *terminal_states()]:
            with self.subTest(reason=state.result.reason), self.assertRaises(RuntimeError):
                evaluator(state)

    def test_bad_states_and_unrepresented_rules_fail_without_poisoning_service(self):
        with BatchedEvaluator(self.make_model(), max_wait_ms=0) as evaluator:
            with self.assertRaises(TypeError):
                evaluator(object())
            with self.assertRaises(ValueError):
                evaluator(engine.State(engine.GameRules(stones_per_player=1)))
            self.assert_prediction(engine.State(), evaluator(engine.State()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_uses_cuda_tensors_and_matches_cpu_for_identical_weights(self):
        model = self.make_model()
        states = sample_states()
        with BatchedEvaluator(model, device="cpu", max_wait_ms=0) as cpu:
            expected = [cpu(state) for state in states]
        observed_devices = []

        def record_cuda(module, inputs, output):
            observed_devices.append((inputs[0].device.type,
                                     next(module.parameters()).device.type,
                                     output[0].device.type, output[1].device.type))

        hook = model.register_forward_hook(record_cuda)
        try:
            barrier = Barrier(len(states))
            with BatchedEvaluator(model, device="cuda", max_batch_size=4,
                                  max_wait_ms=500) as gpu:
                def evaluate(state):
                    barrier.wait(timeout=5)
                    return gpu(state)

                with ThreadPoolExecutor(max_workers=len(states)) as pool:
                    futures = [pool.submit(evaluate, state) for state in states]
                    actual = [future.result(timeout=20) for future in futures]
                self.assertGreater(gpu.stats["max_observed_batch_size"], 1)
        finally:
            hook.remove()
        self.assertTrue(observed_devices)
        self.assertTrue(all(item == ("cuda",) * 4 for item in observed_devices))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        for state, (cpu_policy, cpu_value), result in zip(states, expected, actual):
            self.assert_prediction(state, result)
            gpu_policy, gpu_value = result
            torch.testing.assert_close(torch.tensor(gpu_policy), torch.tensor(cpu_policy),
                                       rtol=2e-4, atol=2e-5)
            self.assertAlmostEqual(cpu_value, gpu_value, delta=2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
