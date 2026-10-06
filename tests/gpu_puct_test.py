"""CUDA PUCT search correctness, residency, deterministic seeds, and lifecycle."""

from copy import deepcopy
import unittest

import torch
import my_board_engine as engine

from kingdom_ai.encoding import encode_state, move_to_action, terminal_value
from kingdom_ai.gpu_puct import GpuPUCT, GpuPUCTOptions
from kingdom_ai.gpu_rules import GpuStateBatch
from kingdom_ai.inference import NeuralAgent
from kingdom_ai.model import PolicyValueNet

from gpu_rules_test import cpu_record, last_liberty_state, make_board


class FixedPolicyValue(torch.nn.Module):
    """An adversarial prior and exact color value expose shallow-search failures."""

    def __init__(self, preferred=0, rejected=None, color_value=False):
        super().__init__()
        logits = torch.zeros(82)
        logits[preferred] = 12.0
        if rejected is not None:
            logits[rejected] = -30.0
        self.register_buffer("logits", logits)
        self.color_value = color_value

    def forward(self, features):
        values = (2 * features[:, 5, 0, 0] - 1 if self.color_value
                  else torch.zeros(len(features), device=features.device))
        return self.logits.expand(len(features), -1), values


def transform_position(point, symmetry):
    row, col = point
    if symmetry >= 4:
        col = 8 - col
    for _ in range(symmetry % 4):
        row, col = col, 8 - row
    return row, col


class GpuSearchOptionsTest(unittest.TestCase):
    def test_tactical_and_fpu_options_require_explicit_valid_values(self):
        self.assertFalse(GpuPUCTOptions().tactical_checks)
        self.assertIsNone(GpuPUCTOptions().fpu_reduction)
        for value in (0, 1, None, "true"):
            with self.subTest(tactical=value), self.assertRaises(TypeError):
                GpuPUCTOptions(tactical_checks=value)
        for value in (True, -1, float("nan"), float("inf"), "0"):
            with self.subTest(fpu=value), self.assertRaises(ValueError):
                GpuPUCTOptions(fpu_reduction=value)
        for value in (0, 0.1, 0.25):
            self.assertEqual(GpuPUCTOptions(fpu_reduction=value).fpu_reduction, value)


def terminal_positions():
    passed = engine.State()
    passed.pass_turn()
    passed.pass_turn()
    captured = last_liberty_state(engine.Cell.Black)
    captured.place(4, 4)
    suicide = engine.State(make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
                           engine.Cell.Black)
    suicide.place(1, 1)
    return [passed, captured, suicide]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
class GpuPUCTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self, *, uniform=False, prefer_center=False):
        torch.manual_seed(829)
        model = PolicyValueNet(channels=8, residual_blocks=1)
        if uniform or prefer_center:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                if prefer_center:
                    model.policy_head[-1].bias[40] = 2.0
        return model

    def options(self, simulations=8, *, epsilon=0, seed=742):
        return GpuPUCTOptions(simulations=simulations, c_puct=1.5, dirichlet_alpha=0.3,
                              dirichlet_epsilon=epsilon, seed=seed)

    def assert_result(self, result, states, simulations):
        count = len(states)
        for field, dtype, shape in (
            ("actions", torch.int64, (count,)),
            ("visits", torch.int32, (count, 82)),
            ("policy", torch.float32, (count, 82)),
            ("values", torch.float32, (count,)),
            ("simulations", torch.int32, (count,)),
            ("network_evaluations", torch.int32, (count,)),
            ("nodes", torch.int32, (count,)),
            ("best_values", torch.float32, (count,)),
        ):
            tensor = getattr(result, field)
            self.assertEqual(tensor.device.type, "cuda", field)
            self.assertEqual(tensor.dtype, dtype, field)
            self.assertEqual(tuple(tensor.shape), shape, field)
        actions = result.actions.cpu().tolist()
        visits = result.visits.cpu()
        policies = result.policy.cpu()
        values = result.values.cpu()
        actual_simulations = result.simulations.cpu().tolist()
        evaluations = result.network_evaluations.cpu().tolist()
        nodes = result.nodes.cpu().tolist()
        best_values = result.best_values.cpu()
        self.assertTrue(torch.isfinite(policies).all().item())
        self.assertTrue(torch.isfinite(values).all().item())
        self.assertTrue((values.abs() <= 1).all().item())
        self.assertTrue(torch.isfinite(best_values).all().item())
        self.assertTrue((best_values.abs() <= 1).all().item())
        self.assertTrue((visits >= 0).all().item())
        for index, state in enumerate(states):
            mask = encode_state(state).legal_mask
            self.assertTrue((visits[index, ~mask] == 0).all().item())
            self.assertTrue((policies[index, ~mask] == 0).all().item())
            if state.result.finished():
                self.assertEqual(actions[index], -1)
                self.assertEqual(actual_simulations[index], 0)
                self.assertEqual(evaluations[index], 0)
                self.assertEqual(nodes[index], 0)
                self.assertEqual(best_values[index].item(), 0.0)
                self.assertEqual(visits[index].sum().item(), 0)
                self.assertEqual(policies[index].sum().item(), 0)
                self.assertEqual(values[index].item(), terminal_value(state))
            else:
                self.assertEqual(actual_simulations[index], simulations)
                self.assertEqual(visits[index].sum().item(), simulations)
                self.assertGreaterEqual(evaluations[index], 1)
                self.assertLessEqual(evaluations[index], simulations + 1)
                self.assertGreaterEqual(nodes[index], 1)
                self.assertLessEqual(nodes[index], simulations + 1)
                self.assertTrue(mask[actions[index]].item())
                self.assertGreater(visits[index, actions[index]].item(), 0)
                torch.testing.assert_close(policies[index], visits[index].float() / simulations,
                                           rtol=0, atol=1e-7)

    def test_uniform_no_noise_search_matches_cpp_oracle_exact_visits(self):
        model = self.make_model(uniform=True)
        states = [engine.State(), engine.State(neutral=None),
                  last_liberty_state(engine.Cell.Black), last_liberty_state(engine.Cell.White)]
        direct = NeuralAgent(deepcopy(model), device="cpu")

        def evaluate(state):
            prediction = direct.predict(state)
            return prediction.policy.tolist(), prediction.value

        expected = [engine.PUCT(engine.PUCTOptions(simulations=8, seed=742,
                                                  dirichlet_epsilon=0)).search(state, evaluate)
                    for state in states]
        batch = GpuStateBatch.from_engine(states)
        unchanged = batch.states.clone()
        result = GpuPUCT(model, self.options()).search(batch)
        self.assert_result(result, states, 8)
        torch.testing.assert_close(batch.states, unchanged, rtol=0, atol=0)
        for index, oracle in enumerate(expected):
            visits = torch.zeros(82, dtype=torch.int32)
            for item in oracle.moves:
                visits[move_to_action(item.move)] = item.visits
            torch.testing.assert_close(result.visits[index].cpu(), visits, rtol=0, atol=0)
            self.assertEqual(result.actions[index].item(), move_to_action(oracle.best_move))
            self.assertEqual(result.network_evaluations[index].item(), oracle.network_evaluations)
            self.assertAlmostEqual(result.values[index].item(), oracle.root_value, delta=1e-6)

    def test_forced_capture_wins_for_both_colors_and_input_stays_unchanged(self):
        states = [last_liberty_state(actor) for actor in (engine.Cell.Black, engine.Cell.White)]
        source_records = [cpu_record(state) for state in states]
        batch = GpuStateBatch.from_engine(states)
        result = GpuPUCT(self.make_model(prefer_center=True), self.options()).search(batch)
        self.assert_result(result, states, 8)
        self.assertEqual(result.actions.cpu().tolist(), [40, 40])
        self.assertEqual(batch.states.cpu().tolist(), source_records)
        continued = batch.clone()
        self.assertTrue(continued.play(result.actions).all().item())
        for state in states:
            actor = state.to_play
            self.assertTrue(state.place(4, 4).accepted())
            self.assertEqual(state.result.reason, engine.EndReason.Capture)
            self.assertEqual(state.result.winner, actor)
        self.assertEqual(continued.states.cpu().tolist(), [cpu_record(state) for state in states])

    def test_tactics_find_low_prior_capture_and_keep_encoded_rule_legality(self):
        states, winning = [], []
        for actor in (engine.Cell.Black, engine.Cell.White):
            for symmetry in range(8):
                own = tuple(transform_position(p, symmetry)
                            for p in ((0, 1), (1, 0), (2, 1)))
                other = (transform_position((1, 1), symmetry),)
                board = make_board(own, other) if actor == engine.Cell.Black else make_board(other, own)
                states.append(engine.State(board, actor))
                winning.append(transform_position((1, 2), symmetry))
        actions = [row * 9 + col for row, col in winning]
        model = FixedPolicyValue(preferred=80, rejected=11)
        baseline = GpuPUCT(model, self.options(simulations=4)).search(
            GpuStateBatch.from_engine(states[:1]))
        self.assertEqual(baseline.visits[0, 11].item(), 0)
        observations = []
        hook = model.register_forward_pre_hook(
            lambda module, inputs: observations.append(inputs[0].detach().cpu().clone()))
        try:
            batch = GpuStateBatch.from_engine(states + [terminal_positions()[0]])
            before = batch.states.clone()
            result = GpuPUCT(model, GpuPUCTOptions(
                simulations=4, tactical_checks=True, dirichlet_epsilon=0.25,
            )).search(batch)
        finally:
            hook.remove()
        self.assert_result(result, states + [terminal_positions()[0]], 4)
        self.assertEqual(result.actions[:-1].cpu().tolist(), actions)
        self.assertTrue((result.best_values[:-1] == 1).all().item())
        self.assertEqual(result.network_evaluations[:-1].cpu().tolist(), [1] * len(states))
        torch.testing.assert_close(batch.states, before, rtol=0, atol=0)
        for index, state in enumerate(states):
            torch.testing.assert_close(observations[0][index], encode_state(state).features,
                                       rtol=0, atol=0)

    def test_tactics_repair_actual_last_liberty_defense_in_all_symmetries(self):
        # A human reported this legal opening. White has precisely one reply
        # avoiding an immediate black capture: (5, 7), in one-based coordinates.
        opening = [(4, 8), (5, 8), (2, 8), (6, 8), (2, 4), (4, 4),
                   (6, 2), (4, 7), (3, 7), (5, 9), (4, 6)]
        states, safe_actions = [], []
        for symmetry in range(8):
            state = engine.State()
            for row, col in opening:
                self.assertTrue(state.place(*transform_position((row - 1, col - 1), symmetry)).accepted())
            row, col = transform_position((4, 6), symmetry)
            safe_actions.append(row * 9 + col)
            states.append(state)
        result = GpuPUCT(FixedPolicyValue(preferred=35), GpuPUCTOptions(
            simulations=8, tactical_checks=True, dirichlet_epsilon=0.25,
        )).search(GpuStateBatch.from_engine(states))
        self.assert_result(result, states, 8)
        self.assertEqual(result.actions.cpu().tolist(), safe_actions)
        for index, (state, expected) in enumerate(zip(states, safe_actions)):
            # Independently enumerate every C++ legal reply after the selected
            # defense, so the regression is not merely the CUDA implementation.
            continued = state.copy()
            self.assertTrue(continued.place(*divmod(expected, 9)).accepted())
            self.assertFalse(continued.result.finished())
            for reply in continued.legal_moves():
                end = continued.copy()
                self.assertTrue(end.play(reply).accepted())
                self.assertFalse(end.result.finished() and end.result.winner == continued.to_play)
            self.assertEqual(result.visits[index, expected].item(), 8)

    def test_tactics_handle_suicide_simultaneous_capture_pass_and_no_safe_fallback(self):
        suicide = engine.State(make_board(((0, 1), (6, 7), (7, 6), (8, 7)),
                               ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1), (7, 7))),
                               engine.Cell.Black)
        capture = engine.State(make_board(((0, 6), (1, 6), (1, 8), (2, 7)),
                               ((0, 7), (1, 7), (2, 8))), engine.Cell.Black)
        white_pass_win = engine.State()
        white_pass_win.pass_turn()
        black_pass_loss = engine.State(engine.Board(), engine.Cell.White)
        black_pass_loss.pass_turn()
        self.assertEqual(black_pass_loss.to_play, engine.Cell.Black)
        # With no empty points, every legal move loses immediately; preserve
        # the sole pass rather than producing an all-illegal search row.
        board = engine.Board([engine.Cell.Black] * 41 + [engine.Cell.White] * 40)
        forced_loss = engine.State(board, engine.Cell.White)
        forced_loss.pass_turn()
        states = [suicide, capture, white_pass_win, black_pass_loss, forced_loss]
        batch = GpuStateBatch.from_engine(states)
        _, legal = batch.encode()
        self.assertTrue(legal[0, 10].item())
        result = GpuPUCT(FixedPolicyValue(preferred=10), GpuPUCTOptions(
            simulations=8, tactical_checks=True, dirichlet_epsilon=0,
        )).search(batch)
        self.assert_result(result, states, 8)
        self.assertEqual(result.visits[0, 10].item(), 0)
        self.assertEqual(result.priors[0, 10].item(), 0)
        self.assertEqual(result.actions[1].item(), 8)
        self.assertEqual(result.best_values[1].item(), 1)
        self.assertEqual(result.actions[2].item(), 81)
        self.assertEqual(result.best_values[2].item(), 1)
        self.assertEqual(result.visits[3, 81].item(), 0)
        self.assertEqual(result.actions[4].item(), 81)
        self.assertEqual(result.best_values[4].item(), -1)
        # Rule legality still allows suicide even though search avoids it.
        self.assertTrue(batch.encode()[1][0, 10].item())

    def test_fpu_uses_node_player_value_to_explore_optimistic_unvisited_moves(self):
        model = FixedPolicyValue(preferred=52, color_value=True)
        state = GpuStateBatch.initial(16)
        legacy = GpuPUCT(model, GpuPUCTOptions(
            simulations=128, dirichlet_epsilon=0.25, seed=103,
        )).search(state)
        optimistic = GpuPUCT(model, GpuPUCTOptions(
            simulations=128, dirichlet_epsilon=0.25, seed=103, fpu_reduction=0.0,
        )).search(state)
        self.assert_result(optimistic, [engine.State()] * 16, 128)
        self.assertGreater((optimistic.visits > 0).sum().item(), (legacy.visits > 0).sum().item())
        torch.testing.assert_close(optimistic.priors, legacy.priors, rtol=0, atol=0)
        self.assertTrue((legacy.visits[:, 52] == 128).all().item())
        self.assertTrue((optimistic.visits[:, 52] < 128).all().item())

    def test_terminal_search_uses_exact_outcome_without_model_forward(self):
        model = self.make_model()

        def reject_forward(module, inputs):
            raise AssertionError("Terminal rows must bypass the policy/value model")

        hook = model.register_forward_pre_hook(reject_forward)
        try:
            states = terminal_positions()
            batch = GpuStateBatch.from_engine(states)
            result = GpuPUCT(model, self.options()).search(batch)
            self.assert_result(result, states, 8)
            self.assertEqual(batch.states.cpu().tolist(), [cpu_record(state) for state in states])
        finally:
            hook.remove()

    def test_mixed_batch_only_forwards_active_cuda_rows(self):
        model = self.make_model()
        states = [engine.State(), terminal_positions()[0], engine.State(neutral=None)]
        contexts = []

        def record(module, inputs, output):
            contexts.append((inputs[0].shape, inputs[0].device.type,
                             next(module.parameters()).device.type,
                             output[0].device.type, output[1].device.type,
                             torch.is_grad_enabled(), module.training))

        hook = model.register_forward_hook(record)
        try:
            batch = GpuStateBatch.from_engine(states)
            result = GpuPUCT(model, self.options(simulations=4)).search(batch)
        finally:
            hook.remove()
        self.assert_result(result, states, 4)
        self.assertTrue(contexts)
        self.assertEqual(tuple(contexts[0][0]), (2, 10, 9, 9))
        self.assertTrue(all(tuple(item[0][1:]) == (10, 9, 9) for item in contexts))
        self.assertTrue(all(0 < item[0][0] <= 2 for item in contexts))
        self.assertTrue(all(item[1:5] == ("cuda",) * 4 for item in contexts))
        self.assertTrue(all(item[5:] == (False, False) for item in contexts))

    def test_source_mixed_modes_weights_devices_and_gradients_are_preserved(self):
        model = self.make_model()
        model.train()
        model.trunk[0].eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        modes = [module.training for module in model.modules()]
        weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        states = [engine.State(), engine.State(neutral=None)]
        batch = GpuStateBatch.from_engine(states)
        result = GpuPUCT(model, self.options(simulations=4)).search(batch)
        self.assert_result(result, states, 4)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    def test_root_noise_repeats_with_the_same_fresh_searcher_seed(self):
        model = self.make_model()
        batch = GpuStateBatch.from_engine([engine.State(), engine.State(neutral=None)])
        original = batch.states.clone()
        first = GpuPUCT(model, self.options(simulations=16, epsilon=0.25, seed=6502)).search(batch)
        second = GpuPUCT(model, self.options(simulations=16, epsilon=0.25, seed=6502)).search(batch)
        for field in ("actions", "visits", "policy", "values", "simulations", "network_evaluations"):
            torch.testing.assert_close(getattr(first, field), getattr(second, field), rtol=0, atol=0)
        torch.testing.assert_close(batch.states, original, rtol=0, atol=0)
        self.assert_result(first, [engine.State(), engine.State(neutral=None)], 16)

    def test_reset_seed_restarts_only_the_dedicated_root_noise_generator(self):
        model = self.make_model()
        batch = GpuStateBatch.from_engine([engine.State(), engine.State(neutral=None)])
        searcher = GpuPUCT(model, self.options(simulations=8, epsilon=0.25, seed=6502))
        cpu_rng = torch.get_rng_state().clone()
        cuda_rngs = [state.clone() for state in torch.cuda.get_rng_state_all()]
        first = searcher.search(batch)
        second = searcher.search(batch)
        self.assertFalse(torch.equal(first.priors, second.priors))
        searcher.reset_seed()
        restarted = searcher.search(batch)
        for field in ("actions", "visits", "policy", "values", "simulations", "network_evaluations",
                      "priors", "best_values", "nodes"):
            torch.testing.assert_close(getattr(first, field), getattr(restarted, field), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rngs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for invalid in (-1, 2**64, True, 1.5):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                searcher.reset_seed(invalid)

    def test_nondefault_stream_orders_state_producers_search_and_result_consumers(self):
        model = self.make_model(uniform=True)
        states = [engine.State(), engine.State(neutral=None)]
        original_records = [cpu_record(state) for state in states]
        stream = torch.cuda.Stream()
        seen_streams = []

        def record_stream(module, inputs, output):
            seen_streams.append(torch.cuda.current_stream().cuda_stream)

        hook = model.register_forward_hook(record_stream)
        try:
            with torch.cuda.stream(stream):
                batch = GpuStateBatch.from_engine(states)
                batch.play(torch.tensor([0, 80], device="cuda", dtype=torch.int32))
                searcher = GpuPUCT(model, self.options(simulations=4))
                result = searcher.search(batch)
                summed_visits = result.visits.sum(dim=1)
                selected_visits = result.visits.gather(1, result.actions[:, None]).flatten()
                continued = batch.clone()
                accepted = continued.play(result.actions)
                continued_records = continued.states.clone()
            stream.synchronize()
        finally:
            hook.remove()
        self.assertEqual([cpu_record(state) for state in states], original_records)
        self.assertTrue(seen_streams)
        self.assertTrue(all(value == stream.cuda_stream for value in seen_streams))
        self.assertEqual(summed_visits.cpu().tolist(), [4, 4])
        self.assertTrue((selected_visits > 0).all().item())
        self.assertEqual(accepted.cpu().tolist(), [True, True])
        for state, action in zip(states, (0, 80)):
            self.assertTrue(state.play(engine.Move.place(*divmod(action, 9))).accepted())
        self.assert_result(result, states, 4)
        self.assertEqual(batch.states.cpu().tolist(), [cpu_record(state) for state in states])
        for state, action in zip(states, result.actions.cpu().tolist()):
            move = engine.Move.pass_turn() if action == 81 else engine.Move.place(*divmod(action, 9))
            self.assertTrue(state.is_legal(move))
            self.assertTrue(state.play(move).accepted())
        self.assertEqual(continued_records.cpu().tolist(), [cpu_record(state) for state in states])

    def test_completed_gpu_selfplay_preserves_legal_actions_and_turn_outcomes(self):
        states = [engine.State(), engine.State(neutral=None),
                  engine.State(neutral=engine.Position(0, 8)), engine.State()]
        batch = GpuStateBatch.from_engine(states)
        searcher = GpuPUCT(self.make_model(uniform=True), self.options(simulations=4))
        for _ in range(164):
            if all(state.result.finished() for state in states):
                break
            result = searcher.search(batch)
            self.assert_result(result, states, 4)
            actions = result.actions.cpu().tolist()
            expected = []
            for state, action in zip(states, actions):
                if state.result.finished():
                    expected.append(False)
                else:
                    move = engine.Move.pass_turn() if action == 81 else engine.Move.place(*divmod(action, 9))
                    self.assertTrue(state.is_legal(move))
                    expected.append(state.play(move).accepted())
            accepted = batch.play(result.actions)
            self.assertEqual(accepted.cpu().tolist(), expected)
            self.assertEqual(batch.states.cpu().tolist(), [cpu_record(state) for state in states])
        self.assertTrue(all(state.result.finished() for state in states))

    def test_wrong_search_input_and_nondefault_rules_are_rejected(self):
        searcher = GpuPUCT(self.make_model(), self.options())
        for invalid in (engine.State(), object(), torch.zeros((2, 170), device="cuda", dtype=torch.int32)):
            with self.subTest(kind=type(invalid)), self.assertRaises((TypeError, ValueError)):
                searcher.search(invalid)
        with self.assertRaises(ValueError):
            GpuStateBatch.from_engine([engine.State(engine.GameRules(stones_per_player=40))])

    def test_nonfinite_model_outputs_raise_and_preserve_input(self):
        for field in ("logits", "values"):
            with self.subTest(field=field):
                model = self.make_model()

                def corrupt(module, inputs, output):
                    logits, values = output
                    if field == "logits":
                        logits = logits.clone()
                        logits[:, 0] = float("nan")
                    else:
                        values = values.clone()
                        values[:] = float("nan")
                    return logits, values

                hook = model.register_forward_hook(corrupt)
                try:
                    batch = GpuStateBatch.initial(2)
                    before = batch.states.clone()
                    with self.assertRaises(ValueError):
                        GpuPUCT(model, self.options()).search(batch)
                    torch.testing.assert_close(batch.states, before, rtol=0, atol=0)
                finally:
                    hook.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
