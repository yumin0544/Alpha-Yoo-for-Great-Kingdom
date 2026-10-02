"""PUCT value perspectives, C++ callbacks and PyTorch self-play integration.

Run with the learning dependencies installed: python tests/puct_test.py
"""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import tempfile
import unittest

import torch

import my_board_engine as engine
from kingdom_ai import (
    ACTION_SIZE, PASS_ACTION, PUCT, PUCTOptions, PolicyValueNet,
    collect_puct_game, encode_state, load_model, make_batch, move_to_action,
    sample_visits, save_model, terminal_value, train_step, visit_policy,
)


def make_board(black=(), white=(), neutral=None):
    board = engine.Board(neutral=neutral)
    for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), color):
                raise AssertionError("Invalid PUCT fixture")
    return board


def last_liberty_state(actor):
    """Two 40-stone groups share their only liberty at the empty center."""
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


def capture_state():
    return engine.State(
        make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),)), engine.Cell.Black
    )


def suicide_state():
    return engine.State(
        make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
        engine.Cell.Black,
    )


def uniform_evaluator(state):
    if state.result.finished():
        raise AssertionError("Terminal positions must not be sent to the network")
    return [1.0] * ACTION_SIZE, 0.0


def one_hot(action, value=0.0):
    def evaluate(state):
        if state.result.finished():
            raise AssertionError("Terminal positions must bypass inference")
        policy = [0.0] * ACTION_SIZE
        policy[action] = 1.0
        return policy, value
    return evaluate


def state_signature(state):
    score = state.score()
    return (
        state.board.to_string(), state.to_play, state.consecutive_passes,
        state.remaining_stones(engine.Cell.Black),
        state.remaining_stones(engine.Cell.White), tuple(state.ownership),
        score.black, score.white, state.result,
    )


def search_signature(result):
    return (
        None if result.best_move is None else move_to_action(result.best_move),
        result.simulations, result.nodes, result.network_evaluations,
        result.root_value, result.best_value,
        tuple((move_to_action(item.move), item.prior, item.visits, item.value)
              for item in result.moves),
    )


class PUCTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def check_result(self, state, result, simulations):
        legal_actions = {move_to_action(move) for move in state.legal_moves()}
        self.assertIsNotNone(result.best_move)
        self.assertTrue(state.is_legal(result.best_move))
        self.assertEqual(result.simulations, simulations)
        self.assertEqual(sum(item.visits for item in result.moves), simulations)
        self.assertEqual({move_to_action(item.move) for item in result.moves}, legal_actions)
        self.assertEqual(len(result.moves), len(legal_actions))
        self.assertGreaterEqual(result.nodes, 2)
        self.assertLessEqual(result.nodes, simulations + 1)
        self.assertGreaterEqual(result.network_evaluations, 1)
        self.assertTrue(math.isfinite(result.elapsed_seconds))
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertAlmostEqual(sum(item.prior for item in result.moves), 1, places=12)
        self.assertTrue(math.isfinite(result.root_value))
        self.assertTrue(math.isfinite(result.best_value))
        self.assertLessEqual(abs(result.root_value), 1)
        self.assertLessEqual(abs(result.best_value), 1)
        best = next(item for item in result.moves if item.move == result.best_move)
        self.assertGreater(best.visits, 0)
        self.assertEqual(best.value, result.best_value)
        self.assertTrue(all(best.visits >= item.visits for item in result.moves))
        for item in result.moves:
            self.assertTrue(math.isfinite(item.prior))
            self.assertGreaterEqual(item.prior, 0)
            self.assertTrue(math.isfinite(item.value))
            self.assertLessEqual(abs(item.value), 1)
            self.assertGreaterEqual(item.visits, 0)
            if item.visits == 0:
                self.assertEqual(item.value, 0)

    def test_budget_all_legal_candidates_and_original_state_preserved(self):
        state = engine.State()
        state.place(0, 0)
        state.pass_turn()
        before = state_signature(state)
        result = engine.PUCT(engine.PUCTOptions(simulations=96)).search(state, uniform_evaluator)
        self.check_result(state, result, 96)
        self.assertEqual(state_signature(state), before)
        policy = visit_policy(result, state)
        self.assertAlmostEqual(policy.sum().item(), 1, places=6)
        self.assertTrue((policy[~encode_state(state).legal_mask] == 0).all())

    def test_child_value_sign_is_flipped_to_root_player(self):
        calls = []

        def evaluate(state):
            calls.append(state.to_play)
            policy = [0.0] * ACTION_SIZE
            policy[0 if state.to_play == engine.Cell.Black else 1] = 1.0
            return policy, 0.25 if state.to_play == engine.Cell.Black else 0.75

        result = engine.PUCT(engine.PUCTOptions(simulations=1)).search(engine.State(), evaluate)
        self.assertEqual(calls, [engine.Cell.Black, engine.Cell.White])
        self.assertEqual(move_to_action(result.best_move), 0)
        self.assertEqual(result.best_value, -0.75)
        self.assertEqual(result.root_value, -0.75)
        self.assertEqual(result.network_evaluations, 2)
        self.assertEqual(result.simulations, 1)

    def test_first_selection_uses_network_prior_and_masks_illegal_mass(self):
        state = engine.State()
        policy = [0.0] * ACTION_SIZE
        policy[40] = 1e100  # Neutral point; this mass must be discarded.
        policy[8] = 4.0
        policy[9] = 1.0

        def evaluate(snapshot):
            if snapshot.to_play == engine.Cell.Black:
                return policy, 0.0
            return [1.0] * ACTION_SIZE, 0.0

        result = engine.PUCT(engine.PUCTOptions(simulations=1)).search(state, evaluate)
        self.check_result(state, result, 1)
        self.assertEqual(move_to_action(result.best_move), 8)
        priors = {move_to_action(item.move): item.prior for item in result.moves}
        self.assertNotIn(40, priors)
        self.assertAlmostEqual(priors[8], 0.8)
        self.assertAlmostEqual(priors[9], 0.2)
        self.assertTrue(all(prior == 0 for action, prior in priors.items() if action not in (8, 9)))

    def test_capture_terminal_leaf_uses_exact_win_and_no_inference(self):
        state = capture_state()
        calls = []

        def evaluate(snapshot):
            calls.append(snapshot.result.finished())
            return one_hot(21, value=-0.9)(snapshot)

        result = engine.PUCT(engine.PUCTOptions(simulations=4)).search(state, evaluate)
        self.check_result(state, result, 4)
        self.assertEqual(move_to_action(result.best_move), 21)
        self.assertEqual(result.best_value, 1.0)
        self.assertEqual(result.root_value, 1.0)
        self.assertEqual(result.network_evaluations, 1)
        self.assertEqual(calls, [False])
        self.assertEqual(result.nodes, 2)
        state.play(result.best_move)
        self.assertEqual(state.result.reason, engine.EndReason.Capture)
        self.assertEqual(state.result.winner, engine.Cell.Black)

    def test_legal_suicide_returns_exact_loss_for_mover(self):
        state = suicide_state()
        self.assertTrue(state.is_legal(engine.Move.place(1, 1)))
        result = engine.PUCT(engine.PUCTOptions(simulations=1)).search(state, one_hot(10, 0.99))
        self.check_result(state, result, 1)
        self.assertEqual(move_to_action(result.best_move), 10)
        self.assertEqual(result.root_value, -1.0)
        self.assertEqual(result.best_value, -1.0)
        self.assertEqual(result.network_evaluations, 1)
        state.play(result.best_move)
        self.assertEqual(state.result.reason, engine.EndReason.Suicide)
        self.assertEqual(state.result.winner, engine.Cell.White)

    def test_second_pass_is_exact_score_win_and_bypasses_network(self):
        state = engine.State()
        state.pass_turn()
        self.assertEqual(state.to_play, engine.Cell.White)
        result = engine.PUCT(engine.PUCTOptions(simulations=4)).search(state, one_hot(PASS_ACTION, -0.8))
        self.assertTrue(result.best_move.is_pass())
        self.assertEqual(result.best_value, 1.0)
        self.assertEqual(result.root_value, 1.0)
        self.assertEqual(result.network_evaluations, 1)
        state.play(result.best_move)
        self.assertEqual(state.result.reason, engine.EndReason.TwoPasses)
        self.assertEqual(state.result.winner, engine.Cell.White)

    def test_opponent_selects_own_capture_after_root_pass(self):
        # At White's root a second Black pass would let White win by score.
        # Black must instead capture: this detects a missing alternating sign
        # that would make Black optimize White's outcome at the next level.
        for actor in (engine.Cell.Black, engine.Cell.White):
            with self.subTest(actor=actor):
                state = last_liberty_state(actor)
                self.assertEqual(len(state.legal_moves()), 2)
                result = engine.PUCT(engine.PUCTOptions(simulations=256, c_puct=1.5)).search(
                    state, uniform_evaluator
                )
                self.check_result(state, result, 256)
                self.assertEqual(move_to_action(result.best_move), 40)
                self.assertEqual(result.best_value, 1.0)
                passed = next(item for item in result.moves if item.move.is_pass())
                self.assertGreater(passed.visits, 1)
                self.assertLess(passed.value, -0.5)
                state.play(result.best_move)
                self.assertEqual(state.result.winner, actor)

    def test_forced_pass_exhausted_stocks_and_analysis_rules(self):
        state = engine.State(engine.GameRules(stones_per_player=1))
        state.place(0, 0)
        state.place(8, 8)
        # The first forced Black pass reaches a nonterminal White node with NN
        # value zero. Subsequent traversals resolve the second-pass loss exactly.
        for actor, expected in ((engine.Cell.Black, -7 / 8), (engine.Cell.White, 1.0)):
            self.assertEqual(state.to_play, actor)
            result = engine.PUCT(engine.PUCTOptions(simulations=8)).search(state, uniform_evaluator)
            self.check_result(state, result, 8)
            self.assertEqual(len(result.moves), 1)
            self.assertTrue(result.best_move.is_pass())
            self.assertEqual(result.best_value, expected)
            self.assertEqual(result.moves[0].prior, 1.0)
            state.play(result.best_move)
        self.assertTrue(state.result.finished())
        forbidden = engine.State(suicide_state().board, engine.Cell.Black,
                                 engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden))
        result = engine.PUCT(engine.PUCTOptions(simulations=8)).search(forbidden, uniform_evaluator)
        self.check_result(forbidden, result, 8)
        self.assertNotIn(10, [move_to_action(item.move) for item in result.moves])

    def test_terminal_roots_return_exact_value_without_callbacks(self):
        captured = capture_state()
        captured.place(2, 3)
        suicide = suicide_state()
        suicide.place(1, 1)
        passed = engine.State()
        passed.pass_turn()
        passed.pass_turn()

        def forbidden_callback(state):
            raise AssertionError("Terminal root must bypass the evaluator")

        searcher = engine.PUCT(engine.PUCTOptions(simulations=8))
        for state in (captured, suicide, passed):
            with self.subTest(reason=state.result.reason):
                before = state_signature(state)
                result = searcher.search(state, forbidden_callback)
                self.assertIsNone(result.best_move)
                self.assertEqual(result.moves, [])
                self.assertEqual(result.simulations, 0)
                self.assertEqual(result.nodes, 0)
                self.assertEqual(result.network_evaluations, 0)
                self.assertEqual(result.root_value, terminal_value(state))
                self.assertEqual(state_signature(state), before)

    def test_noise_is_seeded_normalized_and_limited_to_legal_root_actions(self):
        state = engine.State()
        options = engine.PUCTOptions(simulations=32, seed=318, dirichlet_epsilon=0.25)
        first_searcher = engine.PUCT(options)
        first = first_searcher.search(state, uniform_evaluator)
        second = engine.PUCT(options).search(state, uniform_evaluator)
        self.check_result(state, first, 32)
        self.assertEqual(search_signature(first), search_signature(second))
        self.assertGreater(len({item.prior for item in first.moves}), 1)
        baseline = engine.PUCT(engine.PUCTOptions(simulations=32)).search(state, uniform_evaluator)
        self.assertEqual(len({item.prior for item in baseline.moves}), 1)
        continued = first_searcher.search(state, uniform_evaluator)
        self.assertNotEqual([item.prior for item in first.moves], [item.prior for item in continued.moves])
        no_noise = engine.PUCT(engine.PUCTOptions(simulations=32, seed=912))
        self.assertEqual(search_signature(no_noise.search(state, uniform_evaluator)),
                         search_signature(no_noise.search(state, uniform_evaluator)))

    def test_arena_growth_preserves_values_and_action_links(self):
        state = engine.State()
        result = engine.PUCT(engine.PUCTOptions(simulations=320)).search(state, uniform_evaluator)
        self.check_result(state, result, 320)
        self.assertGreater(result.nodes, 256)
        self.assertEqual(result.network_evaluations, result.nodes)

    def test_soft_time_budget_completes_at_least_one_simulation(self):
        state = engine.State()
        result = engine.PUCT(engine.PUCTOptions(simulations=100000, time_limit_ms=1)).search(
            state, uniform_evaluator
        )
        self.assertGreaterEqual(result.simulations, 1)
        self.assertLess(result.simulations, 100000)
        self.check_result(state, result, result.simulations)

    def test_callback_receives_detached_state_and_cannot_change_search(self):
        state = engine.State()
        before = state_signature(state)
        seen = []

        def mutate_snapshot(snapshot):
            seen.append((snapshot.to_play, snapshot.board.to_string()))
            selected = 0 if snapshot.to_play == engine.Cell.Black else 1
            snapshot.pass_turn()
            snapshot.pass_turn()
            self.assertTrue(snapshot.result.finished())
            return [1.0 if action == selected else 0.0 for action in range(ACTION_SIZE)], 0.25

        result = engine.PUCT(engine.PUCTOptions(simulations=1)).search(state, mutate_snapshot)
        self.assertEqual(state_signature(state), before)
        self.assertEqual(result.network_evaluations, 2)
        self.assertEqual(move_to_action(result.best_move), 0)
        self.assertEqual(seen[0][0], engine.Cell.Black)
        self.assertEqual(seen[1][0], engine.Cell.White)
        self.assertTrue(seen[1][1].startswith("x"))
        self.assertEqual(result.best_value, -0.25)

    def test_invalid_options_are_rejected_and_options_are_copied(self):
        invalid = (
            {"simulations": 0}, {"c_puct": 0}, {"c_puct": -1},
            {"c_puct": float("nan")}, {"c_puct": float("inf")},
            {"dirichlet_alpha": 0}, {"dirichlet_alpha": -1},
            {"dirichlet_alpha": float("nan")}, {"dirichlet_alpha": float("inf")},
            {"dirichlet_epsilon": -0.1}, {"dirichlet_epsilon": 1.1},
            {"dirichlet_epsilon": float("nan")}, {"dirichlet_epsilon": float("inf")},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                engine.PUCT(engine.PUCTOptions(**fields))
        options = engine.PUCTOptions(simulations=3)
        searcher = engine.PUCT(options)
        options.simulations = 100
        detached = searcher.options
        detached.simulations = 200
        self.assertEqual(searcher.options.simulations, 3)
        self.assertEqual(searcher.search(engine.State(), uniform_evaluator).simulations, 3)

    def test_malformed_network_outputs_are_rejected(self):
        invalid = [
            ([1.0] * 81, 0.0), ([1.0] * 83, 0.0),
            ([0.0] * ACTION_SIZE, 0.0),
            ([0.0] * 40 + [1.0] + [0.0] * 41, 0.0),  # Only occupied neutral mass.
            ([1.0] * ACTION_SIZE, 1.001), ([1.0] * ACTION_SIZE, -1.001),
            ([1.0] * ACTION_SIZE, float("nan")),
            ([1.0] * ACTION_SIZE, float("inf")),
        ]
        for value in (-1.0, float("nan"), float("inf")):
            policy = [1.0] * ACTION_SIZE
            policy[0] = value
            invalid.append((policy, 0.0))
        searcher = engine.PUCT(engine.PUCTOptions(simulations=1))
        for output in invalid:
            with self.subTest(output=output), self.assertRaises((ValueError, TypeError, RuntimeError)):
                searcher.search(engine.State(), lambda state: output)
        recovered = searcher.search(engine.State(), uniform_evaluator)
        self.check_result(engine.State(), recovered, 1)

    def test_large_valid_policy_weights_normalize_without_overflow(self):
        policy = [1e308] * ACTION_SIZE
        result = engine.PUCT(engine.PUCTOptions(simulations=1)).search(
            engine.State(), lambda state: (policy, 0.0)
        )
        self.check_result(engine.State(), result, 1)
        for item in result.moves:
            self.assertAlmostEqual(item.prior, 1 / 81)

    def test_large_finite_exploration_coefficient_preserves_visit_distribution(self):
        state = capture_state()
        initial_board = state.board.to_string()
        policy = [0.0] * ACTION_SIZE
        policy[21] = 0.8
        policy[0] = 0.2

        def evaluate(snapshot):
            if snapshot.to_play == state.to_play and snapshot.board.to_string() == initial_board:
                return policy, 0.0
            return uniform_evaluator(snapshot)

        results = [engine.PUCT(engine.PUCTOptions(simulations=128, c_puct=coefficient)).search(
            state, evaluate
        ) for coefficient in (1e100, 1e308)]
        distributions = []
        for result in results:
            self.check_result(state, result, 128)
            self.assertEqual(move_to_action(result.best_move), 21)
            counts = {move_to_action(item.move): item.visits for item in result.moves}
            self.assertGreater(counts[0], 0)
            self.assertGreater(counts[21], counts[0])
            self.assertEqual(counts[21] + counts[0], 128)
            distributions.append(counts)
        # Both coefficients overwhelm bounded Q. A multiplication overflow
        # would turn multiple exploration scores into infinity and bias ties.
        self.assertEqual(distributions[0], distributions[1])

    def test_callback_exception_propagates_and_searcher_recovers(self):
        searcher = engine.PUCT(engine.PUCTOptions(simulations=2))

        def fail(state):
            raise LookupError("network callback failed")

        with self.assertRaisesRegex(LookupError, "network callback failed"):
            searcher.search(engine.State(), fail)
        recovered = searcher.search(engine.State(), uniform_evaluator)
        self.check_result(engine.State(), recovered, 2)

    def test_recursive_same_searcher_callback_fails_without_deadlock(self):
        searcher = engine.PUCT(engine.PUCTOptions(simulations=1))
        calls = []

        def evaluate(snapshot):
            with self.assertRaises(RuntimeError):
                searcher.search(snapshot, uniform_evaluator)
            calls.append(snapshot.to_play)
            return uniform_evaluator(snapshot)

        result = searcher.search(engine.State(), evaluate)
        self.check_result(engine.State(), result, 1)
        self.assertEqual(len(calls), 2)

    def test_shared_searcher_calls_are_serialized_safely(self):
        state = engine.State()
        before = state_signature(state)
        searcher = engine.PUCT(engine.PUCTOptions(simulations=16))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(searcher.search, state, uniform_evaluator) for _ in range(2)]
            results = [future.result(timeout=15) for future in futures]
        self.assertEqual(search_signature(results[0]), search_signature(results[1]))
        self.assertEqual(state_signature(state), before)
        for result in results:
            self.check_result(state, result, 16)

    def test_pytorch_adapter_counts_inference_and_preserves_training_mode(self):
        torch.manual_seed(173)
        model = PolicyValueNet(channels=8, residual_blocks=1)
        model.train()
        searcher = PUCT(model, PUCTOptions(simulations=8), device="cpu")
        state = engine.State()
        before = state_signature(state)
        calls = []

        def record_forward(module, inputs, output):
            calls.append(inputs[0].detach().clone())

        hook = model.register_forward_hook(record_forward)
        try:
            result = searcher.search(state)
        finally:
            hook.remove()
        self.check_result(state, result, 8)
        self.assertEqual(len(calls), result.network_evaluations)
        self.assertTrue(all(features.shape == (1, 10, 9, 9) for features in calls))
        self.assertTrue(all(features[:, 9].any().item() for features in calls))
        self.assertEqual(state_signature(state), before)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_adapter_masks_completed_house_and_rejects_unencoded_rules(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        searcher = PUCT(model, PUCTOptions(simulations=4))
        house = make_board(((1, 2), (2, 1), (2, 3), (3, 2)), neutral=engine.Position(4, 4))
        state = engine.State(house, engine.Cell.White)
        result = searcher.search(state)
        self.check_result(state, result, 4)
        self.assertNotIn(20, [move_to_action(item.move) for item in result.moves])
        self.assertNotIn(40, [move_to_action(item.move) for item in result.moves])
        with self.assertRaises(ValueError):
            searcher.search(engine.State(engine.GameRules(stones_per_player=1)))

    def test_visit_sampling_is_seeded_legal_and_temperature_zero_uses_best(self):
        state = engine.State()
        result = engine.PUCT(engine.PUCTOptions(simulations=128)).search(state, uniform_evaluator)
        self.assertEqual(sample_visits(result, temperature=0), result.best_move)
        generators = [torch.Generator().manual_seed(819) for _ in range(2)]
        sequences = [
            [move_to_action(sample_visits(result, temperature=1, generator=generator))
             for _ in range(20)] for generator in generators
        ]
        self.assertEqual(sequences[0], sequences[1])
        visited = {move_to_action(item.move) for item in result.moves if item.visits > 0}
        self.assertTrue(all(action in visited for action in sequences[0]))
        self.assertTrue(state.is_legal(sample_visits(result, temperature=1e-9)))
        for temperature in (-1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                sample_visits(result, temperature=temperature)
        terminal = engine.State()
        terminal.pass_turn()
        terminal.pass_turn()
        empty = engine.PUCT().search(terminal, uniform_evaluator)
        with self.assertRaises(ValueError):
            sample_visits(empty)

    def test_full_neural_selfplay_targets_train_and_checkpoint_round_trip(self):
        torch.manual_seed(372)
        model = PolicyValueNet(channels=8, residual_blocks=1)
        game = collect_puct_game(
            model, PUCTOptions(simulations=4, seed=713, dirichlet_epsilon=0.25),
            temperature=1.0, seed=713,
        )
        self.assertIn(game.winner, (engine.Cell.Black, engine.Cell.White))
        self.assertIn(game.reason, (engine.EndReason.Capture, engine.EndReason.Suicide,
                                   engine.EndReason.TwoPasses))
        self.assertGreaterEqual(len(game.samples), 2)
        self.assertLessEqual(len(game.samples), 2 * engine.CELL_COUNT + 2)
        for index, sample in enumerate(game.samples):
            with self.subTest(ply=index):
                self.assertEqual(sample.to_play,
                                 engine.Cell.Black if index % 2 == 0 else engine.Cell.White)
                self.assertEqual(sample.value, 1.0 if sample.to_play == game.winner else -1.0)
                self.assertEqual(sample.features.shape, (10, 9, 9))
                self.assertEqual(sample.policy.shape, (82,))
                self.assertAlmostEqual(sample.policy.sum().item(), 1, places=6)
                self.assertTrue((sample.policy[~sample.legal_mask] == 0).all())
                # Targets retain raw four-simulation visit fractions, independent
                # of temperature and the move sampled to continue the game.
                torch.testing.assert_close(sample.policy * 4, (sample.policy * 4).round(),
                                           rtol=0, atol=0)
        batch = make_batch(game.samples)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        metrics = train_step(model, torch.optim.Adam(model.parameters(), lr=0.001), batch)
        self.assertTrue(all(math.isfinite(metrics[key]) for key in ("loss", "policy_loss", "value_loss")))
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in model.named_parameters()))
        model.eval()
        with torch.inference_mode():
            expected = model(batch.features[:2])
        with tempfile.TemporaryDirectory(prefix="kingdom-puct-test-") as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path)
            with torch.inference_mode():
                actual = restored(batch.features[:2])
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
