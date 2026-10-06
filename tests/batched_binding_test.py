"""C++ leaf batches, schema parity, search budgets and retained subtrees.

Run after rebuilding the extension: python tests/batched_binding_test.py
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import my_board_engine as engine

from kingdom_ai.encoding import ACTION_SIZE, encode_state, move_to_action, terminal_value
from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig


def uniform_batch(features, legal_masks):
    return np.ones((len(features), ACTION_SIZE), dtype=np.float64), np.zeros(len(features))


def uniform_scalar(state):
    return [1.0] * ACTION_SIZE, 0.0


def make_board(black=(), white=(), neutral=None):
    board = engine.Board(neutral=neutral)
    for points, player in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), player):
                raise AssertionError("Invalid batched PUCT fixture")
    return board


def capture_state():
    return engine.State(make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),)), engine.Cell.Black)


def suicide_state():
    return engine.State(
        make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
        engine.Cell.Black,
    )


def state_signature(state):
    return (
        tuple(state.board.cells), tuple(state.ownership), state.to_play,
        state.remaining_stones(engine.Cell.Black), state.remaining_stones(engine.Cell.White),
        state.consecutive_passes, state.result,
    )


def search_signature(result):
    # Timing deliberately omitted; every algorithmic legacy field is compared.
    return (
        None if result.best_move is None else move_to_action(result.best_move),
        result.simulations, result.nodes, result.network_evaluations,
        result.root_value, result.best_value,
        tuple((move_to_action(item.move), item.prior, item.visits, item.value)
              for item in result.moves),
    )


class BatchedBindingTest(unittest.TestCase):
    def searcher(self, simulations=32, *, batch_size=8, reuse_tree=True):
        return engine.BatchedPUCT(
            engine.PUCTOptions(simulations=simulations, seed=237),
            leaf_batch_size=batch_size, reuse_tree=reuse_tree,
        )

    def check_result(self, state, result, simulations):
        self.assertEqual(result.simulations, simulations)
        self.assertEqual(sum(item.visits for item in result.moves), simulations)
        self.assertEqual({move_to_action(item.move) for item in result.moves},
                         {move_to_action(move) for move in state.legal_moves()})
        self.assertTrue(state.is_legal(result.best_move))
        self.assertTrue(math.isfinite(result.root_value))
        self.assertLessEqual(abs(result.root_value), 1)
        self.assertAlmostEqual(sum(item.prior for item in result.moves), 1, places=12)
        for item in result.moves:
            self.assertGreaterEqual(item.visits, 0)
            self.assertGreaterEqual(item.prior, 0)
            self.assertTrue(math.isfinite(item.value))
            self.assertLessEqual(abs(item.value), 1)

    def test_cpp_root_and_leaf_encoding_exactly_matches_python_schema(self):
        progressed = engine.State()
        self.assertTrue(progressed.place(0, 0).accepted())
        self.assertTrue(progressed.place(8, 8).accepted())
        self.assertTrue(progressed.pass_turn().accepted())
        house = engine.State(
            make_board(((1, 2), (2, 1), (2, 3), (3, 2)),
                       neutral=engine.Position(4, 4)), engine.Cell.White,
        )
        states = [engine.State(), engine.State(neutral=None),
                  engine.State(neutral=engine.Position(0, 8)), progressed, house]
        for state in states:
            with self.subTest(state=state_signature(state)):
                before = state_signature(state)
                # Uniform priors with an eight-simulation budget visit distinct
                # root actions. Build the independent C++ rule/Python encoding
                # oracle for the root and every possible nonterminal child.
                expected = {}
                candidates = [state.copy()]
                for move in state.legal_moves():
                    child = state.copy()
                    self.assertTrue(child.play(move).accepted())
                    if not child.result.finished():
                        candidates.append(child)
                for candidate in candidates:
                    encoded = encode_state(candidate)
                    expected[encoded.features.numpy().tobytes()] = encoded
                batches = []

                def check_encoding(features, masks):
                    self.assertIsInstance(features, np.ndarray)
                    self.assertIsInstance(masks, np.ndarray)
                    self.assertEqual(features.dtype, np.dtype(np.float32))
                    self.assertEqual(masks.dtype, np.dtype(np.bool_))
                    self.assertEqual(features.shape[1:], (10, 9, 9))
                    self.assertEqual(masks.shape, (len(features), ACTION_SIZE))
                    self.assertTrue(features.flags.c_contiguous)
                    self.assertTrue(masks.flags.c_contiguous)
                    batches.append(len(features))
                    for planes, mask in zip(features, masks):
                        oracle = expected[planes.tobytes()]
                        torch.testing.assert_close(torch.from_numpy(planes), oracle.features,
                                                   rtol=0, atol=0)
                        torch.testing.assert_close(torch.from_numpy(mask), oracle.legal_mask,
                                                   rtol=0, atol=0)
                    return uniform_batch(features, masks)

                result = self.searcher(8).search(state, check_encoding)
                self.check_result(state, result, 8)
                self.assertEqual(state_signature(state), before)
                self.assertEqual(batches[0], 1)
                self.assertGreater(max(batches), 1)
                self.assertEqual(sum(batches), result.network_evaluations)

    def test_standalone_encoder_empty_progressed_passed_and_terminal_batches(self):
        features, masks = engine.encode_puct_batch([])
        self.assertEqual(features.shape, (0, 10, 9, 9))
        self.assertEqual(masks.shape, (0, ACTION_SIZE))
        self.assertEqual(features.dtype, np.dtype(np.float32))
        self.assertEqual(masks.dtype, np.dtype(np.bool_))
        states = []
        for neutral in (None, engine.Position(0, 8), engine.Position(4, 4)):
            rng = random.Random(491)
            state = engine.State(neutral=neutral)
            states.append(state.copy())
            for _ in range(30):
                if state.result.finished():
                    break
                self.assertTrue(state.play(rng.choice(state.legal_moves())).accepted())
                states.append(state.copy())
            passed = engine.State(neutral=neutral)
            passed.pass_turn()
            states.append(passed.copy())
            passed.pass_turn()
            states.append(passed.copy())
        captured = capture_state()
        captured.place(2, 3)
        states.append(captured)
        suicide = suicide_state()
        suicide.place(1, 1)
        states.append(suicide)
        before = [state_signature(state) for state in states]
        features, masks = engine.encode_puct_batch(states)
        expected = [encode_state(state) for state in states]
        torch.testing.assert_close(torch.from_numpy(features),
                                   torch.stack([item.features for item in expected]), rtol=0, atol=0)
        torch.testing.assert_close(torch.from_numpy(masks),
                                   torch.stack([item.legal_mask for item in expected]), rtol=0, atol=0)
        self.assertEqual([state_signature(state) for state in states], before)
        self.assertTrue(features.flags.owndata)
        self.assertTrue(masks.flags.owndata)
        saved = features.copy(), masks.copy()
        engine.encode_puct_batch([engine.State()] * 64)
        np.testing.assert_array_equal(features, saved[0])
        np.testing.assert_array_equal(masks, saved[1])

    def test_single_leaf_without_reuse_matches_legacy_fields_exactly(self):
        state = engine.State()
        self.assertTrue(state.place(0, 0).accepted())
        options = engine.PUCTOptions(simulations=96, seed=237)
        legacy = engine.PUCT(options).search(state, uniform_scalar)
        actual = engine.BatchedPUCT(options, leaf_batch_size=1, reuse_tree=False).search(
            state, uniform_batch,
        )
        self.assertEqual(search_signature(actual), search_signature(legacy))

    def test_nonuniform_values_keep_scalar_selection_and_backup_perspectives(self):
        def policy_and_value(actor_is_black):
            policy = np.zeros(ACTION_SIZE, dtype=np.float64)
            policy[0 if actor_is_black else 1] = 1
            return policy, 0.25 if actor_is_black else 0.75

        def scalar(state):
            policy, value = policy_and_value(state.to_play == engine.Cell.Black)
            return policy.tolist(), value

        def batched(features, masks):
            outputs = [policy_and_value(bool(planes[5, 0, 0])) for planes in features]
            return np.stack([item[0] for item in outputs]), np.array([item[1] for item in outputs])

        options = engine.PUCTOptions(simulations=1)
        legacy = engine.PUCT(options).search(engine.State(), scalar)
        actual = engine.BatchedPUCT(options, leaf_batch_size=1, reuse_tree=False).search(
            engine.State(), batched,
        )
        self.assertEqual(search_signature(actual), search_signature(legacy))
        self.assertEqual(actual.best_value, -0.75)

    def test_leaf_batches_respect_every_budget_and_callback_count(self):
        for simulations in (1, 3, 8, 17, 96):
            for batch_size in (1, 4, 8, 32):
                with self.subTest(simulations=simulations, batch_size=batch_size):
                    batches = []

                    def record(features, masks):
                        batches.append(len(features))
                        self.assertLessEqual(len(features), batch_size)
                        return uniform_batch(features, masks)

                    searcher = self.searcher(simulations, batch_size=batch_size)
                    result = searcher.search(engine.State(), record)
                    self.check_result(engine.State(), result, simulations)
                    self.assertEqual(sum(batches), result.network_evaluations)
                    self.assertEqual(searcher.stats["inference_batches"], len(batches))
                    self.assertEqual(searcher.stats["max_observed_batch_size"], max(batches))
                    self.assertEqual(searcher.stats["network_evaluations"], sum(batches))

    def test_advance_retains_evaluated_subtree_and_counts_only_new_root_visits(self):
        state = engine.State()
        searcher = self.searcher(160, batch_size=1)
        first = searcher.search(state, uniform_batch)
        self.check_result(state, first, 160)
        move = first.best_move
        self.assertTrue(searcher.advance(move))
        self.assertTrue(state.play(move).accepted())
        second = searcher.search(state, uniform_batch)
        self.check_result(state, second, 160)
        stats = searcher.stats
        self.assertGreater(stats["reused_nodes"], 0)
        self.assertGreater(stats["inherited_visits"], 0)
        # A retained, already encoded/evaluated root must not incur the extra
        # root network evaluation of a fresh, nonterminal 160-simulation tree.
        fresh = self.searcher(160, batch_size=1, reuse_tree=False).search(state, uniform_batch)
        self.assertLess(second.network_evaluations, fresh.network_evaluations)
        self.assertEqual(sum(item.visits for item in second.moves), 160)

    def test_repeated_search_reuses_root_but_each_call_spends_new_budget(self):
        searcher = self.searcher(32, batch_size=4)
        state = engine.State()
        first = searcher.search(state, uniform_batch)
        second = searcher.search(state, uniform_batch)
        self.check_result(state, first, 32)
        self.check_result(state, second, 32)
        self.assertGreater(searcher.stats["inherited_visits"], 0)
        self.assertGreater(searcher.stats["reused_nodes"], 0)

    def test_clear_and_mismatched_state_do_not_reuse_tree(self):
        searcher = self.searcher(8)
        initial = engine.State()
        searcher.search(initial, uniform_batch)
        searcher.clear()
        searcher.search(initial, uniform_batch)
        self.assertEqual(searcher.stats["reused_nodes"], 0)
        self.assertEqual(searcher.stats["inherited_visits"], 0)
        different = engine.State(neutral=None)
        result = searcher.search(different, uniform_batch)
        self.check_result(different, result, 8)
        self.assertEqual(searcher.stats["reused_nodes"], 0)
        self.assertEqual(searcher.stats["inherited_visits"], 0)

    def test_illegal_advance_does_not_drop_retained_tree(self):
        searcher = self.searcher(32)
        state = engine.State()
        searcher.search(state, uniform_batch)
        self.assertFalse(searcher.advance(engine.Move.place(4, 4)))
        self.assertFalse(searcher.advance(engine.Move.place(-1, 0)))
        result = searcher.search(state, uniform_batch)
        self.check_result(state, result, 32)
        self.assertGreater(searcher.stats["reused_nodes"], 0)
        self.assertGreater(searcher.stats["inherited_visits"], 0)

    def test_capture_suicide_and_two_pass_terminal_values_bypass_network(self):
        captured = capture_state()
        self.assertTrue(captured.place(2, 3).accepted())
        suicide = suicide_state()
        self.assertTrue(suicide.place(1, 1).accepted())
        passed = engine.State()
        passed.pass_turn()
        passed.pass_turn()
        for state in (captured, suicide, passed):
            with self.subTest(reason=state.result.reason):
                before = state_signature(state)

                def forbidden(features, masks):
                    raise AssertionError("Terminal states must not reach inference")

                result = self.searcher().search(state, forbidden)
                self.assertIsNone(result.best_move)
                self.assertEqual(result.moves, [])
                self.assertEqual(result.simulations, 0)
                self.assertEqual(result.network_evaluations, 0)
                self.assertEqual(result.root_value, terminal_value(state))
                self.assertEqual(state_signature(state), before)

    def test_terminal_leaves_use_exact_outcome_with_one_root_batch(self):
        for state, action, value, simulations in ((capture_state(), 21, 1.0, 4),
                                                  (suicide_state(), 10, -1.0, 1)):
            with self.subTest(action=action):
                batches = []

                def one_hot(features, masks):
                    batches.append(len(features))
                    policies = np.zeros((len(features), ACTION_SIZE), dtype=np.float64)
                    policies[:, action] = 1
                    return policies, np.full(len(features), 0.99)

                # With larger budgets PUCT may explore zero-prior alternatives
                # after a losing move; one simulation isolates the exact leaf.
                result = self.searcher(simulations, batch_size=1).search(state, one_hot)
                self.check_result(state, result, simulations)
                self.assertEqual(move_to_action(result.best_move), action)
                self.assertEqual(result.best_value, value)
                self.assertEqual(result.root_value, value)
                self.assertEqual(batches, [1])

    def test_callback_error_resets_pending_reservations_for_later_search(self):
        searcher = self.searcher(32)
        calls = 0

        def fail_on_leaf(features, masks):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise LookupError("batched callback failed")
            return uniform_batch(features, masks)

        state = engine.State()
        before = state_signature(state)
        with self.assertRaisesRegex(LookupError, "batched callback failed"):
            searcher.search(state, fail_on_leaf)
        self.assertEqual(state_signature(state), before)
        recovered = searcher.search(state, uniform_batch)
        self.check_result(state, recovered, 32)
        fresh = self.searcher(32).search(state, uniform_batch)
        self.assertEqual(search_signature(recovered), search_signature(fresh))

    def test_bad_shapes_nan_negative_and_out_of_range_outputs_are_rejected(self):
        def output_factory(kind):
            def evaluate(features, masks):
                policies, values = uniform_batch(features, masks)
                if kind == "policy_columns":
                    policies = policies[:, :-1]
                elif kind == "policy_rows":
                    policies = np.vstack((policies, policies[:1]))
                elif kind == "value_rows":
                    values = np.append(values, 0)
                elif kind == "value_rank":
                    values = values[:, None]
                elif kind == "policy_nan":
                    policies[:, 0] = np.nan
                elif kind == "policy_negative":
                    policies[:, 0] = -1
                elif kind == "policy_zero":
                    policies[:] = 0
                elif kind == "value_nan":
                    values[:] = np.nan
                elif kind == "value_infinity":
                    values[:] = np.inf
                elif kind == "value_range":
                    values[:] = 1.1
                return policies, values
            return evaluate

        searcher = self.searcher(8)
        for kind in ("policy_columns", "policy_rows", "value_rows", "value_rank",
                     "policy_nan", "policy_negative", "policy_zero", "value_nan",
                     "value_infinity", "value_range"):
            with self.subTest(kind=kind):
                searcher.clear()
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    searcher.search(engine.State(), output_factory(kind))
                self.check_result(engine.State(), searcher.search(engine.State(), uniform_batch), 8)

    def test_float32_outputs_are_accepted_and_inputs_are_detached(self):
        state = engine.State()
        before = state_signature(state)

        def mutate_inputs(features, masks):
            count = len(features)
            features[:] = 99
            masks[:] = False
            return np.ones((count, ACTION_SIZE), dtype=np.float32), np.zeros(count, dtype=np.float32)

        result = self.searcher(16).search(state, mutate_inputs)
        self.check_result(state, result, 16)
        self.assertEqual(state_signature(state), before)

    def test_retained_callback_arrays_outlive_search_and_later_allocations(self):
        retained = []

        def retain(features, masks):
            retained.append((features, masks, features.copy(), masks.copy()))
            return uniform_batch(features, masks)

        searcher = self.searcher(17)
        searcher.search(engine.State(), retain)
        searcher.clear()
        searcher.search(engine.State(neutral=None), uniform_batch)
        for features, masks, expected_features, expected_masks in retained:
            self.assertTrue(features.flags.owndata)
            self.assertTrue(masks.flags.owndata)
            np.testing.assert_array_equal(features, expected_features)
            np.testing.assert_array_equal(masks, expected_masks)

    def test_invalid_options_rejected_and_options_are_copied(self):
        for arguments in ({"leaf_batch_size": 0}, {"leaf_batch_size": -1}):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                engine.BatchedPUCT(**arguments)
        for fields in ({"simulations": 0}, {"c_puct": 0}, {"c_puct": float("nan")},
                       {"dirichlet_epsilon": 0.25}):
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                engine.BatchedPUCT(engine.PUCTOptions(**fields))
        options = engine.PUCTOptions(simulations=3)
        searcher = engine.BatchedPUCT(options, leaf_batch_size=4, reuse_tree=False)
        options.simulations = 100
        detached = searcher.options
        detached.simulations = 200
        self.assertEqual(searcher.options.simulations, 3)
        self.assertEqual(searcher.leaf_batch_size, 4)
        self.assertFalse(searcher.reuse_tree)
        self.assertEqual(searcher.search(engine.State(), uniform_batch).simulations, 3)

    def test_recursive_access_rejected_and_object_remains_usable(self):
        for operation in ("search", "advance", "clear", "stats"):
            with self.subTest(operation=operation):
                searcher = self.searcher(8)

                def recurse(features, masks):
                    if operation == "search":
                        searcher.search(engine.State(), uniform_batch)
                    elif operation == "advance":
                        searcher.advance(engine.Move.place(0, 0))
                    elif operation == "clear":
                        searcher.clear()
                    else:
                        _ = searcher.stats
                    return uniform_batch(features, masks)

                with self.assertRaisesRegex(RuntimeError, "Recursive"):
                    searcher.search(engine.State(), recurse)
                self.check_result(engine.State(), searcher.search(engine.State(), uniform_batch), 8)

    def test_shared_searcher_serializes_threaded_calls_without_gil_deadlock(self):
        searcher = self.searcher(17, batch_size=4, reuse_tree=False)
        state = engine.State()
        before = state_signature(state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(searcher.search, state, uniform_batch) for _ in range(2)]
            results = [future.result(timeout=15) for future in futures]
        self.assertEqual(search_signature(results[0]), search_signature(results[1]))
        self.assertEqual(state_signature(state), before)

    def test_unsupported_encoding_rules_fail_before_callback(self):
        for rules in (engine.GameRules(stones_per_player=1),
                      engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden),
                      engine.GameRules(allow_own_territory_moves=True),
                      engine.GameRules(allow_single_edge_territory=False)):
            with self.subTest(rules=rules):

                def forbidden(features, masks):
                    raise AssertionError("Unsupported rules reached encoding callback")

                with self.assertRaises(ValueError):
                    self.searcher().search(engine.State(rules), forbidden)


class BenchmarkSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "examples" / "batched_evaluation_benchmark.py"
        spec = importlib.util.spec_from_file_location("batched_evaluation_benchmark_tests", path)
        cls.benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.benchmark)

    def test_training_source_keeps_distinct_learner_and_champion_without_training_or_save(self):
        # The strict public loader has its own checkpoint tests. This test
        # isolates the benchmark's wiring without initializing CUDA or Adam.
        config = TrainingConfig()
        learner = PolicyValueNet(channels=4, residual_blocks=0)
        champion = PolicyValueNet(channels=4, residual_blocks=0)
        restored = SimpleNamespace(model=learner, champion=champion, config=config,
                                   iteration=244, self_play_games=976,
                                   training_steps=1952, champion_version=106)
        expected_digests = tuple(self.benchmark.model_digest(model) for model in (learner, champion))
        self.assertNotEqual(*expected_digests)
        checkpoint = Path("fixture-latest.pt")
        args = SimpleNamespace(training_checkpoint=checkpoint, model=None, reference_model=None)
        with patch.object(Trainer, "load_checkpoint", return_value=restored) as load, \
                patch.object(self.benchmark.torch, "load", return_value={"checkpoint_version": 4}) as metadata_load, \
                patch.object(Trainer, "run") as run, \
                patch.object(Trainer, "run_iteration") as run_iteration, \
                patch.object(Trainer, "save_checkpoint") as save:
            candidate, reference, metadata = self.benchmark.load_source_models(args, torch.device("cpu"))
            load.assert_called_once_with(checkpoint, device=torch.device("cpu"))
            metadata_load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            run.assert_not_called()
            run_iteration.assert_not_called()
            save.assert_not_called()
        self.assertIs(candidate, learner)
        self.assertIs(reference, champion)
        self.assertEqual(tuple(self.benchmark.model_digest(model) for model in (candidate, reference)),
                         expected_digests)
        self.assertFalse(candidate.training)
        self.assertFalse(reference.training)
        self.assertEqual(metadata["source_kind"], "training_checkpoint")
        self.assertEqual(metadata["source_model_roles"], ["learner", "champion"])
        self.assertEqual(metadata["source_checkpoint_version"], 4)
        self.assertEqual(metadata["source_progress"], {
            "iteration": 244, "self_play_games": 976, "training_steps": 1952, "champion_version": 106,
        })
        self.assertEqual(metadata["source_training_config"]["self_play_backend"], "cpu")

    def test_cli_rejects_missing_or_conflicting_sources_before_loading(self):
        target = "unused-cli-test-output.jsonl"
        bad_sources = ([], ["--model", "a.pt", "--training-checkpoint", "b.pt"],
                       ["--training-checkpoint", "b.pt", "--reference-model", "a.pt"])
        for sources in bad_sources:
            with self.subTest(sources=sources), patch("sys.argv", ["benchmark", "--output", target, *sources]), \
                    patch("sys.stderr"), patch.object(self.benchmark, "load_source_models") as load, \
                    patch.object(Path, "open") as open_file, patch.object(Path, "mkdir") as mkdir:
                with self.assertRaises(SystemExit) as error:
                    self.benchmark.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
                open_file.assert_not_called()
                mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
