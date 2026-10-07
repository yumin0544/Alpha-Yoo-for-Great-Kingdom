"""Bounded-proof selection and opt-in human-play integration contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.encoding import action_to_move, move_to_action
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.tactical_positions import curriculum_positions, load_position
from kingdom_ai.tactics import (DeepTacticalChoices, TacticalChoices, analyze_deep_tactics,
                               select_deep_tactical_action, select_deep_tactical_move)


def choices(legal=(0, 1, 2), wins=(), losses=()):
    legal, wins, losses = map(frozenset, (legal, wins, losses))
    return DeepTacticalChoices(legal, wins, losses, legal - wins - losses, "mock", 12,
                               4, 4, 0.5, False, ())


def cpu_result(best=0):
    return SimpleNamespace(best_move=action_to_move(best), moves=[
        SimpleNamespace(move=action_to_move(0), visits=100, value=0.95),
        SimpleNamespace(move=action_to_move(1), visits=20, value=0.2),
        SimpleNamespace(move=action_to_move(2), visits=10, value=0.5),
    ])


def gpu_result(best=0):
    visits = torch.zeros((1, 82), dtype=torch.int32)
    visits[0, :3] = torch.tensor([100, 20, 10])
    priors = torch.zeros((1, 82))
    priors[0, :3] = torch.tensor([0.8, 0.1, 0.1])
    # Deliberately [N], not [N,82]: GPU root values are not edge tiebreakers.
    return SimpleNamespace(actions=torch.tensor([best]), visits=visits, priors=priors,
                           values=torch.tensor([0.75]), best_values=torch.tensor([0.9]))


class DeepTacticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / "examples/play_ai.py"
        spec = importlib.util.spec_from_file_location("deep_tactics_play_ai", path)
        cls.play_ai = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.play_ai)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_unknown_allowed_proved_losing_excluded_cpu_and_gpu(self):
        proof = choices(losses=(0,))
        self.assertEqual(proof.preferred_actions, {1, 2})
        self.assertEqual(move_to_action(select_deep_tactical_move(cpu_result(), proof)), 1)
        self.assertEqual(select_deep_tactical_action(gpu_result(), proof), 1)

    def test_proved_win_overrides_more_visited_unknown(self):
        proof = choices(wins=(2,), losses=(0,))
        self.assertEqual(move_to_action(select_deep_tactical_move(cpu_result(), proof)), 2)
        self.assertEqual(select_deep_tactical_action(gpu_result(), proof), 2)

    def test_all_proved_losing_preserves_search_fallback(self):
        proof = choices(losses=(0, 1, 2))
        self.assertEqual(proof.preferred_actions, {0, 1, 2})
        self.assertEqual(move_to_action(select_deep_tactical_move(cpu_result(), proof)), 0)
        self.assertEqual(select_deep_tactical_action(gpu_result(), proof), 0)

    def test_all_unknown_preserves_original_choice_without_safety_claim(self):
        proof = choices()
        self.assertEqual(move_to_action(select_deep_tactical_move(cpu_result(best=2), proof)), 2)
        self.assertEqual(select_deep_tactical_action(gpu_result(best=2), proof), 2)

    def test_existing_one_reply_proofs_merge_with_deep_unknowns(self):
        proof = choices(losses=(0,))
        immediate = TacticalChoices(proof.legal_actions, frozenset(), frozenset({2}))
        self.assertEqual(move_to_action(select_deep_tactical_move(
            cpu_result(), proof, immediate_choices=immediate)), 2)
        self.assertEqual(select_deep_tactical_action(
            gpu_result(), proof, immediate_choices=immediate), 2)

    def test_gpu_tiebreak_uses_priors_not_root_values(self):
        result = gpu_result()
        result.visits[0, 1:3] = 10
        result.priors[0, 1:3] = torch.tensor([0.1, 0.5])
        self.assertEqual(select_deep_tactical_action(result, choices(losses=(0,))), 2)

    def test_wrapper_passes_limits_and_keeps_unknown_partition_state_unchanged(self):
        state = engine.State()
        legal = state.legal_moves()
        before = (state.board.cells, state.ownership, state.to_play, state.consecutive_passes)
        raw = SimpleNamespace(winning_moves=[], losing_moves=[], unknown_moves=legal,
                              outcome="unknown", nodes=1, proof_depth=0, completed_depth=0,
                              elapsed_ms=0.2, budget_exhausted=True, principal_variation=[])
        with patch.object(engine, "TacticalSolverOptions", side_effect=lambda **kw: SimpleNamespace(**kw),
                          create=True), patch.object(engine, "solve_tactics", return_value=raw,
                                                     create=True) as solve:
            proof = analyze_deep_tactics(state, max_depth=9, max_nodes=1, time_limit_ms=0)
        self.assertEqual(proof.unknown_actions, {move_to_action(m) for m in legal})
        self.assertTrue(proof.budget_exhausted)
        options = solve.call_args.args[1]
        self.assertEqual((options.max_depth, options.max_nodes, options.time_limit_ms), (9, 1, 0))
        self.assertIsNot(solve.call_args.args[0], state)
        self.assertEqual((state.board.cells, state.ownership, state.to_play, state.consecutive_passes),
                         before)

    def test_wrapper_rejects_incomplete_partition(self):
        raw = SimpleNamespace(winning_moves=[], losing_moves=[], unknown_moves=[])
        with patch.object(engine, "TacticalSolverOptions", side_effect=lambda **kw: SimpleNamespace(**kw),
                          create=True), patch.object(engine, "solve_tactics", return_value=raw, create=True):
            with self.assertRaises(RuntimeError):
                analyze_deep_tactics(engine.State())

    def test_invalid_limits_terminal_and_gpu_shapes(self):
        for kwargs in ({"max_depth": 0}, {"max_depth": True}, {"max_nodes": 0},
                       {"time_limit_ms": -1}, {"max_nodes": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_deep_tactics(engine.State(), **kwargs)
        with self.assertRaises(TypeError):
            analyze_deep_tactics(None)
        with self.assertRaises(ValueError):
            select_deep_tactical_move(SimpleNamespace(best_move=None, moves=[]), choices())
        wrong = gpu_result()
        wrong.visits = torch.zeros((1, 81))
        with self.assertRaises(ValueError):
            select_deep_tactical_action(wrong, choices(losses=(0,)))

    def test_default_cpu_opponent_never_calls_deep_solver(self):
        model = PolicyValueNet(channels=4, residual_blocks=0)
        opponent = self.play_ai.Opponent(model, "cpu", 2, 42, tactical_checks=False)
        with patch.object(self.play_ai, "analyze_deep_tactics", side_effect=AssertionError("disabled")):
            self.assertTrue(engine.State().is_legal(opponent.choose(engine.State())))
        self.assertIsNone(opponent.last_deep_tactics)

    def test_cpu_opt_in_proof_can_override_neural_search_without_model_change(self):
        model = PolicyValueNet(channels=4, residual_blocks=0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with patch.object(engine, "solve_tactics", create=True):
            opponent = self.play_ai.Opponent(model, "cpu", 2, 42, tactical_checks=False,
                                            tactical_depth=9, tactical_node_budget=123,
                                            tactical_time_ms=0)
        proof = choices(legal=(0, 1), wins=(1,))
        with patch.object(self.play_ai, "analyze_deep_tactics", return_value=proof) as analyze:
            self.assertEqual(move_to_action(opponent.choose(engine.State())), 1)
        self.assertEqual(analyze.call_args.kwargs, {"max_depth": 9, "max_nodes": 123,
                                                  "time_limit_ms": 0})
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_mock_cuda_path_applies_cpu_proof_filter_to_actual_gpu_fields(self):
        model = PolicyValueNet(channels=4, residual_blocks=0)
        searcher = SimpleNamespace(search=lambda state: gpu_result())
        with patch.object(engine, "solve_tactics", create=True), \
                patch.object(self.play_ai, "GpuPUCT", return_value=searcher):
            opponent = self.play_ai.Opponent(model, "cuda", 2, 42, tactical_checks=False,
                                            tactical_depth=9)
        with patch.object(self.play_ai, "analyze_deep_tactics", return_value=choices(losses=(0,))), \
                patch.object(self.play_ai.GpuStateBatch, "from_engine", return_value=object()):
            self.assertEqual(move_to_action(opponent.choose(engine.State())), 1)

    @unittest.skipUnless(hasattr(engine, "solve_tactics"), "Rebuilt solver binding is unavailable")
    def test_actual_engine_solver_immediate_capture_and_node_cut_unknown(self):
        case = next(c for c in curriculum_positions() if c["id"] == "capture_interior")
        state = load_position(case)
        proof = analyze_deep_tactics(state, max_depth=1, max_nodes=200, time_limit_ms=0)
        self.assertIn(11, proof.winning_actions)  # (2,3)
        limited = analyze_deep_tactics(engine.State(), max_depth=12, max_nodes=1, time_limit_ms=0)
        self.assertTrue(limited.unknown_actions)
        self.assertFalse(limited.winning_actions)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(engine, "solve_tactics"),
                         "CUDA and the rebuilt solver binding are required")
    def test_actual_cuda_opt_in_filters_proved_reply_capture(self):
        state = engine.State()
        for row, col in ((4, 8), (5, 8), (2, 8), (6, 8), (2, 4), (4, 4),
                         (6, 2), (4, 7), (3, 7), (5, 9), (4, 6)):
            self.assertTrue(state.place(row - 1, col - 1).accepted())
        before = (state.board.cells, state.ownership, state.to_play)
        model = PolicyValueNet(channels=8, residual_blocks=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_head[-1].bias[35] = 100.0
        opponent = self.play_ai.Opponent(
            model, "cuda", 2, 42, tactical_checks=False,
            tactical_depth=2, tactical_node_budget=10000, tactical_time_ms=0)
        self.assertEqual(move_to_action(opponent.choose(state)), 42)
        self.assertIn(35, opponent.last_deep_tactics.losing_actions)
        self.assertIn(42, opponent.last_deep_tactics.unknown_actions)
        self.assertEqual((state.board.cells, state.ownership, state.to_play), before)


if __name__ == "__main__":
    unittest.main()
