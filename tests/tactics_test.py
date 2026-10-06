"""Immediate tactical safety compared with exhaustive verified engine replies."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import my_board_engine as engine

from kingdom_ai.encoding import action_to_move, move_to_action
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.tactics import analyze_tactics, select_tactical_move


HUMAN_GAME = (
    (4, 8), (5, 8), (2, 8), (6, 8), (2, 4), (4, 4),
    (6, 2), (4, 7), (3, 7), (5, 9), (4, 6),
)


def transform(row, col, rotation, reflection):
    if reflection:
        col = 8 - col
    for _ in range(rotation):
        row, col = col, 8 - row
    return row, col


def human_game(rotation=0, reflection=False):
    state = engine.State()
    for row, col in HUMAN_GAME:
        row, col = transform(row - 1, col - 1, rotation, reflection)
        if not state.place(row, col).accepted():
            raise AssertionError("Human game fixture contains an illegal move")
    return state


def make_board(black=(), white=()):
    board = engine.Board()
    for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), color):
                raise AssertionError("Invalid tactical fixture")
    return board


def brute_force_choices(state):
    """Independent oracle: play every root move and every legal opposing reply."""
    actor = state.to_play
    legal, winning, safe = set(), set(), set()
    for move in state.legal_moves():
        action = move_to_action(move)
        legal.add(action)
        child = state.copy()
        if not child.play(move).accepted():
            raise AssertionError("Engine oracle rejected a legal root move")
        if child.result.finished():
            if child.result.winner == actor:
                winning.add(action)
                safe.add(action)
            continue
        loses = False
        for reply_move in child.legal_moves():
            reply = child.copy()
            if not reply.play(reply_move).accepted():
                raise AssertionError("Engine oracle rejected a legal reply")
            if reply.result.finished() and reply.result.winner != actor:
                loses = True
                break
        if not loses:
            safe.add(action)
    return legal, winning, safe


def record(state):
    return (tuple(state.board.cells), tuple(state.ownership), state.to_play,
            state.consecutive_passes, state.remaining_stones(engine.Cell.Black),
            state.remaining_stones(engine.Cell.White), state.result.reason,
            state.result.winner, state.result.captured_stones)


class TacticalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / "examples" / "play_ai.py"
        spec = importlib.util.spec_from_file_location("tactics_play_ai", path)
        cls.play_ai = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.play_ai)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_oracle(self, state):
        before = record(state)
        choices = analyze_tactics(state)
        expected = brute_force_choices(state)
        self.assertEqual(choices.legal_actions, expected[0])
        self.assertEqual(choices.winning_actions, expected[1])
        self.assertEqual(choices.safe_actions, expected[2])
        self.assertEqual(record(state), before)
        return choices

    def test_human_capture_is_avoided_in_all_eight_symmetries(self):
        for reflection in (False, True):
            for rotation in range(4):
                with self.subTest(rotation=rotation, reflection=reflection):
                    state = human_game(rotation, reflection)
                    choices = self.assert_oracle(state)
                    row, col = transform(4, 6, rotation, reflection)
                    defense = row * 9 + col
                    bad_row, bad_col = transform(3, 8, rotation, reflection)
                    bad_action = bad_row * 9 + bad_col
                    self.assertEqual(choices.winning_actions, set())
                    self.assertEqual(choices.safe_actions, {defense})
                    bad = state.copy()
                    self.assertTrue(bad.play(action_to_move(bad_action)).accepted())
                    self.assertTrue(bad.play(action_to_move(defense)).accepted())
                    self.assertEqual(bad.result.reason, engine.EndReason.Capture)
                    self.assertEqual(bad.result.winner, engine.Cell.Black)

    def test_immediate_capture_wins_are_preferred(self):
        state = engine.State(make_board(((0, 1),), ((0, 0),)), engine.Cell.Black)
        choices = self.assert_oracle(state)
        self.assertEqual(choices.winning_actions, {9})
        self.assertEqual(choices.preferred_actions, {9})

    def test_capture_before_suicide_remains_a_win(self):
        cells = []
        for row in range(9):
            for col in range(9):
                cells.append(engine.Cell.Black if col < 4 or (col == 4 and row < 4)
                             else engine.Cell.White if col > 4 or (col == 4 and row > 4)
                             else engine.Cell.Empty)
        for actor in (engine.Cell.Black, engine.Cell.White):
            with self.subTest(actor=actor):
                state = engine.State(engine.Board(cells), actor)
                choices = self.assert_oracle(state)
                self.assertEqual(choices.winning_actions, {40})
                self.assertEqual(choices.safe_actions, {40})

    def test_immediate_suicide_is_excluded(self):
        state = engine.State(make_board(((0, 1),),
                                       ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
                             engine.Cell.Black)
        choices = self.assert_oracle(state)
        self.assertIn(10, choices.legal_actions)
        self.assertNotIn(10, choices.safe_actions)
        suicide = state.copy()
        self.assertTrue(suicide.place(1, 1).accepted())
        self.assertEqual(suicide.result.reason, engine.EndReason.Suicide)

    def test_two_pass_wins_losses_and_opponent_pass_reply(self):
        initial = engine.State()
        choices = self.assert_oracle(initial)
        self.assertNotIn(81, choices.safe_actions)
        white = initial.copy()
        white.pass_turn()
        white_choices = self.assert_oracle(white)
        self.assertIn(81, white_choices.winning_actions)
        black = engine.State(engine.Board(), engine.Cell.White)
        black.pass_turn()
        black_choices = self.assert_oracle(black)
        self.assertIn(81, black_choices.legal_actions)
        self.assertNotIn(81, black_choices.safe_actions)

    def test_unavoidable_loss_preserves_original_search_choice(self):
        state = engine.State(make_board(((0, 0), (0, 8)),
                                       ((0, 1), (1, 1), (0, 7), (1, 7))),
                             engine.Cell.Black)
        choices = self.assert_oracle(state)
        self.assertFalse(choices.winning_actions)
        self.assertFalse(choices.safe_actions)
        self.assertEqual(choices.preferred_actions, choices.legal_actions)
        original = action_to_move(9)
        result = SimpleNamespace(best_move=original, moves=[])
        self.assertEqual(move_to_action(select_tactical_move(result, choices)), 9)

    def test_safe_move_selection_overrides_a_high_visit_blunder(self):
        state = human_game()
        choices = analyze_tactics(state)
        result = SimpleNamespace(best_move=action_to_move(35), moves=[
            SimpleNamespace(move=action_to_move(35), visits=128, value=0.99),
            SimpleNamespace(move=action_to_move(42), visits=0, value=0.0),
        ])
        self.assertEqual(move_to_action(select_tactical_move(result, choices)), 42)

    def test_cpu_opponent_default_blocks_bad_policy_blunder(self):
        model = PolicyValueNet(channels=8, residual_blocks=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_head[-1].bias[35] = 100.0
        state = human_game()
        before = record(state)
        # The old four positional constructor arguments remain supported.
        guarded = self.play_ai.Opponent(model, "cpu", 2, 42)
        original = self.play_ai.Opponent(model, "cpu", 2, 42, tactical_checks=False)
        self.assertEqual(move_to_action(original.choose(state)), 35)
        self.assertEqual(move_to_action(guarded.choose(state)), 42)
        self.assertEqual(record(state), before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_opponent_uses_tactical_search_and_fpu(self):
        model = PolicyValueNet(channels=8, residual_blocks=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_head[-1].bias[35] = 100.0
        state = human_game()
        before = record(state)
        opponent = self.play_ai.Opponent(model, "cuda", 2, 42)
        self.assertTrue(opponent.searcher.options.tactical_checks)
        self.assertEqual(opponent.searcher.options.fpu_reduction, 0.0)
        self.assertEqual(move_to_action(opponent.choose(state)), 42)
        self.assertEqual(record(state), before)

    def test_terminal_and_invalid_inputs(self):
        state = engine.State()
        state.pass_turn()
        state.pass_turn()
        choices = analyze_tactics(state)
        self.assertFalse(choices.preferred_actions)
        with self.assertRaises(ValueError):
            select_tactical_move(SimpleNamespace(best_move=None, moves=[]), choices)
        with self.assertRaises(TypeError):
            analyze_tactics(None)


if __name__ == "__main__":
    unittest.main()
