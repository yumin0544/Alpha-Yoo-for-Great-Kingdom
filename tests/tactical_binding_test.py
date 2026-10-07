"""Independent finite-horizon oracle for the public, nonmutating proof API."""

import unittest

import my_board_engine as engine


def action(move):
    return 81 if move.is_pass() else move.point.row * 9 + move.point.col


def snapshot(state):
    return (tuple(state.board.cells), tuple(state.ownership), state.to_play,
            state.consecutive_passes, state.remaining_stones(engine.Cell.Black),
            state.remaining_stones(engine.Cell.White), state.result.reason,
            state.result.winner, state.result.captured_stones)


def oracle(state, depth):
    """No heuristics or transpositions: all legal moves at each visited node."""
    if state.result.finished():
        return 1 if state.result.winner == state.to_play else -1
    if depth == 0:
        return 0
    values = []
    for move in state.legal_moves():
        child = state.copy()
        assert child.play(move).accepted()
        values.append(-oracle(child, depth - 1))
        if values[-1] == 1:
            return 1
    return -1 if values and all(value == -1 for value in values) else 0


def setup(black=(), white=(), actor=engine.Cell.Black):
    board = engine.Board()
    for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            assert board.place(engine.Position(row, col), color)
    return engine.State(board, actor)


class TacticalBindingTest(unittest.TestCase):
    def assert_sound(self, state, depth=2):
        original = snapshot(state)
        result = engine.solve_tactics(state, engine.TacticalSolverOptions(
            max_depth=depth, max_nodes=100000, time_limit_ms=10000))
        self.assertEqual(snapshot(state), original)
        expected = oracle(state, depth)
        outcome = {engine.TacticalOutcome.Win: 1, engine.TacticalOutcome.Loss: -1,
                   engine.TacticalOutcome.Unknown: 0}[result.outcome]
        if outcome:
            self.assertEqual(outcome, expected)
        legal = {action(move) for move in state.legal_moves()}
        sets = [{action(move) for move in moves} for moves in (
            result.winning_moves, result.losing_moves, result.unknown_moves)]
        self.assertEqual(set.union(*sets), legal)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for moves, value in ((result.winning_moves, 1), (result.losing_moves, -1)):
            for move in moves:
                child = state.copy()
                self.assertTrue(child.play(move).accepted())
                self.assertEqual(-oracle(child, depth - 1), value)
        return result

    def test_initial_position_is_not_a_proved_draw_or_loss(self):
        result = self.assert_sound(engine.State())
        self.assertEqual(result.outcome, engine.TacticalOutcome.Unknown)
        self.assertIn(81, {action(move) for move in result.losing_moves})

    def test_capture_suicide_and_double_atari_match_full_reply_oracle(self):
        cases = (
            setup(((0, 1),), ((0, 0),)),
            setup(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
            setup(((0, 0), (0, 8)), ((0, 1), (1, 1), (0, 7), (1, 7))),
        )
        for state in cases:
            with self.subTest(board=tuple(state.board.cells)):
                self.assert_sound(state)

    def test_pass_and_terminal_perspectives(self):
        state = engine.State()
        state.pass_turn()
        result = self.assert_sound(state, 1)
        self.assertEqual(result.outcome, engine.TacticalOutcome.Win)
        self.assertIn(81, {action(move) for move in result.winning_moves})
        state.pass_turn()
        result = engine.solve_tactics(state)
        self.assertEqual(result.outcome, engine.TacticalOutcome.Loss)
        self.assertFalse(result.winning_moves or result.losing_moves or result.unknown_moves)

    def test_tiny_budget_never_invents_a_proof(self):
        state = engine.State()
        result = engine.solve_tactics(state, engine.TacticalSolverOptions(
            max_depth=12, max_nodes=1, time_limit_ms=1000))
        self.assertEqual(result.outcome, engine.TacticalOutcome.Unknown)
        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(result.nodes, 1)
        self.assertEqual(snapshot(state), snapshot(engine.State()))

    def test_user_game_forced_capture_beyond_eight_plies(self):
        from kingdom_ai.tactical_positions import load_position, user_game_two_positions
        case = next(case for case in user_game_two_positions()
                    if case["id"] == "user_game_2_after_ply_17")
        state = load_position(case)
        original = snapshot(state)
        result = engine.solve_tactics(state, engine.TacticalSolverOptions(
            max_depth=9, max_nodes=2000000, time_limit_ms=0))
        self.assertEqual(result.outcome, engine.TacticalOutcome.Win)
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(result.proof_depth, 9)
        self.assertIn(29, {action(move) for move in result.winning_moves})  # (4,3)
        played = state.copy()
        for move in result.principal_variation:
            self.assertTrue(played.play(move).accepted())
        self.assertEqual(played.result.reason, engine.EndReason.Capture)
        self.assertEqual(played.result.winner, engine.Cell.White)
        self.assertEqual(played.result.captured_stones, 7)
        self.assertEqual(snapshot(state), original)

    def test_invalid_options_rejected(self):
        for options in (engine.TacticalSolverOptions(max_depth=-1),
                        engine.TacticalSolverOptions(max_nodes=0)):
            with self.assertRaises(ValueError):
                engine.solve_tactics(engine.State(), options)
        with self.assertRaises(ValueError):
            engine.TacticalSolverOptions(time_limit_ms=-1)


if __name__ == "__main__":
    unittest.main()
