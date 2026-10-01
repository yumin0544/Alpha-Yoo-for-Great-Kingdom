"""Behavior and C++ parity checks for the installed/native Python extension.

Run directly, or supply the reference executable built beside the extension:
    python tests/python_binding_test.py --reference build-python/binding_reference.exe
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import threading
import time
import unittest

import my_board_engine as engine


def make_board(black=(), white=(), neutral=None):
    board = engine.Board(neutral=neutral)
    for points, player in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), player):
                raise AssertionError("Invalid Python reference fixture")
    return board


def enum_number(value):
    return int(value.value)


def state_snapshot(state):
    score = state.score()
    result = state.result
    return {
        "board": state.board.to_string(),
        "to_play": enum_number(state.to_play),
        "passes": state.consecutive_passes,
        "remaining": [state.remaining_stones(engine.Cell.Black),
                      state.remaining_stones(engine.Cell.White)],
        "score": [score.black, score.white],
        "owners": [enum_number(owner) for owner in state.ownership],
        "result": {
            "winner": enum_number(result.winner),
            "reason": enum_number(result.reason),
            "score": [result.score.black, result.score.white],
            "captured": result.captured_stones,
            "finished": result.finished(),
        },
    }


def move_snapshot(move):
    if move is None:
        return None
    if move.is_pass():
        return "pass"
    point = move.point
    return [point.row, point.col]


def search_snapshot(result):
    # Wall-clock duration is intentionally excluded from deterministic parity.
    return {
        "best_move": move_snapshot(result.best_move),
        "simulations": result.simulations,
        "nodes": result.nodes,
        "total_rollout_plies": result.total_rollout_plies,
        "win_rate": result.win_rate,
        "moves": [{"move": move_snapshot(item.move), "visits": item.visits,
                   "win_rate": item.win_rate} for item in result.moves],
    }


def reference_cases():
    place = engine.Move.place
    pass_move = engine.Move.pass_turn
    suicide = make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1)))
    house = make_board(((1, 2), (2, 1), (2, 3), (3, 2)))
    forbidden = engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden)
    limited = engine.GameRules(stones_per_player=1)
    return [
        ("initial", engine.State(), []),
        ("opening", engine.State(), [place(0, 0), pass_move(), place(4, 4),
                                     place(-1, 0), place(0, 1)]),
        ("capture", engine.State(make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),)),
                                 engine.Cell.Black), [place(2, 3), pass_move()]),
        ("suicide", engine.State(suicide, engine.Cell.Black), [place(1, 1)]),
        ("forbidden_suicide", engine.State(suicide, engine.Cell.Black, forbidden),
         [place(1, 1), pass_move()]),
        ("simultaneous", engine.State(
            make_board(((0, 6), (1, 6), (1, 8), (2, 7)), ((0, 7), (1, 7), (2, 8))),
            engine.Cell.Black), [place(0, 8)]),
        ("own_house", engine.State(house, engine.Cell.Black), [place(2, 2)]),
        ("opponent_house", engine.State(house, engine.Cell.White), [place(2, 2)]),
        ("two_passes", engine.State(), [pass_move(), pass_move()]),
        ("stock", engine.State(limited), [place(0, 0), place(8, 8), place(0, 1),
                                         pass_move(), pass_move()]),
    ]


class PythonBindingTest(unittest.TestCase):
    reference_executable = None

    def test_default_rules_and_zero_based_coordinates(self):
        state = engine.State()
        self.assertEqual(state.to_play, engine.Cell.Black)
        self.assertEqual(state.board.at(engine.Position(4, 4)), engine.Cell.Neutral)
        self.assertEqual(state.board.count(engine.Cell.Empty), 80)
        self.assertEqual(state.remaining_stones(engine.Cell.Black), 41)
        self.assertEqual(state.remaining_stones(engine.Cell.White), 41)
        self.assertEqual(state.rules.suicide_rule, engine.SuicideRule.Loses)
        self.assertFalse(state.rules.allow_own_territory_moves)
        self.assertTrue(state.rules.allow_single_edge_territory)
        self.assertTrue(state.place(0, 0).accepted())
        self.assertEqual(state.board.at(engine.Position(0, 0)), engine.Cell.Black)
        self.assertEqual(state.remaining_stones(engine.Cell.Black), 40)
        self.assertEqual(state.to_play, engine.Cell.White)
        self.assertEqual(engine.Position(0, 0), engine.Position(0, 0))
        self.assertEqual(engine.Move.place(2, 3), engine.Move.place(2, 3))
        self.assertTrue(engine.Move.pass_turn().is_pass())

    def test_neutral_variants_and_board_sequence_conversion(self):
        self.assertEqual(engine.State(neutral=None).board.count(engine.Cell.Empty), 81)
        shifted = engine.State(neutral=engine.Position(1, 2))
        self.assertEqual(shifted.board.at(engine.Position(1, 2)), engine.Cell.Neutral)
        self.assertEqual(shifted.board.at(engine.Position(4, 4)), engine.Cell.Empty)
        cells = [engine.Cell.Empty] * 81
        cells[9] = engine.Cell.White
        board = engine.Board(cells)
        cells[9] = engine.Cell.Black
        self.assertEqual(board.at(engine.Position(1, 0)), engine.Cell.White)
        state = engine.State(board, engine.Cell.White)
        board.clear(engine.Position(1, 0))
        self.assertEqual(state.board.at(engine.Position(1, 0)), engine.Cell.White)

    def test_rejected_move_preserves_pass_counter_and_state(self):
        state = engine.State()
        state.pass_turn()
        before = state_snapshot(state)
        for row, col, error in ((4, 4, engine.MoveError.Occupied),
                                (-1, 0, engine.MoveError.OutOfBounds),
                                (0, 9, engine.MoveError.OutOfBounds)):
            with self.subTest(row=row, col=col):
                move = engine.Move.place(row, col)
                self.assertFalse(state.is_legal(move))
                outcome = state.play(move)
                self.assertFalse(outcome.accepted())
                self.assertEqual(outcome.error, error)
                self.assertEqual(state_snapshot(state), before)

    def test_capture_and_terminal_state(self):
        board = make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),))
        self.assertEqual(board.liberties(engine.Position(2, 2)), [engine.Position(2, 3)])
        self.assertEqual(board.group_at(engine.Position(2, 2)), [engine.Position(2, 2)])
        state = engine.State(board, engine.Cell.Black)
        outcome = state.place(2, 3)
        self.assertTrue(outcome.accepted())
        self.assertTrue(outcome.result.finished())
        self.assertEqual(state.result.reason, engine.EndReason.Capture)
        self.assertEqual(state.result.winner, engine.Cell.Black)
        self.assertEqual(state.result.captured_stones, 1)
        self.assertEqual(state.board.at(engine.Position(2, 2)), engine.Cell.Empty)
        self.assertEqual(state.legal_moves(), [])
        self.assertEqual(state.pass_turn().error, engine.MoveError.GameOver)

    def test_suicide_and_analysis_variant(self):
        board = make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1)))
        move = engine.Move.place(1, 1)
        state = engine.State(board, engine.Cell.Black)
        self.assertTrue(state.is_legal(move))
        self.assertIn(move, state.legal_moves())
        self.assertTrue(state.play(move).accepted())
        self.assertEqual(state.result.reason, engine.EndReason.Suicide)
        self.assertEqual(state.result.winner, engine.Cell.White)
        self.assertEqual(state.board.at(engine.Position(1, 1)), engine.Cell.Black)
        forbidden = engine.State(board, engine.Cell.Black,
                                 engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden))
        before = state_snapshot(forbidden)
        self.assertEqual(forbidden.play(move).error, engine.MoveError.Suicide)
        self.assertEqual(state_snapshot(forbidden), before)

    def test_simultaneous_surround_prefers_mover(self):
        board = make_board(((0, 6), (1, 6), (1, 8), (2, 7)),
                           ((0, 7), (1, 7), (2, 8)))
        state = engine.State(board, engine.Cell.Black)
        self.assertTrue(state.place(0, 8).accepted())
        self.assertEqual(state.result.reason, engine.EndReason.Capture)
        self.assertEqual(state.result.winner, engine.Cell.Black)
        self.assertEqual(state.result.captured_stones, 2)
        self.assertEqual(state.board.at(engine.Position(0, 8)), engine.Cell.Black)
        self.assertEqual(state.board.at(engine.Position(2, 8)), engine.Cell.White)

    def test_territory_bans_and_single_edge_house(self):
        house = make_board(((1, 2), (2, 1), (2, 3), (3, 2)))
        territory = house.territory()
        self.assertEqual((territory.black, territory.white), (1, 0))
        self.assertEqual(territory.owners[2 * 9 + 2], engine.Cell.Black)
        for player, error in ((engine.Cell.Black, engine.MoveError.OwnTerritory),
                              (engine.Cell.White, engine.MoveError.OpponentTerritory)):
            with self.subTest(player=player):
                state = engine.State(house, player)
                self.assertEqual(state.territory_owner(engine.Position(2, 2)),
                                 engine.Cell.Black)
                self.assertEqual(state.place(2, 2).error, error)
        edge = make_board(((0, 2), (0, 5), (1, 3), (1, 4)))
        self.assertEqual(edge.territory().black, 2)
        self.assertEqual(edge.territory(allow_single_edge=False).black, 0)

    def test_stock_and_two_passes_scoring(self):
        self.assertEqual(engine.Score(black=15, white=12).winner(), engine.Cell.Black)
        self.assertEqual(engine.Score(black=15, white=13).winner(), engine.Cell.White)
        self.assertEqual(engine.Score(black=10, white=10).winner(), engine.Cell.White)
        state = engine.State(engine.GameRules(stones_per_player=1))
        self.assertTrue(state.place(0, 0).accepted())
        self.assertTrue(state.place(8, 8).accepted())
        self.assertEqual(state.remaining_stones(engine.Cell.Black), 0)
        self.assertEqual(state.place(0, 1).error, engine.MoveError.NoStones)
        self.assertEqual(state.legal_moves(), [engine.Move.pass_turn()])
        self.assertTrue(state.pass_turn().accepted())
        self.assertFalse(state.result.finished())
        self.assertTrue(state.pass_turn().accepted())
        self.assertEqual(state.result.reason, engine.EndReason.TwoPasses)
        self.assertEqual(state.result.winner, engine.Cell.White)

    def test_copy_retains_claims_passes_and_terminal_result(self):
        state = engine.State(make_board(((0, 2), (1, 1), (2, 0))), engine.Cell.Black)
        state.pass_turn()
        copied = state.copy()
        self.assertEqual(state_snapshot(copied), state_snapshot(state))
        copied.pass_turn()
        self.assertTrue(copied.result.finished())
        self.assertFalse(state.result.finished())
        self.assertEqual(state.consecutive_passes, 1)
        terminal_copy = copied.copy()
        self.assertEqual(state_snapshot(terminal_copy), state_snapshot(copied))
        self.assertEqual(terminal_copy.place(8, 8).error, engine.MoveError.GameOver)

    def test_mutable_snapshots_cannot_bypass_state_rules(self):
        state = engine.State()
        before = state_snapshot(state)
        detached = state.board
        self.assertTrue(detached.place(engine.Position(0, 0), engine.Cell.White))
        rules = state.rules
        rules.stones_per_player = 1
        rules.allow_own_territory_moves = True
        cells = state.board.cells
        cells[4 * 9 + 4] = engine.Cell.Empty
        owners = state.ownership
        owners[0] = engine.Cell.White
        self.assertEqual(state_snapshot(state), before)
        self.assertEqual(state.rules.stones_per_player, 41)
        move = engine.Move.place(2, 3)
        point = move.point
        point.row = 8
        self.assertEqual(move.point, engine.Position(2, 3))
        options = engine.MCTSOptions(simulations=96, seed=11)
        search = engine.MCTS(options)
        options.simulations = 1
        exposed_options = search.options
        exposed_options.simulations = 2
        self.assertEqual(search.options.simulations, 96)

    def test_cpp_exceptions_translate_to_python(self):
        with self.assertRaises(IndexError):
            engine.Board().at(engine.Position(9, 0))
        with self.assertRaises(IndexError):
            engine.State(neutral=engine.Position(-1, 0))
        with self.assertRaises(ValueError):
            engine.Board().clear(engine.Position(4, 4))
        with self.assertRaises(ValueError):
            engine.State().remaining_stones(engine.Cell.Neutral)
        with self.assertRaises(ValueError):
            engine.State(engine.GameRules(stones_per_player=0))
        with self.assertRaises(ValueError):
            engine.State(engine.Board(), engine.Cell.Empty)
        with self.assertRaises((TypeError, ValueError)):
            engine.Board([engine.Cell.Empty] * 80)
        for exploration in (-1.0, float("nan"), float("inf")):
            with self.subTest(exploration=exploration), self.assertRaises(ValueError):
                engine.MCTS(engine.MCTSOptions(exploration=exploration))
        with self.assertRaises(ValueError):
            engine.MCTS(engine.MCTSOptions(simulations=0))

    def test_mcts_outputs_legal_move_and_preserves_state(self):
        state = engine.State()
        state.place(0, 0)
        state.pass_turn()
        before = state_snapshot(state)
        result = engine.MCTS(engine.MCTSOptions(simulations=128, seed=1729)).search(state)
        self.assertEqual(result.simulations, 128)
        self.assertIsNotNone(result.best_move)
        self.assertTrue(state.is_legal(result.best_move))
        self.assertEqual(sum(item.visits for item in result.moves), 128)
        self.assertEqual(state_snapshot(state), before)
        self.assertTrue(math.isfinite(result.elapsed_seconds))
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        for item in result.moves:
            self.assertTrue(state.is_legal(item.move))
            self.assertGreaterEqual(item.win_rate, 0.0)
            self.assertLessEqual(item.win_rate, 1.0)
        terminal = state.copy()
        terminal.pass_turn()
        empty = engine.MCTS().search(terminal)
        self.assertIsNone(empty.best_move)
        self.assertEqual((empty.simulations, empty.nodes, empty.total_rollout_plies),
                         (0, 0, 0))
        self.assertEqual(empty.moves, [])

    def test_mcts_releases_gil_during_search(self):
        stop = threading.Event()
        ready = threading.Event()
        ticks = []

        def record_progress():
            ready.set()
            while not stop.is_set():
                ticks.append(time.monotonic())
                time.sleep(0.002)

        worker = threading.Thread(target=record_progress, daemon=True)
        worker.start()
        self.assertTrue(ready.wait(timeout=5.0))
        search = engine.MCTS(engine.MCTSOptions(simulations=1_000_000, time_limit_ms=100))
        try:
            started = time.monotonic()
            result = search.search(engine.State())
            ended = time.monotonic()
        finally:
            stop.set()
            worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertGreater(result.simulations, 0)
        duration = ended - started
        self.assertTrue(any(started + duration * 0.1 < tick < started + duration * 0.9
                            for tick in ticks), "Python made no progress inside C++ search")

    def test_same_mcts_concurrent_searches_preserve_rng_sequence(self):
        state = engine.State()
        options = engine.MCTSOptions(simulations=128, seed=2255)
        serial = engine.MCTS(options)
        expected = [search_snapshot(serial.search(state)) for _ in range(2)]
        concurrent = engine.MCTS(options)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(concurrent.search, state) for _ in range(2)]
            actual = [search_snapshot(future.result(timeout=30)) for future in futures]
        # Both calls have the same input; scheduling may reverse their completion order.
        encoded = lambda values: sorted(json.dumps(value, sort_keys=True) for value in values)
        self.assertEqual(encoded(actual), encoded(expected))

    def test_exact_cpp_and_python_parity(self):
        if self.reference_executable is None:
            self.skipTest("Supply --reference for same-compiler C++ parity verification")
        completed = subprocess.run([str(self.reference_executable)], check=True,
                                   capture_output=True, text=True, encoding="utf-8", timeout=30)
        reference = json.loads(completed.stdout)
        actual = []
        for name, state, moves in reference_cases():
            errors = [enum_number(state.play(move).error) for move in moves]
            actual.append({"name": name, "errors": errors, "state": state_snapshot(state)})
        self.assertEqual(actual, reference["cases"])
        state = engine.State()
        state.place(0, 0)
        state.pass_turn()
        search = engine.MCTS(engine.MCTSOptions(simulations=192, exploration=1.25, seed=99113))
        self.assertEqual(search_snapshot(search.search(state)), reference["search"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path)
    arguments, remaining = parser.parse_known_args()
    PythonBindingTest.reference_executable = arguments.reference
    unittest.main(argv=[__file__, *remaining], verbosity=2)
