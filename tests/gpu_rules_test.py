"""Compare CUDA-resident rule transitions and observations with the C++ oracle."""

import random
import unittest

import torch
import my_board_engine as engine

from kingdom_ai.encoding import encode_state, move_to_action
from kingdom_ai.gpu_rules import GpuStateBatch


REASON_CODES = {
    engine.EndReason.None_: 0,
    engine.EndReason.Capture: 1,
    engine.EndReason.Suicide: 2,
    engine.EndReason.TwoPasses: 3,
}


def make_board(black=(), white=(), neutral=None):
    board = engine.Board(neutral=neutral)
    for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), color):
                raise AssertionError("Invalid GPU rule fixture")
    return board


def cpu_record(state):
    """Independent serialized reference, including explicit reason translation."""
    return (
        [int(cell) for cell in state.board.cells]
        + [int(owner) for owner in state.ownership]
        + [int(state.to_play), state.remaining_stones(engine.Cell.Black),
           state.remaining_stones(engine.Cell.White), state.consecutive_passes,
           REASON_CODES[state.result.reason], int(state.result.winner),
           state.result.captured_stones, 0]
    )


def apply_action(state, action):
    move = engine.Move.pass_turn() if action == 81 else engine.Move.place(*divmod(action, 9))
    return state.play(move).accepted()


def last_liberty_state(actor):
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


def square_enclosure(neutral=None, enemy=False):
    points = []
    for coordinate in range(2, 7):
        points.extend(((2, coordinate), (6, coordinate)))
    for row in range(3, 6):
        points.extend(((row, 2), (row, 6)))
    return make_board(points, ((4, 4),) if enemy else (), neutral=neutral)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
class GpuRulesTest(unittest.TestCase):
    def assert_same_states(self, batch, states, *, check_encoding=True):
        self.assertEqual(batch.states.device.type, "cuda")
        self.assertEqual(batch.states.dtype, torch.int32)
        self.assertEqual(tuple(batch.states.shape), (len(states), 170))
        self.assertEqual(batch.states.cpu().tolist(), [cpu_record(state) for state in states])
        if check_encoding:
            features, masks = batch.encode()
            self.assertEqual(features.device.type, "cuda")
            self.assertEqual(masks.device.type, "cuda")
            self.assertEqual(features.dtype, torch.float32)
            self.assertEqual(masks.dtype, torch.bool)
            self.assertEqual(tuple(features.shape), (len(states), 10, 9, 9))
            self.assertEqual(tuple(masks.shape), (len(states), 82))
            expected = [encode_state(state) for state in states]
            torch.testing.assert_close(features.cpu(), torch.stack([item.features for item in expected]),
                                       rtol=0, atol=0)
            torch.testing.assert_close(masks.cpu(), torch.stack([item.legal_mask for item in expected]),
                                       rtol=0, atol=0)

    def step(self, batch, states, actions, *, check_encoding=True):
        expected = [apply_action(state, action) for state, action in zip(states, actions)]
        accepted = batch.play(torch.tensor(actions, device="cuda", dtype=torch.int32))
        self.assertEqual(accepted.device.type, "cuda")
        self.assertEqual(accepted.dtype, torch.bool)
        self.assertEqual(accepted.cpu().tolist(), expected)
        self.assert_same_states(batch, states, check_encoding=check_encoding)
        return expected

    def test_a_nondefault_stream_first_use_orders_import_encode_and_play(self):
        # Alphabetical ordering makes this the first rules-kernel use when
        # this file runs alone. Keep the whole producer/consumer chain on a
        # non-default PyTorch stream; synchronize only at the final boundary.
        states = [engine.State(), engine.State(neutral=None)]
        expected_before = [encode_state(state) for state in states]
        stream = torch.cuda.Stream()
        self.assertNotEqual(stream.cuda_stream, torch.cuda.default_stream().cuda_stream)
        with torch.cuda.stream(stream):
            batch = GpuStateBatch.from_engine(states)
            features_before, masks_before = batch.encode()
            accepted = batch.play(torch.tensor([0, 80], device="cuda", dtype=torch.int32))
            records_after = batch.states.clone()
            features_after, masks_after = batch.encode()
        stream.synchronize()
        self.assertEqual(accepted.cpu().tolist(), [True, True])
        torch.testing.assert_close(features_before.cpu(), torch.stack([item.features for item in expected_before]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(masks_before.cpu(), torch.stack([item.legal_mask for item in expected_before]),
                                   rtol=0, atol=0)
        self.assertTrue(apply_action(states[0], 0))
        self.assertTrue(apply_action(states[1], 80))
        self.assertEqual(records_after.cpu().tolist(), [cpu_record(state) for state in states])
        expected_after = [encode_state(state) for state in states]
        torch.testing.assert_close(features_after.cpu(), torch.stack([item.features for item in expected_after]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(masks_after.cpu(), torch.stack([item.legal_mask for item in expected_after]),
                                   rtol=0, atol=0)

    def test_huge_int64_actions_are_rejected_without_wrapping_to_legal_points(self):
        states = [engine.State(), engine.State()]
        batch = GpuStateBatch.from_engine(states)
        before = batch.states.clone()
        accepted = batch.play(torch.tensor([-(2**40), 2**40], device="cuda", dtype=torch.int64))
        self.assertEqual(accepted.cpu().tolist(), [False, False])
        torch.testing.assert_close(batch.states, before, rtol=0, atol=0)
        self.assert_same_states(batch, states)

    def test_initial_clone_and_neutral_variants(self):
        for neutral in ((4, 4), (0, 8), None):
            with self.subTest(neutral=neutral):
                point = None if neutral is None else engine.Position(*neutral)
                states = [engine.State(neutral=point) for _ in range(3)]
                batch = GpuStateBatch.initial(3, device="cuda", neutral=neutral)
                self.assert_same_states(batch, states)
                copied = batch.clone()
                self.assertNotEqual(copied.states.data_ptr(), batch.states.data_ptr())
                unchanged = batch.states.clone()
                self.step(copied, [state.copy() for state in states], [0, 1, 2])
                torch.testing.assert_close(batch.states, unchanged, rtol=0, atol=0)

    def test_capture_groups_simultaneous_surround_neutral_wall_and_both_colors(self):
        states = [
            engine.State(make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),)), engine.Cell.Black),
            engine.State(make_board(((1, 2), (1, 3), (2, 1), (3, 2), (3, 3)),
                                   ((2, 2), (2, 3))), engine.Cell.Black),
            engine.State(make_board(((0, 6), (1, 6), (1, 8), (2, 7)),
                                   ((0, 7), (1, 7), (2, 8))), engine.Cell.Black),
            engine.State(make_board(((3, 3), (4, 2)), ((4, 3),), engine.Position(4, 4)),
                         engine.Cell.Black),
            engine.State(make_board(((2, 2),), ((1, 2), (2, 1), (3, 2))), engine.Cell.White),
            last_liberty_state(engine.Cell.Black),
            last_liberty_state(engine.Cell.White),
        ]
        batch = GpuStateBatch.from_engine(states, device="cuda")
        self.assert_same_states(batch, states)
        self.step(batch, states, [21, 22, 8, 48, 21, 40, 40])
        self.assertEqual([state.result.reason for state in states], [engine.EndReason.Capture] * 7)
        self.assertEqual([state.result.captured_stones for state in states], [1, 2, 2, 1, 1, 40, 40])
        # Further actions on every terminal row must be rejected unchanged.
        self.assertEqual(self.step(batch, states, [81] * len(states)), [False] * len(states))

    def test_a_remaining_liberty_does_not_capture(self):
        state = engine.State(make_board(((1, 2), (2, 1), (3, 2), (3, 3)),
                                       ((2, 2), (2, 3))), engine.Cell.Black)
        batch = GpuStateBatch.from_engine([state])
        self.step(batch, [state], [12])
        self.assertFalse(state.result.finished())
        self.assertEqual(state.board.count(engine.Cell.White), 2)

    def test_suicide_is_legal_and_loses_for_both_players(self):
        black = ((0, 1),)
        white = ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))
        states = [engine.State(make_board(black, white), engine.Cell.Black),
                  engine.State(make_board(white, black), engine.Cell.White)]
        batch = GpuStateBatch.from_engine(states)
        _, masks = batch.encode()
        self.assertTrue(masks[:, 10].all().item())
        self.assertEqual(self.step(batch, states, [10, 10]), [True, True])
        for state in states:
            self.assertEqual(state.result.reason, engine.EndReason.Suicide)
            self.assertEqual(state.result.winner, state.to_play)
            self.assertEqual(len(state.board.group_at(engine.Position(0, 1))), 2)

    def test_territory_edges_neutral_walls_and_enemy_exclusion(self):
        three_edges = make_board(tuple((row, 2) for row in range(9)), ((0, 3),))
        boards = [
            make_board(((1, 2), (2, 1), (2, 3), (3, 2))),
            make_board(((0, 3), (1, 2), (1, 4), (2, 1), (2, 4),
                        (3, 1), (3, 4), (4, 2), (4, 3), (4, 4))),
            make_board(((0, 2), (0, 5), (1, 3), (1, 4))),
            make_board(((0, 2), (1, 1), (2, 0))), three_edges,
            make_board(((4, 4),)), square_enclosure(enemy=True),
            square_enclosure(engine.Position(4, 4)),
            make_board(((3, 5), (4, 6), (5, 5)), neutral=engine.Position(4, 4)),
        ]
        states = [engine.State(board, engine.Cell.Black) for board in boards]
        self.assertEqual([state.score().black for state in states], [1, 5, 2, 3, 18, 0, 0, 8, 1])
        batch = GpuStateBatch.from_engine(states)
        self.assert_same_states(batch, states)
        self.step(batch, states, [81] * len(states))
        self.step(batch, states, [81] * len(states))
        self.assertEqual([state.result.winner for state in states], [
            engine.Cell.White, engine.Cell.Black, engine.Cell.White, engine.Cell.Black,
            engine.Cell.Black, engine.Cell.White, engine.Cell.White,
            engine.Cell.Black, engine.Cell.White,
        ])

    def test_closed_house_persists_and_forbids_both_players(self):
        state = engine.State(make_board(((1, 2), (2, 1), (2, 3))), engine.Cell.Black)
        batch = GpuStateBatch.from_engine([state])
        self.step(batch, [state], [29])  # Close the house at (2,2).
        self.assertEqual(state.territory_owner(engine.Position(2, 2)), engine.Cell.Black)
        self.step(batch, [state], [80])
        self.assertEqual(self.step(batch, [state], [20]), [False])
        self.step(batch, [state], [81])
        self.assertEqual(self.step(batch, [state], [20]), [False])
        self.assertEqual(state.score().black, 1)

    def test_illegal_placements_preserve_actor_passes_stock_and_ownership(self):
        states = [engine.State() for _ in range(4)]
        for state in states:
            state.place(0, 0)
            state.pass_turn()
        batch = GpuStateBatch.from_engine(states)
        before = batch.states.clone()
        self.assertEqual(self.step(batch, states, [0, 40, -1, 82]), [False] * 4)
        torch.testing.assert_close(batch.states, before, rtol=0, atol=0)
        self.step(batch, states, [1, 2, 3, 4])
        self.assertTrue(all(state.consecutive_passes == 0 for state in states))

    def test_default_41_stone_stock_cannot_be_exceeded(self):
        points = tuple(divmod(action, 9) for action in range(41))
        state = engine.State(make_board(points), engine.Cell.Black)
        self.assertEqual(state.remaining_stones(engine.Cell.Black), 0)
        batch = GpuStateBatch.from_engine([state])
        self.assert_same_states(batch, [state])
        _, mask = batch.encode()
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(mask[0, 81].item())
        self.assertEqual(self.step(batch, [state], [80]), [False])
        self.assertEqual(self.step(batch, [state], [81]), [True])

    def test_two_pass_threshold_requires_exactly_three_more_house_cells(self):
        boards = [make_board(((0, 2), (1, 1), (2, 0))),
                  make_board(((0, 2), (0, 5), (1, 3), (1, 4))),
                  engine.Board()]
        states = [engine.State(board, engine.Cell.Black) for board in boards]
        self.assertEqual([state.score().black - state.score().white for state in states], [3, 2, 0])
        batch = GpuStateBatch.from_engine(states)
        self.step(batch, states, [81] * 3)
        self.assertTrue(all(not state.result.finished() for state in states))
        self.step(batch, states, [81] * 3)
        self.assertEqual([state.result.reason for state in states], [engine.EndReason.TwoPasses] * 3)
        self.assertEqual([state.result.winner for state in states], [engine.Cell.Black, engine.Cell.White,
                                                                   engine.Cell.White])

    def test_batched_transitions_are_independent_of_batch_partitions(self):
        states = [engine.State(), engine.State(neutral=None), last_liberty_state(engine.Cell.Black),
                  engine.State(make_board(((0, 2), (1, 1), (2, 0))), engine.Cell.White)]
        actions = [0, 80, 40, 81]
        batched = GpuStateBatch.from_engine(states)
        singles = [GpuStateBatch.from_engine([state]) for state in states]
        self.step(batched, states, actions)
        for index, (single, action) in enumerate(zip(singles, actions)):
            single.play(torch.tensor([action], device="cuda", dtype=torch.int32))
            torch.testing.assert_close(single.states[0], batched.states[index], rtol=0, atol=0)

    def test_seeded_complete_games_match_cpu_at_every_transition(self):
        variants = [(4, 4), None, (0, 8), (4, 4), None, (8, 0), (4, 4), None]
        states = [engine.State(neutral=None if neutral is None else engine.Position(*neutral))
                  for neutral in variants]
        generators = [random.Random(6800 + index) for index in range(len(states))]
        batch = GpuStateBatch.from_engine(states)
        for ply in range(164):
            if all(state.result.finished() for state in states):
                break
            actions = [
                81 if state.result.finished() else move_to_action(generator.choice(state.legal_moves()))
                for state, generator in zip(states, generators)
            ]
            self.step(batch, states, actions, check_encoding=(ply % 7 == 0))
        self.assertTrue(all(state.result.finished() for state in states))
        self.assert_same_states(batch, states)

    def test_invalid_host_schema_and_gpu_action_shapes_are_rejected(self):
        for rules in (
            engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden),
            engine.GameRules(allow_own_territory_moves=True),
            engine.GameRules(allow_single_edge_territory=False),
            engine.GameRules(stones_per_player=40),
        ):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                GpuStateBatch.from_engine([engine.State(rules)])
        batch = GpuStateBatch.initial(2)
        before = batch.states.clone()
        for actions in (torch.tensor([0], device="cuda", dtype=torch.int32),
                        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
                        torch.tensor([0.0, 1.0], device="cuda")):
            with self.subTest(shape=actions.shape, dtype=actions.dtype), self.assertRaises((TypeError, ValueError)):
                batch.play(actions)
        torch.testing.assert_close(batch.states, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
