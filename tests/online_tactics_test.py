"""Bounded current-cycle tactical mining uses histories, never tensor guesses."""

import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

from kingdom_ai.encoding import action_to_move, encode_state
from kingdom_ai.online_tactics import TacticalPositionMiner
from kingdom_ai.tactical_positions import load_position, user_game_two_positions
from kingdom_ai.training import GameData, TrainingSample


def _actions(history):
    return tuple(81 if point == "pass" else (point[0] - 1) * 9 + point[1] - 1
                 for point in history)


def _game(history):
    """Produce an exact, small engine-played fixture with pre-move observations."""
    actions = _actions(history)
    state = engine.State()
    positions = []
    for action in actions:
        encoded = encode_state(state)
        policy = torch.zeros(82)
        policy[action] = 1.0
        positions.append((encoded, policy))
        if not state.play(action_to_move(action)).accepted():
            raise AssertionError("Invalid fixture move")
    if not state.result.finished():
        raise AssertionError("Fixture must finish")
    samples = [TrainingSample(encoded.features, encoded.legal_mask, policy,
                              1.0 if encoded.to_play == state.result.winner else -1.0,
                              encoded.to_play) for encoded, policy in positions]
    return GameData(samples, state.result.winner, state.result.reason, actions)


def _snapshots(game):
    state = engine.State()
    states = [state.copy()]
    for action in game.action_history:
        if not state.play(action_to_move(action)).accepted():
            raise AssertionError("Invalid fixture history")
        states.append(state.copy())
    return states


def _assert_state(test, left, right):
    test.assertEqual(left.board.cells, right.board.cells)
    test.assertEqual(left.ownership, right.ownership)
    test.assertEqual(left.to_play, right.to_play)
    test.assertEqual(left.consecutive_passes, right.consecutive_passes)
    test.assertEqual(left.remaining_stones(engine.Cell.Black),
                     right.remaining_stones(engine.Cell.Black))
    test.assertEqual(left.remaining_stones(engine.Cell.White),
                     right.remaining_stones(engine.Cell.White))
    test.assertEqual(left.result.reason, right.result.reason)
    test.assertEqual(left.result.winner, right.result.winner)


class OnlineTacticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing user game prefixes stop at 25, so append the observed finish.
        history = user_game_two_positions()[-1]["position"]["history"] + [[6, 2]]
        cls.capture = _game(history)
        cls.houses_and_passes = _game(
            [[1, 2], [9, 9], [2, 1], "pass", [9, 8], "pass", "pass"])

    def miner(self, max_cases=8, iteration=1, seed=103):
        return TacticalPositionMiner(max_cases,
            generator=torch.Generator(device="cpu").manual_seed(seed), iteration=iteration)

    def test_no_replay_or_feature_decoding_during_add_game(self):
        miner = self.miner()
        with patch("kingdom_ai.online_tactics.engine.State",
                   side_effect=AssertionError("Self-play must not replay eagerly")):
            miner.add_game(self.capture, 0)
        self.assertEqual(miner.seen_games, 1)
        self.assertEqual(miner.sampled_games, 1)
        self.assertEqual(miner.candidate_count, 0)
        self.assertGreater(len(miner.cases()), 0)

    def test_every_capture_prefix_is_exact_and_has_no_guessed_labels(self):
        miner = self.miner(iteration=12)
        original = tuple(sample.features.clone() for sample in self.capture.samples)
        miner.add_game(self.capture, 27)
        cases = miner.cases()
        snapshots = _snapshots(self.capture)
        self.assertLessEqual(len(cases), 4)
        self.assertTrue(cases)
        for case in cases:
            ply = case["source"]["ply"]
            self.assertEqual(case["id"], f"online_cycle_12_game_27_ply_{ply}")
            self.assertEqual(case["family_id"], "online_cycle_12_game_27")
            self.assertEqual(case["source"]["kind"], "fresh_self_play_replay")
            self.assertNotIn("expected", case)
            self.assertNotIn("policy", case)
            self.assertNotIn("value", case)
            self.assertFalse(load_position(case).result.finished())
            _assert_state(self, load_position(case), snapshots[ply])
        for tensor, sample in zip(original, self.capture.samples):
            self.assertTrue(torch.equal(tensor, sample.features))

    def test_multi_stone_threat_onset_followup_late_and_defense_are_all_retained(self):
        miner = self.miner(max_cases=8, iteration=0)
        miner.add_game(self.capture, 0)
        cases = {case["source"]["selection_slot"]: case for case in miner.cases()}
        self.assertEqual(set(cases),
                         {"threat_onset", "threat_follow_up", "best_overall", "atari_defense"})
        onset = cases["threat_onset"]
        followup = cases["threat_follow_up"]
        self.assertGreaterEqual(followup["source"]["ply"], onset["source"]["ply"] + 2)
        self.assertGreater(cases["best_overall"]["source"]["ply"], followup["source"]["ply"])
        onset_state, followup_state = load_position(onset), load_position(followup)
        opponent = engine.opponent(onset_state.to_play)
        chains = []
        for index, cell in enumerate(onset_state.board.cells):
            point = engine.Board.position(index)
            if (cell == opponent and len(onset_state.board.group_at(point)) >= 2
                    and len(onset_state.board.liberties(point)) == 2):
                chains.append(onset_state.board.group_at(point))
        self.assertTrue(chains)
        anchor = min(engine.Board.index(point) for point in max(chains, key=len))
        anchor_point = engine.Board.position(anchor)
        self.assertEqual(followup_state.board.cells[anchor], opponent)
        self.assertEqual(len(followup_state.board.liberties(anchor_point)), 2)
        self.assertGreaterEqual(len(followup_state.board.group_at(anchor_point)), 2)
        # Regression for the observed long chase: the nine-ply position is now
        # eligible without hard-coding the game's name/coordinates in mining.
        self.assertEqual(followup["source"]["ply"], 17)
        snapshots = _snapshots(self.capture)
        for case in cases.values():
            _assert_state(self, load_position(case), snapshots[case["source"]["ply"]])
            self.assertNotIn("expected", case)
            self.assertNotIn("policy", case)
            self.assertNotIn("value", case)

    def test_full_size_cycle_first_pass_includes_rotated_temporal_slots(self):
        miner = self.miner(max_cases=32, iteration=7)
        for index in range(512):
            miner.add_game(self.capture, index)
        cases = miner.cases()
        self.assertEqual(miner.sampled_games, 64)
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({case["family_id"] for case in cases}), 32)
        slots = {case["source"]["selection_slot"] for case in cases}
        self.assertEqual(slots,
                         {"threat_onset", "threat_follow_up", "best_overall", "atari_defense"})
        self.assertLessEqual(miner.candidate_count, 128)
        # Each game's first chosen slot is rotated before global selection, not
        # an always-best choice with onset positions left for a later pass.
        for case in cases:
            game_index = case["source"]["game_index"]
            expected = ("threat_onset", "threat_follow_up", "best_overall", "atari_defense")
            self.assertEqual(case["source"]["selection_slot"], expected[(7 + game_index) % 4])

    def test_prefix_replay_preserves_permanent_house_stock_and_pass_history(self):
        miner = self.miner()
        miner.add_game(self.houses_and_passes, 0)
        cases = miner.cases()
        snapshots = _snapshots(self.houses_and_passes)
        self.assertTrue(any("pass" in case["position"]["history"] for case in cases))
        for case in cases:
            ply = case["source"]["ply"]
            actual = load_position(case)
            _assert_state(self, actual, snapshots[ply])
            if ply >= 3:
                self.assertEqual(actual.territory_owner(engine.Position(0, 0)), engine.Cell.Black)
        self.assertTrue(any(case["source"]["ply"] >= 3 for case in cases))

    def test_completed_cycle_uniform_reservoir_and_candidate_bounds(self):
        miner = self.miner(max_cases=3)
        for index in range(200):
            miner.add_game(self.capture, index)
        self.assertEqual(miner.seen_games, 200)
        self.assertEqual(miner.sampled_games, 6)
        self.assertEqual(len(set(miner.sampled_game_indices)), 6)
        # A deterministic fixture verifies that reservoir sampling reaches late
        # games, instead of spending all solver calls on the first few games.
        self.assertGreater(max(miner.sampled_game_indices), 6)
        reference = list(range(6))
        generator = torch.Generator(device="cpu").manual_seed(103)
        for index in range(6, 200):
            slot = int(torch.randint(index + 1, (), generator=generator).item())
            if slot < 6:
                reference[slot] = index
        self.assertEqual(miner.sampled_game_indices, tuple(reference))
        cases = miner.cases()
        self.assertLessEqual(len(cases), 3)
        self.assertLessEqual(miner.candidate_count, 12)
        self.assertEqual(len({case["family_id"] for case in cases}), len(cases))

    def test_generator_roundtrip_and_cached_calls_are_reproducible(self):
        original_generator = torch.Generator(device="cpu").manual_seed(803)
        restored_generator = torch.Generator(device="cpu").manual_seed(999)
        restored_generator.set_state(original_generator.get_state())
        left = TacticalPositionMiner(4, generator=original_generator, iteration=9)
        right = TacticalPositionMiner(4, generator=restored_generator, iteration=9)
        for index in range(120):
            left.add_game(self.capture, index)
            right.add_game(self.capture, index)
        self.assertEqual(left.sampled_game_indices, right.sampled_game_indices)
        self.assertEqual(left.cases(), right.cases())
        before = original_generator.get_state().clone()
        cases = left.cases()
        cases[0]["position"]["history"].clear()
        self.assertTrue(left.cases()[0]["position"]["history"])
        self.assertTrue(torch.equal(before, original_generator.get_state()))

    def test_new_cycle_cannot_reuse_old_ids_and_no_games_means_no_cases(self):
        first, second = self.miner(iteration=1), self.miner(iteration=2)
        self.assertEqual(first.cases(), [])
        first.add_game(self.capture, 0)
        second.add_game(self.capture, 0)
        self.assertFalse({case["id"] for case in first.cases()}
                         & {case["id"] for case in second.cases()})

    def test_malformed_or_missing_action_history_fails_closed(self):
        for history in (None, (), [81, 81], (True, 81), (-1, 81), (82, 81),
                        (0.5, 81), (81,) * 165):
            with self.subTest(history=history), self.assertRaises(ValueError):
                self.miner().add_game(GameData([None] * (len(history) if history else 0),
                    engine.Cell.White, engine.EndReason.TwoPasses, history), 0)
        with self.assertRaises(ValueError):
            self.miner().add_game(GameData([], engine.Cell.White, engine.EndReason.TwoPasses,
                                          (81, 81)), 0)

    def test_exact_engine_replay_rejects_illegal_moves_and_wrong_results(self):
        for actions, winner, reason in (
                ((40, 81), engine.Cell.White, engine.EndReason.TwoPasses),
                ((0, 0, 81, 81), engine.Cell.White, engine.EndReason.TwoPasses),
                ((81, 81, 0), engine.Cell.White, engine.EndReason.TwoPasses),
                ((0, 1), engine.Cell.White, engine.EndReason.TwoPasses),
                ((81, 81), engine.Cell.Black, engine.EndReason.TwoPasses),
                ((81, 81), engine.Cell.White, engine.EndReason.Capture)):
            with self.subTest(actions=actions, winner=winner, reason=reason):
                miner = self.miner()
                miner.add_game(GameData([None] * len(actions), winner, reason, actions), 0)
                with self.assertRaises(ValueError):
                    miner.cases()

    def test_options_reject_invalid_values(self):
        generator = torch.Generator(device="cpu")
        for max_cases in (0, -1, True, 1.5):
            with self.subTest(max_cases=max_cases), self.assertRaises(ValueError):
                TacticalPositionMiner(max_cases, generator=generator)
        for iteration in (-1, True, 1.5):
            with self.subTest(iteration=iteration), self.assertRaises(ValueError):
                TacticalPositionMiner(1, generator=generator, iteration=iteration)
        with self.assertRaises(ValueError):
            TacticalPositionMiner(1, generator=object())
        with self.assertRaises(TypeError):
            self.miner().add_game(object(), 0)
        with self.assertRaises(ValueError):
            self.miner().add_game(self.capture, -1)


if __name__ == "__main__":
    unittest.main()
