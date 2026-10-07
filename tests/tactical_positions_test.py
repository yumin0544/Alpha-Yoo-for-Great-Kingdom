"""Engine-replay validation of user observations and composed curricula."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import my_board_engine as engine

from kingdom_ai.tactical_positions import (curriculum_positions, load_casebook,
                                           load_position, replay_position,
                                           user_game_two_positions)

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data/tactics/user_game_2_replay_2026_10_07.json"
CASEBOOK = ROOT / "data/tactics/user_capture_curriculum_2026_10_07.json"


class TacticalPositionTests(unittest.TestCase):
    def test_exact_user_game_replay_all_27_console_snapshots(self):
        states = replay_position(REPLAY)
        self.assertEqual(len(states), 27)
        self.assertEqual(states[0].board.count(engine.Cell.Black), 0)
        final = states[-1]
        self.assertEqual(final.result.winner, engine.Cell.White)
        self.assertEqual(final.result.reason, engine.EndReason.Capture)
        self.assertEqual(final.result.captured_stones, 6)
        self.assertEqual(final.remaining_stones(engine.Cell.Black), 28)
        self.assertEqual(final.remaining_stones(engine.Cell.White), 28)
        # Reconstructing only the post-capture board loses Black's consumed
        # supply and the terminal flag; replay deliberately avoids that bug.
        lossy = engine.State(final.board, final.to_play)
        self.assertEqual(lossy.remaining_stones(engine.Cell.Black), 34)
        self.assertFalse(lossy.result.finished())
        self.assertTrue(final.result.finished())

    def test_user_central_group_is_chased_nine_recorded_plies_not_assumed_best_play(self):
        states = replay_position(REPLAY)
        point = engine.Position(2, 4)
        self.assertEqual(len(states[17].board.group_at(point)), 3)
        self.assertEqual(len(states[17].board.liberties(point)), 2)
        self.assertEqual(26 - 17, 9)
        for ply in (16, 18, 20, 22, 24):
            self.assertEqual(len(states[ply].board.liberties(point)), 1)
        alternate = states[24].copy()
        self.assertTrue(alternate.place(5, 1).accepted())  # Black (6,2), unlike log
        self.assertFalse(alternate.result.finished())
        self.assertTrue(alternate.place(5, 0).accepted())  # White (6,1)
        self.assertEqual(alternate.result.winner, engine.Cell.White)
        self.assertEqual(alternate.result.captured_stones, 7)

    def test_user_prefixes_are_full_states_and_one_family(self):
        states = replay_position(REPLAY)
        checked = load_casebook(CASEBOOK)
        builtin = user_game_two_positions()
        self.assertEqual(len(checked), 12)
        self.assertEqual([case["id"] for case in checked], [case["id"] for case in builtin])
        self.assertEqual(len({case["family_id"] for case in checked + builtin}), 1)
        for ply, (recorded, composed) in enumerate(zip(checked, builtin), 14):
            for case in (recorded, composed):
                state = load_position(case)
                self.assertEqual(state.board.cells, states[ply].board.cells)
                self.assertEqual(state.ownership, states[ply].ownership)
                self.assertEqual(state.to_play, states[ply].to_play)
                self.assertEqual(state.remaining_stones(engine.Cell.Black),
                                 states[ply].remaining_stones(engine.Cell.Black))
                self.assertFalse(state.result.finished())

    def test_curriculum_has_seven_original_motifs_and_no_hand_labels(self):
        cases = curriculum_positions()
        self.assertEqual(len(cases), 17)
        self.assertEqual(len({case["motif"] for case in cases}), 7)
        for case in cases:
            state = load_position(case)
            self.assertFalse(state.result.finished())
            self.assertGreater(len(state.legal_moves()), 1)
            self.assertNotIn("value", case)
            self.assertNotIn("best_move", case)
        ladders = {c["family_id"] for c in cases
                   if c["id"] in {"ladder_open_board", "ladder_neutral_wall", "ladder_breaker"}}
        self.assertEqual(len(ladders), 1)
        self.assertEqual(load_position(next(c for c in cases if c["id"] == "ladder_open_board"))
                         .board.count(engine.Cell.Neutral), 0)

    def test_composed_immediate_capture_and_counter_capture_are_real_engine_wins(self):
        by_name = {case["id"]: case for case in curriculum_positions()}
        for name, point, winner, count in (
                ("capture_corner", (2, 1), engine.Cell.Black, 1),
                ("capture_edge", (2, 4), engine.Cell.Black, 1),
                ("capture_interior", (2, 3), engine.Cell.Black, 1),
                ("capture_chain", (2, 4), engine.Cell.Black, 2),
                ("double_atari_shared_liberty", (3, 3), engine.Cell.Black, 2),
                ("counter_capture", (3, 1), engine.Cell.White, 1)):
            with self.subTest(name=name):
                state = load_position(by_name[name])
                self.assertTrue(state.place(point[0] - 1, point[1] - 1).accepted())
                self.assertEqual(state.result.winner, winner)
                self.assertEqual(state.result.reason, engine.EndReason.Capture)
                self.assertEqual(state.result.captured_stones, count)

    def test_history_preserves_ownership_passes_and_snapshots(self):
        spec = {"format_version": 1, "id": "one_pass", "position": {
            "history": [[1, 2], [9, 9], [2, 1], "pass"]}}
        snapshots = replay_position(spec)
        self.assertEqual(snapshots[-1].consecutive_passes, 1)
        self.assertEqual(snapshots[-1].territory_owner(engine.Position(0, 0)), engine.Cell.Black)
        snapshots[-1].pass_turn()
        self.assertFalse(snapshots[-2].result.finished())
        self.assertEqual(snapshots[-2].consecutive_passes, 0)
        self.assertEqual(snapshots[0].remaining_stones(engine.Cell.Black), 41)

    def test_expected_checks_detect_corrupted_board_owner_stock_and_turn(self):
        with REPLAY.open(encoding="utf-8") as stream:
            original = json.load(stream)
        corruptions = [
            ("board", ["x........"] + ["........."] * 8),
            ("ownership", ["B........"] + ["........."] * 8),
            ("remaining_stones", {"black": 40, "white": 41}),
            ("to_play", "white"),
            ("consecutive_passes", 1),
            ("score", {"black": 1, "white": 0}),
        ]
        for key, value in corruptions:
            with self.subTest(key=key):
                changed = deepcopy(original)
                changed["checkpoints"][0]["expected"][key] = value
                with self.assertRaises(ValueError):
                    load_position(changed)

    def test_strict_schema_rejects_bad_coordinates_unknown_keys_and_invalid_histories(self):
        valid = {"format_version": 1, "id": "empty", "position": {"history": []}}
        variants = []
        for move in ([0, 1], [10, 1], [True, 1], [1.0, 2], [1], "p", None):
            changed = deepcopy(valid)
            changed["position"]["history"] = [move]
            variants.append(changed)
        changed = deepcopy(valid)
        changed["format_version"] = True
        variants.append(changed)
        changed = deepcopy(valid)
        changed["policy"] = [1.0] * 82
        variants.append(changed)
        variants += [
            {**valid, "position": {}},
            {**valid, "position": {"history": [], "to_play": "white"}},
            {**valid, "position": {"history": [[5, 5]]}},
            {**valid, "position": {"history": [[1, 1], [1, 1]]}},
            {**valid, "position": {"history": ["pass", "pass", [1, 1]]}},
            {**valid, "position": {"board": ["oxo......", "ooo......", ".o......."]
                                   + ["........."] * 6, "to_play": "black"}},
            {**valid, "checkpoints": [{"ply": 0, "expected": {}}, {"ply": 0, "expected": {}}]},
        ]
        for changed in variants:
            with self.subTest(spec=changed):
                with self.assertRaises(ValueError):
                    load_position(changed)

    def test_casebook_rejects_duplicate_ids_and_future_version(self):
        case = curriculum_positions()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.json"
            for book in ({"format_version": 1, "positions": [case, case]},
                         {"format_version": 2, "positions": [case]}):
                path.write_text(json.dumps(book), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_casebook(path)


if __name__ == "__main__":
    unittest.main()
