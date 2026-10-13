"""Cheap capture-risk gating must not manufacture tactical training targets."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.capture_mining import (
    CaptureDefenseMiner, _capture_risk, collect_capture_defense_samples,
)
from kingdom_ai.encoding import action_to_move
from kingdom_ai.tactical_positions import load_position, user_game_two_positions
from kingdom_ai.training import GameData


def _case(history, action, actor="White", case_id="risk_fixture"):
    return {"format_version": 1, "id": case_id, "family_id": "risk_fixture_game",
            "position": {"history": deepcopy(history), "neutral": [5, 5]},
            "source": {"bad_action": action, "loser": actor}}


def _transition(history, action):
    before = load_position(_case(history, action))
    after = before.copy()
    if not after.play(action_to_move(action)).accepted():
        raise AssertionError("Fixture action is not legal")
    return before, after


def _game(history):
    state = engine.State()
    actions = tuple(81 if point == "pass" else (point[0] - 1) * 9 + point[1] - 1
                    for point in history)
    for action in actions:
        if not state.play(action_to_move(action)).accepted():
            raise AssertionError("Invalid completed-game fixture")
    if not state.result.finished():
        raise AssertionError("Fixture must be terminal")
    return GameData([None] * len(actions), state.result.winner, state.result.reason, actions)


def _snapshot(state):
    return (state.board.cells, state.ownership, state.to_play, state.consecutive_passes,
            state.remaining_stones(engine.Cell.Black),
            state.remaining_stones(engine.Cell.White), state.result)


def _unknown(_state, _options):
    return SimpleNamespace(outcome="UNKNOWN", nodes=1, proof_depth=0,
                           elapsed_ms=0.0, budget_exhausted=False,
                           principal_variation=[])


class CaptureRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = user_game_two_positions()[-1]["position"]["history"] + [[6, 2]]
        cls.capture = _game(cls.history)

    def collect(self, cases, **kwargs):
        return collect_capture_defense_samples(cases, engine.TacticalSolverOptions(
            max_depth=30, max_nodes=1000, time_limit_ms=1000), **kwargs)

    def miner(self, count=8, lookback=30, iteration=0):
        return CaptureDefenseMiner(count, generator=torch.Generator().manual_seed(103),
                                   lookback_plies=lookback, iteration=iteration)

    def test_quiet_center_and_opponent_low_liberty_group_do_not_trigger(self):
        for history in ([[3, 3]], [[1, 1]]):
            with self.subTest(history=history):
                before, after = _transition(history, 60)  # White at (7, 7).
                self.assertEqual(before.to_play, engine.Cell.White)
                self.assertEqual(len(after.board.liberties(engine.Position(6, 6))), 4)
                self.assertIsNone(_capture_risk(before, after))

    def test_candidate_creates_risk_from_quiet_parent_at_edge_or_corner(self):
        for action, liberties in ((0, 2), (4, 3), (41, 3)):
            with self.subTest(action=action):
                before, after = _transition([[3, 3]], action)
                risk = _capture_risk(before, after)
                self.assertIsNotNone(risk)
                self.assertIsNone(risk["own_min_before"])
                self.assertEqual(risk["own_min_after"], liberties)
                self.assertEqual(risk["own_low_chains_before"], 0)
                self.assertEqual(risk["own_low_chains_after"], 1)
                self.assertIn("played_action_leaves_low_liberty_chain", risk["reasons"])
                self.assertFalse(risk["is_proof"])

    def test_single_stone_with_three_liberties_triggers_even_on_distant_move(self):
        before, after = _transition([[3, 3], [1, 5], [7, 3]], 60)
        risk = _capture_risk(before, after)
        self.assertEqual(len(before.board.group_at(engine.Position(0, 4))), 1)
        self.assertEqual(risk["threshold"], 3)
        self.assertEqual(risk["own_min_before"], 3)
        self.assertEqual(risk["own_low_chains_before"], 1)
        self.assertEqual(risk["own_low_chains_after"], 1)
        self.assertIn("own_chain_at_most_three_liberties", risk["reasons"])
        self.assertFalse(risk["is_proof"])

    def test_neutral_stone_is_wall_not_own_chain_and_caller_is_unchanged(self):
        before, after = _transition([], 20)  # Black at (3, 3), four liberties.
        snapshots = (_snapshot(before), _snapshot(after))
        self.assertIsNone(_capture_risk(before, after))
        self.assertEqual((_snapshot(before), _snapshot(after)), snapshots)
        self.assertEqual(before.board.cells[40], engine.Cell.Neutral)
        self.assertEqual(after.board.cells[40], engine.Cell.Neutral)
        before, after = _transition([], 41)  # Black touches the fixed neutral wall.
        risk = _capture_risk(before, after)
        self.assertEqual(risk["own_low_chains_after"], 1)
        self.assertEqual(risk["own_min_after"], 3)

    def test_house_ownership_and_legal_moves_are_not_reclassified_by_heuristic(self):
        # A completed own house is an engine-owned empty point, not a stone.
        history = [[2, 3], [7, 3], [3, 2], [7, 7],
                   [3, 4], [8, 5], [4, 3], [8, 8]]
        before, after = _transition(history, 24)  # Black at (3, 7), away from all walls.
        self.assertEqual(before.territory_owner(engine.Position(2, 2)), engine.Cell.Black)
        self.assertFalse(before.is_legal(engine.Move.place(2, 2)))
        snapshots = (_snapshot(before), _snapshot(after))
        self.assertIsNone(_capture_risk(before, after))
        self.assertEqual((_snapshot(before), _snapshot(after)), snapshots)
        self.assertEqual(after.territory_owner(engine.Position(2, 2)), engine.Cell.Black)

    def test_terminal_actor_loss_qualifies_but_immediate_capture_win_does_not(self):
        before, after = _transition([[1, 2], [1, 1]], 9)
        self.assertEqual(after.result.reason, engine.EndReason.Capture)
        self.assertEqual(after.result.winner, before.to_play)
        self.assertIsNone(_capture_risk(before, after))
        # There are no endangered own stones here; terminal loss is sufficient.
        before, after = _transition([[3, 3], "pass"], 81)
        self.assertEqual(after.result.reason, engine.EndReason.TwoPasses)
        self.assertNotEqual(after.result.winner, before.to_play)
        risk = _capture_risk(before, after)
        self.assertEqual(risk["own_min_before"], 4)
        self.assertIn("played_action_loses_immediately", risk["reasons"])
        self.assertFalse(risk["is_proof"])

    def test_quiet_collection_skips_even_shallow_analysis_and_forged_risk(self):
        case = _case([[3, 3]], 60)
        case["source"]["risk"] = {"is_proof": True, "threshold": 99,
                                  "reasons": ["forced_loss"]}
        original = deepcopy(case)
        with patch("kingdom_ai.capture_mining.analyze_tactics",
                   side_effect=AssertionError("Quiet positions must not enumerate replies")), patch(
                "kingdom_ai.capture_mining.engine.solve_tactics",
                side_effect=AssertionError("Quiet positions must not enter deep search")), patch(
                "kingdom_ai.capture_mining.encode_state",
                side_effect=AssertionError("Quiet positions must not prepare training tensors")):
            samples, audit = self.collect([case])
        self.assertEqual(case, original)
        self.assertEqual(samples, [])
        self.assertEqual(audit["excluded_low_risk"], 1)
        self.assertEqual(audit["solver_positions"], 0)
        self.assertEqual(audit["solver_calls"], 0)
        record = audit["records"][0]
        self.assertEqual(record["outcome"], "LOW_RISK")
        self.assertFalse(record["training_label"])
        self.assertEqual(record["nodes"], 0)
        self.assertEqual(record["calls"], [])
        self.assertNotIn("proof_depth", record)
        self.assertNotIn("target_kind", record)

    def test_recomputed_risk_not_fake_certificate_unknown_still_teaches_nothing(self):
        case = _case([[1, 2]], 0)
        case["source"]["risk"] = {"is_proof": True, "outcome": "ACTION_LOSS"}
        samples, audit = self.collect([case], solver=_unknown)
        self.assertEqual(samples, [])
        self.assertEqual(audit["excluded_unknown"], 1)
        self.assertEqual(audit["excluded_low_risk"], 0)
        self.assertEqual(audit["solver_positions"], 1)
        self.assertEqual(audit["solver_calls"], 7)
        record = audit["records"][0]
        self.assertEqual([call["max_depth"] for call in record["calls"]],
                         [1, 3, 5, 9, 15, 21, 29])
        self.assertFalse(record["risk"]["is_proof"])
        self.assertEqual(record["risk"]["heuristic"], "own_liberties_v1")
        self.assertFalse(record["training_label"])
        self.assertNotIn("value", record)

    def test_collection_counts_only_positions_and_calls_that_reach_solver(self):
        quiet = _case([[3, 3]], 60, case_id="quiet")
        risky = _case([[1, 2]], 0, case_id="risky")
        own_win = _case([[1, 2], [1, 1]], 9, "Black", "own_win")
        samples, audit = self.collect([quiet, risky, own_win])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].case_id, "risky")
        self.assertEqual(samples[0].proof["proof_depth"], 2)
        self.assertFalse(samples[0].proof["risk"]["is_proof"])
        self.assertEqual(audit["cases_solved"], 3)
        self.assertEqual(audit["excluded_low_risk"], 1)
        self.assertEqual(audit["excluded_action_win"], 1)
        self.assertEqual(audit["solver_positions"], 1)
        self.assertEqual(audit["solver_calls"], 1)
        self.assertEqual(audit["solver_positions"],
                         sum(bool(row["calls"]) for row in audit["records"]))
        self.assertEqual(audit["solver_calls"],
                         sum(len(row["calls"]) for row in audit["records"]))

    def test_miner_scans_every_decision_then_keeps_onset_and_temporal_diversity(self):
        miner = self.miner()
        miner.add_game(self.capture, 0)
        cases = miner.cases()
        self.assertEqual(miner.scanned_decisions, 13)
        self.assertEqual(miner.risky_decisions, 10)
        self.assertEqual(miner.quiet_decisions, 3)
        self.assertEqual(miner.scanned_decisions, miner.risky_decisions + miner.quiet_decisions)
        self.assertEqual({row["source"]["ply"] for row in cases}, {6, 18, 22, 24})
        self.assertEqual({row["source"]["selection_slot"] for row in cases},
                         {"pre_trap", "earlier_defense", "early_error", "last_decision"})
        onset = next(row for row in cases if row["source"]["selection_slot"] == "early_error")
        self.assertEqual(onset["source"]["risk"]["own_min_before"], 4)
        self.assertEqual(onset["source"]["risk"]["own_min_after"], 3)
        for case in cases:
            before = load_position(case)
            after = before.copy()
            self.assertTrue(after.play(action_to_move(case["source"]["bad_action"])).accepted())
            self.assertEqual(case["source"]["risk"], _capture_risk(before, after))
            self.assertFalse(case["source"]["risk"]["is_proof"])
            self.assertNotIn("value", case)
            self.assertNotIn("policy", case)

    def test_miner_cache_copies_and_counters_reset_on_new_capture_game(self):
        miner = self.miner()
        miner.add_game(self.capture, 0)
        cases = miner.cases()
        original = deepcopy(cases)
        counters = (miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions)
        rng = miner.generator.get_state().clone()
        cases[0]["source"]["risk"]["reasons"].clear()
        cases[0]["position"]["history"].clear()
        with patch("kingdom_ai.capture_mining.engine.State",
                   side_effect=AssertionError("Cached cases must not replay again")):
            self.assertEqual(miner.cases(), original)
        self.assertEqual((miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions),
                         counters)
        self.assertTrue(torch.equal(rng, miner.generator.get_state()))
        miner.add_game(self.capture, 1)
        self.assertEqual((miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions),
                         (0, 0, 0))
        self.assertEqual(len(miner.cases()), 8)
        self.assertEqual((miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions),
                         (26, 20, 6))
        miner.add_game(_game(["pass", "pass"]), 2)
        self.assertEqual((miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions),
                         (26, 20, 6))
        self.assertEqual(miner.seen_games, 3)
        self.assertEqual(miner.seen_losses, 2)

    def test_miner_short_lookback_counts_individual_plies_without_quiet_padding(self):
        miner = self.miner(lookback=5)
        miner.add_game(self.capture, 0)
        cases = miner.cases()
        self.assertEqual((miner.scanned_decisions, miner.risky_decisions, miner.quiet_decisions),
                         (2, 2, 0))
        self.assertEqual({row["source"]["ply"] for row in cases}, {22, 24})
        corner = self.miner()
        corner.add_game(_game([[1, 2], [1, 1], [2, 1]]), 1)
        only_case, = corner.cases()
        self.assertEqual(only_case["source"]["ply"], 1)
        self.assertEqual(corner.scanned_decisions, 1)
        self.assertEqual(corner.risky_decisions, 1)
        self.assertEqual(corner.quiet_decisions, 0)

    def test_family_round_robin_retains_diverse_slots_with_bounded_output(self):
        miner = self.miner(count=32, iteration=7)
        for index in range(40):
            miner.add_game(self.capture, index)
        cases = miner.cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({row["family_id"] for row in cases}), 32)
        self.assertEqual({row["source"]["selection_slot"] for row in cases},
                         {"pre_trap", "earlier_defense", "early_error", "last_decision"})
        self.assertLessEqual(miner.candidate_count, miner.candidate_capacity)
        self.assertEqual(miner.scanned_decisions, 40 * 13)
        self.assertEqual(miner.risky_decisions, 40 * 10)
        self.assertEqual(miner.quiet_decisions, 40 * 3)


if __name__ == "__main__":
    unittest.main()
