"""Losing-side mining and bounded, policy-only action certificates."""

from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

from kingdom_ai.capture_mining import CaptureDefenseMiner, collect_capture_defense_samples
from kingdom_ai.capture_defense import CaptureDefenseReplay
from kingdom_ai.encoding import action_to_move, encode_state
from kingdom_ai.tactical_positions import load_position, user_game_two_positions
from kingdom_ai.tactical_training import collect_certified_samples
from kingdom_ai.training import (GameData, ProofTrainingBatch, TrainingSample, make_batch,
                                 train_mixed_step)


def _game(history):
    state = engine.State()
    actions = tuple(81 if item == "pass" else (item[0] - 1) * 9 + item[1] - 1
                    for item in history)
    for action in actions:
        if not state.play(action_to_move(action)).accepted():
            raise AssertionError("Invalid fixture")
    if not state.result.finished():
        raise AssertionError("Fixture must terminate")
    # Feature values are deliberately absent: mining must use histories only.
    return GameData([None] * len(actions), state.result.winner, state.result.reason, actions)


def _case(history, action, actor="White", case_id="bad_corner"):
    return {"format_version": 1, "id": case_id, "family_id": "fixture_game",
            "motif": "capture_defense_fixture",
            "position": {"history": history, "neutral": [5, 5]},
            "source": {"bad_action": action, "loser": actor}}


def _unknown(nodes=1, exhausted=False):
    return SimpleNamespace(outcome="UNKNOWN", nodes=nodes, proof_depth=0,
                           elapsed_ms=0.0, budget_exhausted=exhausted,
                           principal_variation=[])


class CaptureMiningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = _game(user_game_two_positions()[-1]["position"]["history"] + [[6, 2]])
        cls.corner = _game([[1, 2], [1, 1], [2, 1]])
        cls.passes = _game(["pass", "pass"])
        cls.suicide_history = [[1, 2], [1, 1], [8, 9], [1, 3], [8, 8], [2, 1],
                               [8, 7], [2, 3], [7, 9], [3, 2], [7, 8], [9, 9], [2, 2]]
        cls.suicide = _game(cls.suicide_history)

    def miner(self, count=8, *, iteration=0, lookback=30, seed=103):
        return CaptureDefenseMiner(count, generator=torch.Generator().manual_seed(seed),
                                   iteration=iteration, lookback_plies=lookback)

    def collect(self, cases=None, **kwargs):
        options = kwargs.pop("options", engine.TacticalSolverOptions(
            max_depth=30, max_nodes=100000, time_limit_ms=1000))
        return collect_capture_defense_samples(
            [_case([[1, 2]], 0)] if cases is None else cases, options, **kwargs)

    def test_mining_replays_losing_colour_pre_move_complete_histories(self):
        miner = self.miner(iteration=4)
        with patch("kingdom_ai.capture_mining.engine.State",
                   side_effect=AssertionError("No eager replay or tensor decoding")):
            miner.add_game(self.capture, 17)
        cases = miner.cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual({case["source"]["selection_slot"] for case in cases},
                         {"pre_trap", "earlier_defense", "early_error", "last_decision"})
        for case in cases:
            ply = case["source"]["ply"]
            state = load_position(case)
            self.assertEqual(state.to_play, engine.opponent(self.capture.winner))
            self.assertEqual(case["source"]["bad_action"], self.capture.action_history[ply])
            replay = engine.State()
            for action in self.capture.action_history[:ply]:
                self.assertTrue(replay.play(action_to_move(action)).accepted())
            self.assertEqual(state.board.cells, replay.board.cells)
            self.assertEqual(state.ownership, replay.ownership)
            self.assertEqual(state.consecutive_passes, replay.consecutive_passes)
            self.assertEqual(state.remaining_stones(state.to_play),
                             replay.remaining_stones(replay.to_play))
            self.assertFalse(state.result.finished())
            self.assertNotIn("value", case)
            self.assertNotIn("policy", case)

    def test_lookback_is_total_plies_and_pass_games_are_not_defense_examples(self):
        miner = self.miner(lookback=5)
        miner.add_game(self.capture, 0)
        miner.add_game(self.passes, 1)
        self.assertEqual(miner.seen_games, 2)
        self.assertEqual(miner.seen_losses, 1)
        self.assertEqual(miner.sampled_game_indices, (0,))
        for case in miner.cases():
            self.assertGreaterEqual(case["source"]["ply"], len(self.capture.action_history) - 5)
            self.assertLessEqual(case["source"]["plies_to_end"], 5)

    def test_reservoir_uniformity_bounds_rotation_and_cached_reproducibility(self):
        miner = self.miner(32, iteration=7)
        reference = list(range(64))
        rng = torch.Generator().manual_seed(103)
        for index in range(200):
            miner.add_game(self.capture, index)
            if index >= 64:
                slot = int(torch.randint(index + 1, (), generator=rng).item())
                if slot < 64:
                    reference[slot] = index
        self.assertEqual(miner.sampled_game_indices, tuple(reference))
        before = miner.generator.get_state().clone()
        cases = miner.cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({row["family_id"] for row in cases}), 32)
        self.assertEqual({row["source"]["selection_slot"] for row in cases},
                         {"pre_trap", "earlier_defense", "early_error", "last_decision"})
        self.assertLessEqual(miner.candidate_count, miner.candidate_capacity)
        cases[0]["position"]["history"].clear()
        self.assertNotEqual(cases, miner.cases())
        self.assertTrue(torch.equal(before, miner.generator.get_state()))

    def test_miner_rejects_bad_options_histories_and_terminal_claims(self):
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.miner(count)
        for lookback in (0, 31, True):
            with self.assertRaises(ValueError):
                self.miner(lookback=lookback)
        with self.assertRaises(ValueError):
            CaptureDefenseMiner(1, generator=object())
        with self.assertRaises(TypeError):
            self.miner().add_game(object(), 0)
        for history in (None, (), [0, 1], (True, 1), (82, 1), (1,) * 165):
            with self.assertRaises(ValueError):
                self.miner().add_game(GameData([None] * (len(history) if history else 0),
                    engine.Cell.White, engine.EndReason.Capture, history), 0)
        for history, winner, reason in (((40, 0), engine.Cell.White, engine.EndReason.Capture),
                                       ((81, 81, 0), engine.Cell.White, engine.EndReason.Capture),
                                       ((81, 81), engine.Cell.Black, engine.EndReason.Capture)):
            miner = self.miner()
            miner.add_game(GameData([None] * len(history), winner, reason, history), 0)
            with self.assertRaises(ValueError):
                miner.cases()

    def test_real_successor_win_certifies_only_played_action_not_parent_value(self):
        samples, audit = self.collect()
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.proof["outcome"], "ACTION_LOSS")
        self.assertEqual(sample.proof["successor_outcome"], "WIN")
        self.assertEqual(sample.proof["proof_depth"], 2)
        self.assertEqual(sample.proof["losing_actions"], [0])
        self.assertEqual(sample.losing_mask.nonzero().flatten().tolist(), [0])
        self.assertTrue(bool((sample.legal_mask & ~sample.losing_mask).any()))
        self.assertEqual(sample.features.shape, (10, 9, 9))
        self.assertEqual(sample.features.device.type, "cpu")
        self.assertNotIn("value", sample.proof)
        self.assertEqual(audit["certified_samples"], 1)

    def test_immediate_suicide_is_one_ply_but_own_capture_win_is_not_bad(self):
        case = _case(self.suicide_history[:-1], 10, "Black", "self_suicide")
        with patch("kingdom_ai.capture_mining.engine.solve_tactics",
                   side_effect=AssertionError("Terminal moves need no solver")):
            samples, audit = self.collect([case])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].proof["proof_depth"], 1)
        self.assertEqual(samples[0].proof["terminal_reason"], "Suicide")
        own_win = _case([[1, 2], [1, 1]], 9, "Black", "own_capture")
        samples, audit = self.collect([own_win])
        self.assertEqual(samples, [])
        self.assertEqual(audit["excluded_action_win"], 1)

    def test_unavoidable_final_atari_is_excluded_before_search_or_policy_target(self):
        prefix = user_game_two_positions()[-2]["position"]["history"]
        case = _case(prefix, self.capture.action_history[len(prefix)], "Black", "too_late")
        samples, audit = self.collect([case], solver=lambda *_: self.fail("Unavoidable already"))
        self.assertEqual(samples, [])
        self.assertEqual(audit["excluded_already_forced"], 1)
        self.assertEqual(audit["records"][0]["nodes"], 0)

    def test_unknown_ladder_shares_one_node_budget_and_never_teaches_unknown(self):
        calls = []
        def solver(_, options):
            calls.append((options.max_depth, options.max_nodes))
            nodes = min(2, options.max_nodes)
            return _unknown(nodes, nodes == options.max_nodes)
        samples, audit = self.collect(solver=solver, options=engine.TacticalSolverOptions(
            max_depth=30, max_nodes=5, time_limit_ms=1000))
        self.assertEqual(samples, [])
        self.assertEqual(calls, [(1, 5), (3, 3), (5, 1)])
        self.assertEqual(audit["records"][0]["nodes"], 5)
        self.assertTrue(audit["records"][0]["budget_exhausted"])
        self.assertEqual(audit["excluded_unknown"], 1)

    def test_every_stage_shares_case_time_and_global_remaining_deadline(self):
        # Sub-millisecond/rounding remainders are discarded, never rounded up.
        for case_ms, seconds, expected in ((10, 0.1, [10, 7, 4]),
                                          (100, 0.008, [8, 5, 2])):
            clock, times = [0.0], []
            def solver(_, options):
                times.append(options.time_limit_ms)
                clock[0] += min(3, options.time_limit_ms) / 1000
                return _unknown()
            with self.subTest(case_ms=case_ms), patch(
                    "kingdom_ai.capture_mining.perf_counter", side_effect=lambda: clock[0]):
                samples, audit = self.collect(solver=solver, generation_seconds=seconds,
                    options=engine.TacticalSolverOptions(max_depth=30, max_nodes=100,
                                                         time_limit_ms=case_ms))
            self.assertEqual(samples, [])
            self.assertEqual(times, expected)
            self.assertLessEqual(audit["generation_seconds"], min(seconds, case_ms / 1000))

    def test_depth_ladder_counts_selected_action_and_stops_at_successor_29(self):
        calls = []
        def solver(_, options):
            calls.append(options.max_depth)
            return _unknown()
        samples, audit = self.collect(solver=solver)
        self.assertEqual(calls, [1, 3, 5, 9, 15, 21, 29])
        self.assertEqual(samples, [])
        calls.clear()
        self.collect(solver=solver, options=engine.TacticalSolverOptions(
            max_depth=5, max_nodes=100, time_limit_ms=1000))
        self.assertEqual(calls, [1, 3, 4])

    def test_prefix_actor_action_and_proof_metadata_are_fail_closed(self):
        for change in ({"loser": "Black"}, {"loser": "red"}, {"bad_action": True},
                       {"bad_action": 40}, {"bad_action": 1}):
            case = _case([[1, 2]], 0)
            case["source"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.collect([case])
        case = _case([[1, 2], [1, 1], [2, 1]], 0, "White", "terminal")
        with self.assertRaises(ValueError):
            self.collect([case])
        case = _case([[1, 2]], 0)
        case["position"] = {"board": ["........."] * 9, "to_play": "white"}
        with self.assertRaises(ValueError):
            self.collect([case])
        bad = _unknown(nodes=100001)
        with self.assertRaises(ValueError):
            self.collect(solver=lambda *_: bad)
        bad = SimpleNamespace(outcome="WIN", nodes=1, proof_depth=1, elapsed_ms=0.0,
                              budget_exhausted=False, principal_variation=[engine.Move.pass_turn()])
        with self.assertRaises(ValueError):
            self.collect(solver=lambda *_: bad)

    def test_collection_options_are_strict_and_max_cases_is_respected(self):
        for kwargs in ({"max_cases": True}, {"generation_seconds": 0},
                       {"generation_seconds": float("nan")}, {"bound_remaining_time": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.collect(**kwargs)
        with self.assertRaises(TypeError):
            self.collect(options=object())
        for depth in (0, 31):
            with self.assertRaises(ValueError):
                self.collect(options=engine.TacticalSolverOptions(max_depth=depth))
        cases = [_case([[1, 2]], 0, case_id=f"case_{i}") for i in range(3)]
        samples, audit = self.collect(cases, max_cases=1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(audit["cases_solved"], 1)

    def test_actual_four_ply_and_terminal_certificates_replay_and_mixed_training(self):
        history = user_game_two_positions()[-1]["position"]["history"] + [[6, 2]]
        point = history[22]
        trap = _case(history[:22], (point[0] - 1) * 9 + point[1] - 1,
                     "Black", "four_ply_trap")
        suicide = _case(self.suicide_history[:-1], 10, "Black", "terminal_suicide")
        rows, audit = self.collect([trap, suicide], options=engine.TacticalSolverOptions(
            max_depth=5, max_nodes=100000, time_limit_ms=1000), generation_seconds=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.proof["proof_depth"] for row in rows], [4, 1])
        self.assertEqual([call["max_depth"] for call in rows[0].proof["calls"]], [1, 3])
        for row in rows:
            self.assertEqual(row.proof["successor_outcome"], "WIN")
            self.assertEqual(row.proof["target_kind"], "negative_policy_only")
        replay = CaptureDefenseReplay(4)
        replay.extend(rows)
        restored = CaptureDefenseReplay.from_state_dict(replay.state_dict())
        defense = restored.sample(2, generator=torch.Generator().manual_seed(0))

        encoded = encode_state(engine.State())
        policy = torch.zeros(82)
        policy[0] = 1
        normal = make_batch([TrainingSample(encoded.features, encoded.legal_mask,
                                            policy, 1.0, encoded.to_play)])
        teacher_rows, _ = collect_certified_samples(
            [_case([[1, 2], [1, 1]], 9, "Black", "native_teacher_win")],
            engine.TacticalSolverOptions(max_depth=1, max_nodes=1000, time_limit_ms=100),
            generation_seconds=1)
        teacher = make_batch([row.sample for row in teacher_rows])
        proof = ProofTrainingBatch(teacher.features, teacher.legal_mask, teacher.policy,
                                   teacher.value, teacher.value == 1)
        class DirectModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(82))
                self.value = torch.nn.Parameter(torch.tensor(0.0))
            def forward(self, features):
                return (self.logits.expand(features.shape[0], -1),
                        self.value.expand(features.shape[0]))
        model = DirectModel()
        metrics = train_mixed_step(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                   normal, proof, defense_batch=defense)
        self.assertEqual(metrics["normal_rows"], 1)
        self.assertEqual(metrics["teacher_rows"], 1)
        self.assertEqual(metrics["defense_rows"], 2)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["defense_policy_loss"], 0)


if __name__ == "__main__":
    unittest.main()
