"""Completed match records, color pairing, observer isolation and local Elo."""

from dataclasses import asdict
from threading import Barrier, Event, Lock, get_ident
import json
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.encoding import action_to_move
from kingdom_ai.evaluation import evaluate_models
from kingdom_ai.match import MatchCancelled, MatchOptions, play_match, series_ratings
from kingdom_ai.model import PolicyValueNet


def rendered_board(state):
    """Render from the engine independently of the match display helper."""
    board = state.board
    rows = []
    for row in range(9):
        points = []
        for col in range(9):
            point = engine.Position(row, col)
            cell = board.at(point)
            if cell == engine.Cell.Black:
                points.append("x")
            elif cell == engine.Cell.White:
                points.append("o")
            elif cell == engine.Cell.Neutral:
                points.append("#")
            else:
                owner = state.territory_owner(point)
                points.append("B" if owner == engine.Cell.Black else
                              "W" if owner == engine.Cell.White else ".")
        rows.append("".join(points))
    return rows


class PassSearcher:
    """Use actual engine two-pass endings when inspecting protocol settings."""
    def __init__(self, model, options):
        self.model = model
        self.options = options

    def search(self, state):
        move = engine.Move.pass_turn()
        return SimpleNamespace(
            best_move=move, simulations=self.options.simulations,
            moves=[SimpleNamespace(move=move, visits=self.options.simulations)],
        )


class ModelMatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(791)
            self.model_a = PolicyValueNet(channels=4, residual_blocks=0)
            self.model_b = PolicyValueNet(channels=4, residual_blocks=0)

    def prepare_caller_state(self):
        self.model_a.train()
        self.model_a.policy_head.eval()
        self.model_b.eval()
        self.model_b.value_head.train()
        for model in (self.model_a, self.model_b):
            for index, parameter in enumerate(model.parameters()):
                parameter.grad = None if index % 2 else torch.full_like(parameter, .25)
        return {
            "modes": {module: module.training
                      for model in (self.model_a, self.model_b) for module in model.modules()},
            "weights": [{name: value.clone() for name, value in model.state_dict().items()}
                        for model in (self.model_a, self.model_b)],
            "gradients": [[None if parameter.grad is None else parameter.grad.clone()
                           for parameter in model.parameters()]
                          for model in (self.model_a, self.model_b)],
            "torch_rng": torch.random.get_rng_state().clone(),
            "python_rng": random.getstate(),
            "threads": torch.get_num_threads(),
        }

    def assert_caller_state(self, before):
        self.assertEqual({module: module.training for module in before["modes"]},
                         before["modes"])
        self.assertEqual(random.getstate(), before["python_rng"])
        self.assertEqual(torch.get_num_threads(), before["threads"])
        torch.testing.assert_close(torch.random.get_rng_state(), before["torch_rng"],
                                   rtol=0, atol=0)
        for model, weights, gradients in zip(
                (self.model_a, self.model_b), before["weights"], before["gradients"]):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
            for parameter, expected in zip(model.parameters(), gradients):
                if expected is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

    def assert_replay(self, record):
        state = engine.State()
        for action in record["actions"]:
            self.assertFalse(state.result.finished())
            self.assertTrue(state.play(action_to_move(action)).accepted())
        self.assertTrue(state.result.finished())
        self.assertEqual(state.result.winner.name, record["winner_color"])
        self.assertEqual(state.result.reason.name, record["reason"])
        self.assertEqual(record["plies"], len(record["actions"]))
        self.assertEqual(record["winner"], "a" if record["winner_color"] ==
                         record["model_a_color"] else "b")
        score = state.score()
        self.assertEqual(record["territory"], {"black": score.black, "white": score.white})
        self.assertEqual(record["board"], rendered_board(state))
        self.assertEqual(record["board"][4][4], "#")

    def test_identical_models_pair_colors_and_records_replay_to_real_results(self):
        self.model_b.load_state_dict(self.model_a.state_dict())
        result = play_match(self.model_a, self.model_b, MatchOptions(
            games=4, simulations=2, seed=271, opening_moves=3,
        ))
        self.assertEqual((result["games"], result["wins_a"], result["wins_b"]), (4, 2, 2))
        self.assertEqual(result["win_rate_a"], .5)
        self.assertEqual(result["wins_a_as_black"] + result["wins_a_as_white"], 2)
        self.assertEqual(sum(result["endings"].values()), 4)
        self.assertEqual(result["total_plies"], sum(r["plies"] for r in result["records"]))
        for index, record in enumerate(result["records"], 1):
            with self.subTest(game=index):
                self.assertEqual(record["index"], index)
                self.assertEqual(record["pair_index"], (index + 1) // 2)
                self.assert_replay(record)
        for black, white in zip(result["records"][::2], result["records"][1::2]):
            self.assertEqual((black["model_a_color"], white["model_a_color"]),
                             ("Black", "White"))
            self.assertEqual(black["seed"], white["seed"])
            self.assertEqual(black["actions"], white["actions"])
            self.assertNotEqual(black["winner"], white["winner"])
        json.dumps(result, allow_nan=False)

    def test_summary_matches_existing_evaluation_with_different_models(self):
        options = MatchOptions(games=2, simulations=2, seed=418, opening_moves=3,
                               tactical_checks=True)
        expected = evaluate_models(self.model_a, self.model_b, **asdict(options))
        actual = play_match(self.model_a, self.model_b, options)
        self.assertEqual(actual["games"], expected.games)
        self.assertEqual(actual["wins_a"], expected.wins)
        self.assertEqual(actual["wins_b"], expected.losses)
        self.assertEqual(actual["win_rate_a"], expected.win_rate)
        self.assertEqual(actual["wins_a_as_black"], expected.wins_as_black)
        self.assertEqual(actual["wins_a_as_white"], expected.wins_as_white)
        self.assertEqual(actual["endings"], expected.endings)
        self.assertEqual(actual["total_plies"], expected.total_plies)

    def test_color_schedule_seed_wrap_and_equal_search_budgets(self):
        constructions = []
        searches = []

        class RecordingSearcher(PassSearcher):
            def __init__(searcher, model, options):
                super().__init__(model, options)
                constructions.append((model, options))

            def search(searcher, state):
                searches.append((searcher.model, state.to_play))
                return super().search(state)

        with patch("kingdom_ai.evaluation.PUCT", RecordingSearcher):
            result = play_match(self.model_a, self.model_b, MatchOptions(
                games=4, simulations=13, c_puct=2., seed=2**64 - 1,
                opening_moves=1, opening_temperature=.5,
            ))
        self.assertEqual([r["seed"] for r in result["records"]], [2**64 - 1] * 2 + [0] * 2)
        self.assertEqual(searches, [
            (self.model_a, engine.Cell.Black), (self.model_b, engine.Cell.White),
            (self.model_b, engine.Cell.Black), (self.model_a, engine.Cell.White),
        ] * 2)
        for _, options in constructions:
            self.assertEqual(options.simulations, 13)
            self.assertEqual(options.c_puct, 2.)
            self.assertEqual(options.time_limit_ms, 0)
            self.assertEqual(options.dirichlet_epsilon, 0)
        self.assertEqual(result["endings"], {"TwoPasses": 4})
        self.assertEqual(result["wins_a_as_black"], 0)
        self.assertEqual(result["wins_a_as_white"], 2)

    def test_callbacks_report_real_moves_and_cannot_mutate_match_records(self):
        options = MatchOptions(games=2, simulations=2, seed=519, opening_moves=3)
        expected = play_match(self.model_a, self.model_b, options)
        replays = {}
        completed = []

        def on_move(index, actor, move, state):
            replay = replays.setdefault(index, engine.State())
            self.assertEqual(actor, replay.to_play)
            self.assertTrue(replay.play(move).accepted())
            self.assertEqual(rendered_board(state), rendered_board(replay))
            self.assertEqual(state.consecutive_passes, replay.consecutive_passes)
            self.assertEqual(state.result.reason, replay.result.reason)
            if not state.result.finished():
                self.assertTrue(state.pass_turn().accepted())

        def on_game(record):
            completed.append(record["index"])
            self.assertTrue(replays[record["index"]].result.finished())
            self.assert_replay(record)
            record["actions"].clear()
            record["board"][0] = "........."
            record["territory"]["black"] = -99
            record["winner"] = "changed"

        actual = play_match(self.model_a, self.model_b, options,
                            on_game=on_game, on_move=on_move)
        self.assertEqual(completed, [1, 2])
        self.assertEqual(actual, expected)

    def test_repeatable_match_preserves_weights_gradients_modes_and_global_rng(self):
        before = self.prepare_caller_state()

        def consume_rng(record):
            torch.rand(3)
            random.random()

        options = MatchOptions(games=2, simulations=2, seed=317, opening_moves=3)
        first = play_match(self.model_a, self.model_b, options, on_game=consume_rng)
        self.assert_caller_state(before)
        second = play_match(self.model_a, self.model_b, options, on_game=consume_rng)
        self.assertEqual(first, second)
        self.assert_caller_state(before)

    def test_callback_failures_restore_modes_and_global_rng(self):
        for callback_name in ("on_game", "on_move"):
            with self.subTest(callback=callback_name):
                before = self.prepare_caller_state()

                def fail(*args):
                    torch.rand(3)
                    random.random()
                    self.model_a.eval()
                    self.model_b.train()
                    raise LookupError("observer failed")

                with patch("kingdom_ai.evaluation.PUCT", PassSearcher), \
                        self.assertRaisesRegex(LookupError, "observer failed"):
                    play_match(self.model_a, self.model_b,
                               MatchOptions(games=2, simulations=1),
                               **{callback_name: fail})
                self.assert_caller_state(before)

    def test_inference_failure_restores_caller_state(self):
        before = self.prepare_caller_state()

        def fail(*args):
            torch.rand(3)
            random.random()
            raise RuntimeError("network failed")

        with patch.object(PolicyValueNet, "forward", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "network failed"):
            play_match(self.model_a, self.model_b, MatchOptions(games=2, simulations=1))
        self.assert_caller_state(before)

    def test_invalid_options_models_and_callbacks_fail_before_search(self):
        invalid = {
            "games": (0, 1, 3, -2, True, 2.0),
            "simulations": (0, -1, True, 1.0),
            "c_puct": (0, -1, float("nan"), float("inf"), True, "1.5"),
            "seed": (-1, 2**64, True, 0.0),
            "opening_moves": (-1, True, 1.0),
            "opening_temperature": (-1, float("nan"), float("inf"), True, "1"),
            "tactical_checks": (None, 0, 1, "true"),
            "workers": (0, -1, True, 1.0),
            "backend": (None, "unknown", 1),
            "leaf_batch_size": (0, True, 2.0),
            "reuse_tree": (None, 0, "true"),
            "inference_wait_ms": (-1, True, float("nan")),
        }
        with patch("kingdom_ai.evaluation.PUCT") as searcher:
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(option=name, value=value), \
                            self.assertRaises((TypeError, ValueError)):
                        MatchOptions(**{name: value})
            for models in ((None, self.model_b), (self.model_a, object())):
                with self.assertRaises(TypeError):
                    play_match(*models)
            with self.assertRaises(TypeError):
                play_match(self.model_a, self.model_b, {})
            for name in ("on_game", "on_move", "on_game_start", "should_stop"):
                with self.subTest(callback=name), self.assertRaises(TypeError):
                    play_match(self.model_a, self.model_b, **{name: 3})
            searcher.assert_not_called()

    def test_parallel_backends_replay_and_restore_caller_state(self):
        for backend in ("legacy", "batched_cpp"):
            with self.subTest(backend=backend):
                before = self.prepare_caller_state()
                completed, diagnostics = [], {}
                result = play_match(self.model_a, self.model_b, MatchOptions(
                    games=4, simulations=3, workers=3, backend=backend, leaf_batch_size=2,
                ), on_game=lambda record: completed.append(record["index"]), diagnostics=diagnostics)
                self.assertEqual(sorted(completed), [1, 2, 3, 4])
                self.assertEqual([row["index"] for row in result["records"]], [1, 2, 3, 4])
                for record in result["records"]:
                    self.assert_replay(record)
                self.assertEqual(diagnostics["workers"], 3)
                self.assertGreater(diagnostics["model_a"]["inference_batches"], 0)
                self.assert_caller_state(before)

    def test_1000_game_schedule_is_bounded_concurrent_and_drains_completed_results(self):
        # All three games enter together. #1 waits until a later game has been
        # saved, proving completion callbacks do not wait for index order.
        barrier, finish, release, stop = Barrier(3, timeout=10), Barrier(2, timeout=10), Event(), Event()
        lock = Lock()
        started, completed, worker_ids = [], [], set()
        caller = get_ident()

        def game(a, b, color, *, seed, on_move, **kwargs):
            with lock:
                started.append((color, seed))
                worker_ids.add(get_ident())
            barrier.wait()
            first = color == engine.Cell.Black and seed == 42
            if first:
                self.assertTrue(release.wait(10))
            state = engine.State()
            for ply in range(2):
                actor, move = state.to_play, engine.Move.pass_turn()
                state.play(move)
                on_move(actor, move, state.copy())
                if ply == 0 and not first:
                    finish.wait()
            return state.result.winner, state.result.reason, 2

        def saved(record):
            self.assertEqual(get_ident(), caller)
            completed.append(record["index"])
            if len(completed) == 1:
                stop.set()
                release.set()

        before = self.prepare_caller_state()
        with patch("kingdom_ai.match._play_batched_game", side_effect=game), \
                self.assertRaises(MatchCancelled):
            play_match(self.model_a, self.model_b, MatchOptions(games=1000, workers=3),
                       on_game=saved, should_stop=stop.is_set)
        self.assertEqual(len(started), 3)
        self.assertEqual(len(worker_ids), 3)
        self.assertEqual(set(completed), {2, 3})
        self.assert_caller_state(before)

    def test_parallel_service_and_observer_failures_release_workers(self):
        before = self.prepare_caller_state()
        with patch.object(PolicyValueNet, "forward", side_effect=RuntimeError("network failed")), \
                self.assertRaisesRegex(RuntimeError, "network failed"):
            play_match(self.model_a, self.model_b, MatchOptions(
                games=1000, simulations=3, workers=4, backend="batched_cpp"))
        self.assert_caller_state(before)
        for name in ("on_move", "on_game"):
            with self.subTest(callback=name), self.assertRaisesRegex(LookupError, "observer failed"):
                def fail(*args):
                    raise LookupError("observer failed")
                play_match(self.model_a, self.model_b, MatchOptions(
                    games=1000, simulations=1, workers=4, backend="batched_cpp"), **{name: fail})
            self.assert_caller_state(before)

    def test_series_ratings_math_zero_sum_and_series_weight(self):
        win = series_ratings(1500., 1500., 2, 2)
        self.assertEqual(win["before"], {"a": 1500., "b": 1500.})
        self.assertEqual(win["after"], {"a": 1516., "b": 1484.})
        self.assertEqual(win["delta_a"], 16.)
        self.assertEqual(series_ratings(1500., 1500., 20, 20), win)
        tied = series_ratings(1500., 1500., 10, 20)
        self.assertEqual(tied["after"], tied["before"])
        favorite = series_ratings(1800., 1400., 1, 2)
        self.assertAlmostEqual(favorite["delta_a"], 32. * (.5 - 10. / 11.))
        self.assertAlmostEqual(sum(favorite["after"].values()), 3200.)
        extreme = series_ratings(-1e12, 1e12, 2, 2)
        self.assertEqual(extreme["delta_a"], 32.)
        self.assertTrue(all(math.isfinite(value) for value in extreme["after"].values()))
        json.dumps(extreme, allow_nan=False)

    def test_invalid_rating_inputs_are_rejected(self):
        invalid = {
            "rating_a": (True, None, "1500", float("nan"), float("inf")),
            "rating_b": (False, None, "1500", float("nan"), -float("inf")),
            "wins_a": (-1, 3, True, 1.0),
            "games": (0, 1, 3, True, 2.0),
            "k": (0, -1, True, "32", float("nan"), float("inf")),
        }
        for name, values in invalid.items():
            for value in values:
                kwargs = dict(rating_a=1500., rating_b=1500., wins_a=1, games=2, k=32.)
                kwargs[name] = value
                with self.subTest(option=name, value=value), \
                        self.assertRaises((TypeError, ValueError)):
                    series_ratings(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
