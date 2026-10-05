"""Check completed self-play remains reproducible when worker counts change."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "examples" / "python_benchmark.py"
spec = importlib.util.spec_from_file_location("parallel_benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ParallelBenchmarkTests(unittest.TestCase):
    def assert_same_games(self, first, second):
        for field in ("kind", "simulations_per_move", "games", "mean_plies",
                      "simulations", "endings", "winners", "seed_strategy", "result_checksum"):
            with self.subTest(field=field):
                self.assertEqual(first[field], second[field])

    def assert_complete(self, result):
        self.assertEqual(sum(result["endings"].values()), result["games"])
        self.assertEqual(sum(result["winners"].values()), result["games"])
        self.assertTrue(set(result["winners"]).issubset({"Black", "White"}))
        self.assertTrue(set(result["endings"]).issubset({"Capture", "Suicide", "TwoPasses"}))
        total_plies = round(result["mean_plies"] * result["games"])
        self.assertGreater(total_plies, 0)
        self.assertLessEqual(total_plies, result["games"] * (2 * benchmark.engine.CELL_COUNT + 2))
        self.assertEqual(result["simulations"], total_plies * result["simulations_per_move"])
        self.assertEqual(result["seed_strategy"], "per_game")
        self.assertGreater(result["seconds"], 0)
        self.assertAlmostEqual(result["games_per_second"], result["games"] / result["seconds"])

    def test_worker_counts_preserve_completed_games_and_search_work(self):
        baseline = benchmark.self_play(16, 8, 42, workers=1)
        self.assert_complete(baseline)
        for workers in (2, 4, 6):
            with self.subTest(workers=workers):
                parallel = benchmark.self_play(16, 8, 42, workers=workers)
                self.assert_same_games(baseline, parallel)
                self.assert_complete(parallel)
                self.assertEqual(parallel["worker_threads"], workers)
                self.assertEqual(parallel["requested_workers"], workers)

    def test_seed_wrap_remains_reproducible(self):
        # Seeds cross the unsigned 64-bit boundary after the second game.
        baseline = benchmark.self_play(8, 6, 2 ** 64 - 2, workers=1)
        parallel = benchmark.self_play(8, 6, 2 ** 64 - 2, workers=4)
        self.assert_same_games(baseline, parallel)
        self.assert_complete(parallel)

    def test_fewer_games_than_workers_preserves_exact_game_count(self):
        baseline = benchmark.self_play(8, 2, 1729, workers=1)
        parallel = benchmark.self_play(8, 2, 1729, workers=12)
        self.assert_same_games(baseline, parallel)
        self.assert_complete(parallel)
        self.assertEqual(parallel["games"], 2)
        self.assertEqual(parallel["worker_threads"], 2)
        self.assertEqual(parallel["requested_workers"], 12)

    def test_exception_from_parallel_game_is_not_counted_as_completion(self):
        original = benchmark.seeded_game

        def fail_one_game(simulations_per_move, seed):
            if seed == 43:
                raise RuntimeError("game worker failed")
            return original(simulations_per_move, seed)

        with patch.object(benchmark, "seeded_game", side_effect=fail_one_game):
            with self.assertRaisesRegex(RuntimeError, "game worker failed"):
                benchmark.self_play(8, 4, 42, workers=2)

    def test_rejected_search_move_is_not_counted_as_completion(self):
        searcher = SimpleNamespace(search=lambda state: SimpleNamespace(
            best_move=benchmark.engine.Move.place(4, 4), simulations=1))
        with self.assertRaisesRegex(RuntimeError, "accepted move"):
            benchmark.completed_game(searcher)

    def test_invalid_configuration_is_rejected_before_starting_games(self):
        invalid = ((0, 2, 42, 1), (8, 0, 42, 1), (8, 2, 42, 0),
                   (8, 2, 42, True), (8, 2, 42, 1.5),
                   (8, 2, -1, 1), (8, 2, 2 ** 64, 1), (8, 2, True, 1))
        with patch.object(benchmark, "seeded_game") as play:
            for simulations, games, seed, workers in invalid:
                with self.subTest(configuration=(simulations, games, seed, workers)):
                    with self.assertRaises(ValueError):
                        benchmark.self_play(simulations, games, seed, workers=workers)
            play.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
