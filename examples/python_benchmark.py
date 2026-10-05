"""Measure completed self-play games separately from MCTS simulations."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
import platform
from statistics import median
from time import perf_counter

import my_board_engine as engine


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def completed_game(searcher, game=None):
    game = engine.State() if game is None else game
    plies = simulations = 0
    while not game.result.finished():
        search = searcher.search(game)
        if search.best_move is None or not game.play(search.best_move).accepted():
            raise RuntimeError("Search did not produce an accepted move")
        plies += 1
        simulations += search.simulations
        if plies > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Game did not terminate within the finite move bound")
    return plies, simulations, game.result.reason.name


def seeded_game(simulations_per_move, seed):
    """Each game owns its state and RNG, regardless of worker scheduling."""
    options = engine.MCTSOptions(simulations=simulations_per_move, seed=seed)
    game = engine.State()
    turns, searches, reason = completed_game(engine.MCTS(options), game)
    return turns, searches, reason, game.result.winner.name, game.board.to_string()


def self_play(simulations_per_move, games, seed, workers=1):
    for name, value in (("simulations_per_move", simulations_per_move),
                        ("games", games), ("workers", workers)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    worker_threads = min(workers, games)
    seeded_game(simulations_per_move, seed)  # Warm up outside the timed interval.
    seeds = ((seed + index) % (2 ** 64) for index in range(games))
    play = partial(seeded_game, simulations_per_move)
    plies = simulations = 0
    endings = Counter()
    winners = Counter()
    checksum = hashlib.sha256()

    def collect(results):
        nonlocal plies, simulations
        for turns, searches, reason, winner, board in results:
            plies += turns
            simulations += searches
            endings[reason] += 1
            winners[winner] += 1
            checksum.update(json.dumps(
                [turns, searches, reason, winner, board], separators=(",", ":")
            ).encode("ascii"))
            checksum.update(b"\n")

    started = perf_counter()
    if worker_threads == 1:
        collect(map(play, seeds))
    else:
        # search() releases the GIL; independent MCTS objects can use many cores.
        # map returns results in game order, making the checksum schedule-independent.
        with ThreadPoolExecutor(max_workers=worker_threads) as executor:
            collect(executor.map(play, seeds))
    elapsed = perf_counter() - started
    return {
        "kind": "completed_self_play",
        "requested_workers": workers,
        "worker_threads": worker_threads,
        "seed_strategy": "per_game",
        "result_checksum": checksum.hexdigest(),
        "simulations_per_move": simulations_per_move,
        "games": games,
        "seconds": elapsed,
        "games_per_second": games / elapsed,
        "seconds_per_game": elapsed / games,
        "mean_plies": plies / games,
        "simulations": simulations,
        "simulations_per_second": simulations / elapsed,
        "endings": dict(endings),
        "winners": dict(winners),
    }


def opening_search(simulations, seed):
    engine.MCTS(engine.MCTSOptions(simulations=64, seed=seed)).search(engine.State())
    searcher = engine.MCTS(engine.MCTSOptions(simulations=simulations, seed=seed))
    started = perf_counter()
    result = searcher.search(engine.State())
    elapsed = perf_counter() - started
    return {
        "kind": "opening_mcts_search",
        "simulations": result.simulations,
        "seconds": elapsed,
        "simulations_per_second": result.simulations / elapsed,
        "mean_rollout_plies": result.total_rollout_plies / result.simulations,
        "nodes": result.nodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=positive_integer, nargs="+", default=[64, 1000])
    parser.add_argument("--games", type=positive_integer, default=20)
    parser.add_argument("--workers", type=positive_integer, nargs="+", default=[1],
                        help="Concurrent games; use 1 2 4 6 12 to compare CPU scaling")
    parser.add_argument("--repeats", type=positive_integer, default=1,
                        help="Measurements per configuration; report median throughput")
    parser.add_argument("--opening-simulations", type=positive_integer, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64:
        parser.error("seed must be an unsigned 64-bit integer")
    print(json.dumps({
        "kind": "environment",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "module": engine.__file__,
        "logical_cpus": os.cpu_count(),
        "requested_workers": args.workers,
        "repeats": args.repeats,
        "seed_strategy": "per_game",
        "seed": args.seed,
        "time_limit_ms": 0,
        "neural_network": False,
    }), flush=True)
    worker_counts = list(dict.fromkeys(args.workers))
    for budget in args.simulations:
        measurements = {workers: [] for workers in worker_counts}
        expected_checksum = None
        for repeat in range(args.repeats):
            # Rotate configurations so the same worker count isn't always last.
            offset = repeat % len(worker_counts)
            order = worker_counts[offset:] + worker_counts[:offset]
            for workers in order:
                result = self_play(budget, args.games, args.seed, workers)
                if expected_checksum is None:
                    expected_checksum = result["result_checksum"]
                elif result["result_checksum"] != expected_checksum:
                    raise RuntimeError("Serial and parallel game results differ")
                result["repeat"] = repeat + 1
                measurements[workers].append(result)
                print(json.dumps(result), flush=True)
        reference = median(r["games_per_second"] for r in measurements[1]) if 1 in measurements else None
        summaries = []
        for workers, results in measurements.items():
            rate = median(r["games_per_second"] for r in results)
            summaries.append({
                "requested_workers": workers,
                "worker_threads": results[0]["worker_threads"],
                "median_games_per_second": rate,
                "median_seconds": median(r["seconds"] for r in results),
                "speedup_vs_one_worker": rate / reference if reference is not None else None,
            })
        print(json.dumps({
            "kind": "scaling_summary",
            "simulations_per_move": budget,
            "games_per_repeat": args.games,
            "repeats": args.repeats,
            "recommended_workers": max(summaries, key=lambda r: r["median_games_per_second"])["worker_threads"],
            "measurements": summaries,
        }), flush=True)
    print(json.dumps(opening_search(args.opening_simulations, args.seed)), flush=True)


if __name__ == "__main__":
    main()
