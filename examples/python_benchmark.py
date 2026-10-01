"""Measure completed self-play games separately from MCTS simulations."""

import argparse
from collections import Counter
import json
import os
import platform
from time import perf_counter

import my_board_engine as engine


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def completed_game(searcher):
    game = engine.State()
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


def self_play(simulations_per_move, games, seed):
    options = engine.MCTSOptions(simulations=simulations_per_move, seed=seed)
    completed_game(engine.MCTS(options))  # Warm up outside the timed interval.
    searcher = engine.MCTS(options)
    plies = simulations = 0
    endings = Counter()
    started = perf_counter()
    for _ in range(games):
        turns, searches, reason = completed_game(searcher)
        plies += turns
        simulations += searches
        endings[reason] += 1
    elapsed = perf_counter() - started
    return {
        "kind": "completed_self_play",
        "simulations_per_move": simulations_per_move,
        "games": games,
        "seconds": elapsed,
        "games_per_second": games / elapsed,
        "seconds_per_game": elapsed / games,
        "mean_plies": plies / games,
        "simulations": simulations,
        "simulations_per_second": simulations / elapsed,
        "endings": dict(endings),
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
        "worker_threads": 1,
        "seed": args.seed,
        "time_limit_ms": 0,
        "neural_network": False,
    }), flush=True)
    for budget in args.simulations:
        print(json.dumps(self_play(budget, args.games, args.seed)), flush=True)
    print(json.dumps(opening_search(args.opening_simulations, args.seed)), flush=True)


if __name__ == "__main__":
    main()
