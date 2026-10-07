"""Compare two models with paired colors and equal deterministic search budgets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from types import SimpleNamespace

import my_board_engine as engine
import torch

from .batching import BatchedEvaluator
from .encoded_batching import EncodedBatchedEvaluator
from .model import PolicyValueNet
from .puct import PUCT, PUCTOptions, sample_visits
from .encoding import move_to_action
from .tactics import analyze_tactics, select_tactical_move


@dataclass(frozen=True)
class EvaluationResult:
    """Results from the candidate's perspective; ``endings`` uses enum names."""

    games: int
    wins: int
    losses: int
    wins_as_black: int
    wins_as_white: int
    total_plies: int
    endings: dict[str, int]

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _real(name: str, value: float, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _sample_tactical_visits(result, choices, *, temperature, generator):
    """Keep opening visit sampling within the same tactical set for either model.

    A shallow statistics view leaves the actual search result untouched. If
    all preferred moves are unvisited, the tactical selector still supplies a
    legal move instead of sampling known losing visited actions.
    """
    move = select_tactical_move(result, choices)
    moves = [item for item in result.moves if move_to_action(item.move) in choices.preferred_actions]
    visits = sum(item.visits for item in moves)
    if visits == 0:
        return move
    filtered = SimpleNamespace(best_move=move, simulations=visits, moves=moves)
    return sample_visits(filtered, temperature=temperature, generator=generator)


def _play_game(
    candidate: PolicyValueNet,
    reference: PolicyValueNet,
    candidate_color: engine.Cell,
    *,
    simulations: int,
    c_puct: float,
    seed: int,
    opening_moves: int,
    opening_temperature: float,
    tactical_checks: bool = False,
    on_move=None,
) -> tuple[engine.Cell, engine.EndReason, int]:
    # New searchers and a reset generator make each swapped-color pair use
    # the same randomness. Search itself has no noise or wall-clock budget.
    searchers = {
        candidate_color: PUCT(candidate, PUCTOptions(
            simulations=simulations, c_puct=c_puct, seed=seed,
            time_limit_ms=0, dirichlet_epsilon=0,
        )),
        engine.Cell.White if candidate_color == engine.Cell.Black else engine.Cell.Black:
            PUCT(reference, PUCTOptions(
                simulations=simulations, c_puct=c_puct, seed=seed,
                time_limit_ms=0, dirichlet_epsilon=0,
            )),
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    game = engine.State()
    plies = 0
    while not game.result.finished():
        actor = game.to_play
        result = searchers[actor].search(game)
        temperature = opening_temperature if plies < opening_moves else 0.0
        if tactical_checks:
            move = _sample_tactical_visits(result, analyze_tactics(game),
                                           temperature=temperature, generator=generator)
        else:
            move = sample_visits(result, temperature=temperature, generator=generator)
        if not game.play(move).accepted():
            raise RuntimeError("Evaluation PUCT produced a rejected move")
        plies += 1
        if plies > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Evaluation game exceeded its finite move bound")
        if on_move is not None:
            on_move(actor, move, game.copy())
    if game.result.winner not in (engine.Cell.Black, engine.Cell.White):
        raise RuntimeError("A completed evaluation game must have a winner")
    return game.result.winner, game.result.reason, plies


def _play_batched_game(
    candidate_evaluator: BatchedEvaluator,
    reference_evaluator: BatchedEvaluator,
    candidate_color: engine.Cell,
    *,
    simulations: int,
    c_puct: float,
    seed: int,
    opening_moves: int,
    opening_temperature: float,
    tactical_checks: bool = False,
) -> tuple[engine.Cell, engine.EndReason, int]:
    """Play one worker-owned game through shared batched model callbacks."""
    reference_color = (
        engine.Cell.White if candidate_color == engine.Cell.Black else engine.Cell.Black
    )
    options = PUCTOptions(
        simulations=simulations, c_puct=c_puct, seed=seed,
        time_limit_ms=0, dirichlet_epsilon=0,
    )
    # A C++ PUCT object owns mutable tree/RNG state, so no searcher is shared
    # between game workers. Only the thread-safe inference services are shared.
    searchers = {
        candidate_color: engine.PUCT(options),
        reference_color: engine.PUCT(options),
    }
    evaluators = {
        candidate_color: candidate_evaluator,
        reference_color: reference_evaluator,
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    game = engine.State()
    plies = 0
    while not game.result.finished():
        actor = game.to_play
        result = searchers[actor].search(game, evaluators[actor])
        temperature = opening_temperature if plies < opening_moves else 0.0
        if tactical_checks:
            move = _sample_tactical_visits(result, analyze_tactics(game),
                                           temperature=temperature, generator=generator)
        else:
            move = sample_visits(result, temperature=temperature, generator=generator)
        if not game.play(move).accepted():
            raise RuntimeError("Evaluation PUCT produced a rejected move")
        plies += 1
        if plies > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Evaluation game exceeded its finite move bound")
    if game.result.winner not in (engine.Cell.Black, engine.Cell.White):
        raise RuntimeError("A completed evaluation game must have a winner")
    return game.result.winner, game.result.reason, plies


def _play_encoded_game(
    candidate_evaluator: EncodedBatchedEvaluator,
    reference_evaluator: EncodedBatchedEvaluator,
    candidate_color: engine.Cell,
    *, simulations: int, c_puct: float, seed: int, opening_moves: int,
    opening_temperature: float, tactical_checks: bool,
    leaf_batch_size: int, reuse_tree: bool,
):
    """Keep one reusable tree per model, never mix opposing model values."""
    reference_color = (engine.Cell.White if candidate_color == engine.Cell.Black
                       else engine.Cell.Black)
    options = PUCTOptions(simulations=simulations, c_puct=c_puct, seed=seed,
                          time_limit_ms=0, dirichlet_epsilon=0)
    searchers = {
        candidate_color: engine.BatchedPUCT(options, leaf_batch_size=leaf_batch_size,
                                           reuse_tree=reuse_tree),
        reference_color: engine.BatchedPUCT(options, leaf_batch_size=leaf_batch_size,
                                           reuse_tree=reuse_tree),
    }
    evaluators = {candidate_color: candidate_evaluator, reference_color: reference_evaluator}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    game = engine.State()
    plies = 0
    stats = {}
    while not game.result.finished():
        actor = game.to_play
        result = searchers[actor].search(game, evaluators[actor])
        # Search-local counters must be collected now, before this model's next
        # search overwrites them. Lifecycle advance/reuse counters are cumulative.
        for key, value in searchers[actor].stats.items():
            if key not in ("advance_calls", "reuse_hits"):
                stats[key] = (max(stats.get(key, 0), value) if key.startswith("max_")
                              else stats.get(key, 0) + value)
        temperature = opening_temperature if plies < opening_moves else 0.0
        move = (_sample_tactical_visits(result, analyze_tactics(game),
                                       temperature=temperature, generator=generator)
                if tactical_checks else sample_visits(result, temperature=temperature,
                                                       generator=generator))
        if not game.play(move).accepted():
            raise RuntimeError("Evaluation BatchedPUCT produced a rejected move")
        # Advance BOTH model-owned trees after every real move, including the
        # opponent's. An unexplored branch safely drops that model's cache.
        for searcher in searchers.values():
            searcher.advance(move)
        plies += 1
        if plies > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Evaluation game exceeded its finite move bound")
    if game.result.winner not in (engine.Cell.Black, engine.Cell.White):
        raise RuntimeError("A completed evaluation game must have a winner")
    for searcher in searchers.values():
        for key in ("advance_calls", "reuse_hits"):
            stats[key] = stats.get(key, 0) + searcher.stats.get(key, 0)
    return game.result.winner, game.result.reason, plies, stats


def evaluate_models(
    candidate: PolicyValueNet,
    reference: PolicyValueNet,
    *,
    games: int = 20,
    simulations: int = 128,
    c_puct: float = 1.5,
    seed: int = 42,
    opening_moves: int = 6,
    opening_temperature: float = 1.0,
    tactical_checks: bool = False,
    workers: int = 1,
    backend: str = "legacy",
    leaf_batch_size: int = 8,
    reuse_tree: bool = True,
    inference_wait_ms: float = 0.0,
    diagnostics: dict | None = None,
) -> EvaluationResult:
    """Play equal-budget pairs, with the candidate first black then white.

    Pair ``i`` uses ``(seed + i) % 2**64`` for both games. During the first
    ``opening_moves`` plies, moves are sampled from root visits; subsequent
    moves use the most visited root action. Root Dirichlet noise is disabled.
    Identical deterministic models therefore score exactly 50% in each pair,
    even when the rules or their preferred opening favor one color.

    Parameters, gradients, model devices, global RNG and thread settings are
    preserved. Every module's train/eval flag is restored, including on error.
    The default ``workers=1`` path retains the original sequential execution.
    Larger values run independent games in parallel and batch each model's
    inference requests; the effective worker count is capped at ``games``.
    ``backend='batched_cpp'`` collects several C++ leaves before each callback,
    receives C++-encoded arrays, and optionally reuses each model's own tree.
    Its visit distribution can differ from legacy sequential PUCT. The leaf
    batching limit is independent of the number of parallel game workers.
    ``inference_wait_ms`` controls optional partial-batch coalescing only in
    this new backend. A supplied ``diagnostics`` dict receives inference and
    cumulative search statistics without changing ``EvaluationResult``.
    All games use the default engine rules.
    With ``tactical_checks=True``, both models take immediate wins and select
    only one-move safe root actions whenever such alternatives exist. Opening
    temperature still samples their actual visits; an unvisited tactical action
    is selected deterministically if none of these actions has visits.
    """
    if not isinstance(candidate, PolicyValueNet) or not isinstance(reference, PolicyValueNet):
        raise TypeError("candidate and reference must be PolicyValueNet models")
    _integer("games", games, 2)
    if games % 2:
        raise ValueError("games must be even so colors can be paired")
    _integer("simulations", simulations, 1)
    _real("c_puct", c_puct, positive=True)
    _integer("seed", seed, 0)
    if seed >= 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    _integer("opening_moves", opening_moves, 0)
    _real("opening_temperature", opening_temperature, positive=False)
    if type(tactical_checks) is not bool:
        raise TypeError("tactical_checks must be a bool")
    _integer("workers", workers, 1)
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in ("legacy", "batched_cpp"):
        raise ValueError("backend must be 'legacy' or 'batched_cpp'")
    _integer("leaf_batch_size", leaf_batch_size, 1)
    if type(reuse_tree) is not bool:
        raise TypeError("reuse_tree must be a bool")
    _real("inference_wait_ms", inference_wait_ms, positive=False)
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError("diagnostics must be a dict or None")
    if backend == "batched_cpp" and not hasattr(engine, "BatchedPUCT"):
        raise RuntimeError("batched_cpp evaluation requires rebuilt BatchedPUCT bindings")

    # NeuralAgent restores the top-level flag after each inference. Preserve
    # all flags here as well, because a caller may use mixed module modes.
    modes = {module: module.training for model in (candidate, reference)
             for module in model.modules()}
    wins = wins_as_black = wins_as_white = total_plies = 0
    endings: dict[str, int] = {}
    try:
        jobs = [
            (color, (seed + pair) % (2 ** 64))
            for pair in range(games // 2)
            for color in (engine.Cell.Black, engine.Cell.White)
        ]
        if backend == "batched_cpp":
            worker_count = min(workers, games)
            candidate_device = next(candidate.parameters()).device
            reference_device = next(reference.parameters()).device
            row_cap = worker_count * leaf_batch_size
            with EncodedBatchedEvaluator(
                candidate, device=candidate_device, max_batch_size=row_cap,
                max_wait_ms=inference_wait_ms,
            ) as candidate_evaluator, EncodedBatchedEvaluator(
                reference, device=reference_device, max_batch_size=row_cap,
                max_wait_ms=inference_wait_ms,
            ) as reference_evaluator:
                def play_encoded(job):
                    color, pair_seed = job
                    return _play_encoded_game(
                        candidate_evaluator, reference_evaluator, color,
                        simulations=simulations, c_puct=c_puct, seed=pair_seed,
                        opening_moves=opening_moves, opening_temperature=opening_temperature,
                        tactical_checks=tactical_checks, leaf_batch_size=leaf_batch_size,
                        reuse_tree=reuse_tree,
                    )

                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    detailed = list(pool.map(play_encoded, jobs))
                results = [item[:3] for item in detailed]
                if diagnostics is not None:
                    search_stats = {}
                    for item in detailed:
                        for key, value in item[3].items():
                            search_stats[key] = (
                                max(search_stats.get(key, 0), value) if key.startswith("max_")
                                else search_stats.get(key, 0) + value
                            )
                    diagnostics.update({
                        "backend": backend, "workers": worker_count,
                        "leaf_batch_size": leaf_batch_size, "reuse_tree": reuse_tree,
                        "inference_wait_ms": inference_wait_ms,
                        "candidate_inference": candidate_evaluator.stats,
                        "reference_inference": reference_evaluator.stats,
                        "search": search_stats,
                    })
        elif workers == 1:
            results = [
                _play_game(
                    candidate, reference, color, simulations=simulations,
                    c_puct=c_puct, seed=pair_seed, opening_moves=opening_moves,
                    opening_temperature=opening_temperature,
                    tactical_checks=tactical_checks,
                )
                for color, pair_seed in jobs
            ]
        else:
            worker_count = min(workers, games)
            candidate_device = next(candidate.parameters()).device
            reference_device = next(reference.parameters()).device
            with BatchedEvaluator(
                candidate, device=candidate_device, max_batch_size=worker_count,
                max_wait_ms=0,
            ) as candidate_evaluator, BatchedEvaluator(
                reference, device=reference_device, max_batch_size=worker_count,
                max_wait_ms=0,
            ) as reference_evaluator:
                def play(job):
                    color, pair_seed = job
                    return _play_batched_game(
                        candidate_evaluator, reference_evaluator, color,
                        simulations=simulations, c_puct=c_puct, seed=pair_seed,
                        opening_moves=opening_moves,
                        opening_temperature=opening_temperature,
                        tactical_checks=tactical_checks,
                    )

                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    # executor.map preserves paired input order even though games
                    # finish out of order, keeping aggregation deterministic.
                    results = list(pool.map(play, jobs))
                if diagnostics is not None:
                    diagnostics.update({
                        "backend": backend, "workers": worker_count,
                        "candidate_inference": candidate_evaluator.stats,
                        "reference_inference": reference_evaluator.stats,
                    })

        if diagnostics is not None and backend == "legacy" and workers == 1:
            diagnostics.update({"backend": backend, "workers": 1})

        for (color, _), (winner, reason, plies) in zip(jobs, results):
            total_plies += plies
            endings[reason.name] = endings.get(reason.name, 0) + 1
            if winner == color:
                wins += 1
                wins_as_black += color == engine.Cell.Black
                wins_as_white += color == engine.Cell.White
    finally:
        for module, training in modes.items():
            module.training = training
    return EvaluationResult(
        games=games, wins=wins, losses=games - wins,
        wins_as_black=wins_as_black, wins_as_white=wins_as_white,
        total_plies=total_plies, endings=endings,
    )
