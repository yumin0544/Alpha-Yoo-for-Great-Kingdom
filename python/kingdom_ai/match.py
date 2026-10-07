"""Recorded, color-paired model matches with bounded parallel scheduling."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import math
import random
from threading import Event, RLock

import my_board_engine as engine
import torch

from .encoding import move_to_action
from .batching import BatchedEvaluator
from .encoded_batching import EncodedBatchedEvaluator
from .evaluation import _integer, _play_batched_game, _play_encoded_game, _play_game, _real
from .model import PolicyValueNet


@dataclass(frozen=True)
class MatchOptions:
    games: int = 20
    simulations: int = 128
    c_puct: float = 1.5
    seed: int = 42
    opening_moves: int = 6
    opening_temperature: float = 1.0
    tactical_checks: bool = False
    workers: int = 1
    backend: str = "legacy"
    leaf_batch_size: int = 8
    reuse_tree: bool = True
    inference_wait_ms: float = 0.0

    def __post_init__(self):
        _integer("games", self.games, 2)
        if self.games % 2:
            raise ValueError("games must be even for color pairing")
        _integer("simulations", self.simulations, 1)
        _integer("seed", self.seed, 0)
        if self.seed >= 2 ** 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        _integer("opening_moves", self.opening_moves, 0)
        _real("c_puct", self.c_puct, positive=True)
        _real("opening_temperature", self.opening_temperature, positive=False)
        if type(self.tactical_checks) is not bool:
            raise TypeError("tactical_checks must be a bool")
        _integer("workers", self.workers, 1)
        if self.backend not in ("legacy", "batched_cpp"):
            raise ValueError("backend must be 'legacy' or 'batched_cpp'")
        _integer("leaf_batch_size", self.leaf_batch_size, 1)
        if type(self.reuse_tree) is not bool:
            raise TypeError("reuse_tree must be a bool")
        _real("inference_wait_ms", self.inference_wait_ms, positive=False)


class MatchCancelled(Exception):
    """Stop new games and abandon unfinished games after their current search."""


def board_rows(state):
    """Final or live board, including permanent territory ownership."""
    cells, owners = state.board.cells, state.ownership
    stones = {engine.Cell.Black: "x", engine.Cell.White: "o", engine.Cell.Neutral: "#"}
    territory = {engine.Cell.Black: "B", engine.Cell.White: "W"}
    points = [stones.get(cell, territory.get(owner, "."))
              for cell, owner in zip(cells, owners)]
    return ["".join(points[row * 9:(row + 1) * 9]) for row in range(9)]


def series_ratings(rating_a, rating_b, wins_a, games, k=32.0):
    """A local convention: one Elo-style update for the entire paired series.

    K weights a series, irrespective of its game count. This is a provisional
    internal score, not an absolute strength estimate or a persistent league.
    """
    for name, value in (("rating_a", rating_a), ("rating_b", rating_b)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite real number")
    _real("k", k, positive=True)
    _integer("games", games, 2)
    _integer("wins_a", wins_a, 0)
    if games % 2 or wins_a > games:
        raise ValueError("games must be even and wins_a cannot exceed games")
    exponent = (rating_b - rating_a) / 400.0
    expected = (0.0 if exponent > 300 else 1.0 if exponent < -300
                else 1.0 / (1.0 + 10.0 ** exponent))
    delta = k * (wins_a / games - expected)
    after_a, after_b = rating_a + delta, rating_b - delta
    if not math.isfinite(after_a) or not math.isfinite(after_b):
        raise ValueError("Rating update overflowed")
    return {"method": "internal_series_elo", "k_per_series": k,
            "before": {"a": rating_a, "b": rating_b},
            "after": {"a": after_a, "b": after_b}, "delta_a": delta}


def play_match(model_a, model_b, options=None, *, on_game=None, on_move=None,
               on_game_start=None, should_stop=None, diagnostics=None):
    """Run A-black/B-white then B-black/A-white for every seeded pair.

    Callbacks receive independent snapshots so display code cannot alter the
    active game or returned records. Display callbacks are serialized; should_stop
    can be polled concurrently. on_move and on_game_start run on
    game workers, on_game on the calling thread in completion order. Returned
    records are sorted by game number. At most min(workers, games) games are
    submitted at once, so large series do not allocate thousands of trees.
    Cancellation drains completed games before raising, without a summary.
    """
    if not isinstance(model_a, PolicyValueNet) or not isinstance(model_b, PolicyValueNet):
        raise TypeError("Both participants must be PolicyValueNet models")
    options = MatchOptions() if options is None else options
    if not isinstance(options, MatchOptions):
        raise TypeError("options must be MatchOptions")
    for callback in (on_game, on_move, on_game_start, should_stop):
        if callback is not None and not callable(callback):
            raise TypeError("Callbacks must be callable or None")
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError("diagnostics must be a dict or None")
    if options.backend == "batched_cpp" and not hasattr(engine, "BatchedPUCT"):
        raise RuntimeError("The installed engine does not support BatchedPUCT")
    modes = {module: module.training for model in (model_a, model_b)
             for module in model.modules()}
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    devices = {parameter.device for model in (model_a, model_b)
               for parameter in model.parameters() if parameter.device.type == "cuda"}
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    records = []
    search_stats = {}
    aborted = Event()
    callbacks = RLock()
    workers = min(options.workers, options.games)

    def check_stop():
        if aborted.is_set() or (should_stop is not None and should_stop()):
            aborted.set()
            raise MatchCancelled("사용자 중단")

    def run_game(index, participants, player):
        check_stop()
        pair = (index - 1) // 2
        color = engine.Cell.Black if index % 2 else engine.Cell.White
        pair_seed = (options.seed + pair) % (2 ** 64)
        if on_game_start is not None:
            with callbacks:
                check_stop()
                on_game_start(index, color, pair_seed)
        actions = []
        final_state = None

        def observe(actor, move, state):
            nonlocal final_state
            # A game that has already finished remains a durable result even
            # if another worker observes a stop at the same moment.
            with callbacks:
                if not state.result.finished():
                    check_stop()
                actions.append(move_to_action(move))
                final_state = state
                if on_move is not None:
                    on_move(index, actor, move, state.copy())

        kwargs = dict(simulations=options.simulations, c_puct=options.c_puct,
                      seed=pair_seed, opening_moves=options.opening_moves,
                      opening_temperature=options.opening_temperature,
                      tactical_checks=options.tactical_checks, on_move=observe)
        if options.backend == "batched_cpp":
            kwargs.update(leaf_batch_size=options.leaf_batch_size, reuse_tree=options.reuse_tree)
        result = player(*participants, color, **kwargs)
        winner, reason, plies = result[:3]
        if final_state is None or not final_state.result.finished():
            raise RuntimeError("Match ended without a completed state")
        score = final_state.score()
        record = {
            "index": index, "pair_index": pair + 1, "seed": pair_seed,
            "model_a_color": color.name,
            "winner": "a" if winner == color else "b",
            "winner_color": winner.name, "reason": reason.name,
            "plies": plies, "actions": actions,
            "territory": {"black": score.black, "white": score.white},
            "board": board_rows(final_state),
        }
        return record, result[3] if len(result) > 3 else {}

    def accept(result):
        record, stats = result
        records.append(record)
        for key, value in stats.items():
            search_stats[key] = (max(search_stats.get(key, 0), value) if key.startswith("max_")
                                 else search_stats.get(key, 0) + value)
        if on_game is not None:
            with callbacks:
                on_game(deepcopy(record))

    try:
        model_a.eval()
        model_b.eval()
        with ExitStack() as stack:
            participants, player = (model_a, model_b), _play_game
            if options.backend == "batched_cpp" or workers > 1:
                encoded = options.backend == "batched_cpp"
                service = EncodedBatchedEvaluator if encoded else BatchedEvaluator
                participants = tuple(stack.enter_context(service(
                    model, device=next(model.parameters()).device,
                    max_batch_size=workers * (options.leaf_batch_size if encoded else 1),
                    max_wait_ms=options.inference_wait_ms if encoded else 0.0,
                )) for model in (model_a, model_b))
                player = _play_encoded_game if encoded else _play_batched_game
            if workers == 1:
                for index in range(1, options.games + 1):
                    accept(run_game(index, participants, player))
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="match") as pool:
                    pending = set()
                    next_index = 1
                    failure = None
                    callback_failed = False
                    try:
                        while pending or (next_index <= options.games and failure is None):
                            while len(pending) < workers and next_index <= options.games and failure is None:
                                if should_stop is not None and should_stop():
                                    failure = MatchCancelled("사용자 중단")
                                    aborted.set()
                                    break
                                pending.add(pool.submit(run_game, next_index, participants, player))
                                next_index += 1
                            if not pending:
                                break
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for future in done:
                                try:
                                    result = future.result()
                                except Exception as error:
                                    if failure is None or isinstance(failure, MatchCancelled):
                                        failure = error
                                    aborted.set()
                                else:
                                    if not callback_failed:
                                        try:
                                            accept(result)
                                        except Exception as error:
                                            failure, callback_failed = error, True
                                            aborted.set()
                        if failure is not None:
                            raise failure
                    finally:
                        # Also release worker searches if the calling thread is
                        # interrupted. Services remain alive until workers exit.
                        aborted.set()
            if diagnostics is not None:
                diagnostics.update(backend=options.backend, workers=workers,
                                   leaf_batch_size=options.leaf_batch_size,
                                   reuse_tree=options.reuse_tree, search=search_stats)
                if player != _play_game:
                    diagnostics.update(model_a=participants[0].stats, model_b=participants[1].stats)
    finally:
        for module, training in modes.items():
            module.training = training
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        for device, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
    records.sort(key=lambda record: record["index"])
    wins_a = sum(record["winner"] == "a" for record in records)
    endings = {}
    for record in records:
        endings[record["reason"]] = endings.get(record["reason"], 0) + 1
    return {
        "games": len(records), "wins_a": wins_a, "wins_b": len(records) - wins_a,
        "win_rate_a": wins_a / len(records),
        "wins_a_as_black": sum(record["winner"] == "a" and record["model_a_color"] == "Black"
                               for record in records),
        "wins_a_as_white": sum(record["winner"] == "a" and record["model_a_color"] == "White"
                               for record in records),
        "endings": endings, "total_plies": sum(record["plies"] for record in records),
        "records": records,
    }
