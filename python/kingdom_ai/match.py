"""Visible two-model matches using the existing sequential evaluation protocol."""

from copy import deepcopy
from dataclasses import dataclass
import math
import random

import my_board_engine as engine
import torch

from .encoding import move_to_action
from .evaluation import _integer, _play_game, _real
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


def play_match(model_a, model_b, options=None, *, on_game=None, on_move=None):
    """Run A-black/B-white then B-black/A-white for every seeded pair.

    Callbacks receive independent snapshots so display code cannot alter the
    active game or returned records. A callback error propagates; module modes
    are restored in all cases. No incomplete match summary is returned.
    """
    if not isinstance(model_a, PolicyValueNet) or not isinstance(model_b, PolicyValueNet):
        raise TypeError("Both participants must be PolicyValueNet models")
    options = MatchOptions() if options is None else options
    if not isinstance(options, MatchOptions):
        raise TypeError("options must be MatchOptions")
    for callback in (on_game, on_move):
        if callback is not None and not callable(callback):
            raise TypeError("Callbacks must be callable or None")
    modes = {module: module.training for model in (model_a, model_b)
             for module in model.modules()}
    python_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    devices = {parameter.device for model in (model_a, model_b)
               for parameter in model.parameters() if parameter.device.type == "cuda"}
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    records = []
    try:
        model_a.eval()
        model_b.eval()
        for pair in range(options.games // 2):
            pair_seed = (options.seed + pair) % (2 ** 64)
            for color in (engine.Cell.Black, engine.Cell.White):
                index = len(records) + 1
                actions = []
                final_state = None

                def observe(actor, move, state):
                    nonlocal final_state
                    actions.append(move_to_action(move))
                    final_state = state
                    if on_move is not None:
                        on_move(index, actor, move, state.copy())

                winner, reason, plies = _play_game(
                    model_a, model_b, color, simulations=options.simulations,
                    c_puct=options.c_puct, seed=pair_seed,
                    opening_moves=options.opening_moves,
                    opening_temperature=options.opening_temperature,
                    tactical_checks=options.tactical_checks, on_move=observe,
                )
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
                records.append(record)
                if on_game is not None:
                    on_game(deepcopy(record))
    finally:
        for module, training in modes.items():
            module.training = training
        random.setstate(python_rng)
        torch.set_rng_state(cpu_rng)
        for device, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
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
