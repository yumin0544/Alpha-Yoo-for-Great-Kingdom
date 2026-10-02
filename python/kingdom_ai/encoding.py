"""Versioned observation/action contract for the confirmed game rules."""

from dataclasses import dataclass
from numbers import Integral

import torch
import my_board_engine as engine

BOARD_SIZE = 9
PASS_ACTION = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = PASS_ACTION + 1
FORMAT_VERSION = 1
FEATURE_NAMES = (
    "own_stones", "opponent_stones", "neutral_stones",
    "own_territory", "opponent_territory", "black_to_play",
    "own_remaining_fraction", "opponent_remaining_fraction",
    "consecutive_passes_fraction", "legal_placements",
)
INPUT_CHANNELS = len(FEATURE_NAMES)


@dataclass(frozen=True)
class EncodedState:
    features: torch.Tensor
    legal_mask: torch.Tensor
    to_play: engine.Cell


def _check_rules(state):
    rules = state.rules
    if (rules.suicide_rule != engine.SuicideRule.Loses
            or rules.allow_own_territory_moves
            or not rules.allow_single_edge_territory
            or rules.stones_per_player != 41):
        raise ValueError("Encoding v1 supports the confirmed default GameRules only")


def move_to_action(move: engine.Move) -> int:
    if not isinstance(move, engine.Move):
        raise TypeError("Expected an engine Move")
    if move.is_pass():
        return PASS_ACTION
    point = move.point
    if not engine.Board.in_bounds(point):
        raise ValueError("Move coordinates must be between 0 and 8")
    return point.row * BOARD_SIZE + point.col


def action_to_move(action: int) -> engine.Move:
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("Action must be an integer")
    if not 0 <= action < ACTION_SIZE:
        raise ValueError("Action must be between 0 and 81")
    action = int(action)
    if action == PASS_ACTION:
        return engine.Move.pass_turn()
    return engine.Move.place(action // BOARD_SIZE, action % BOARD_SIZE)


def encode_state(state: engine.State, device="cpu") -> EncodedState:
    if not isinstance(state, engine.State):
        raise TypeError("Expected an engine State")
    snapshot = state.copy()
    _check_rules(snapshot)
    actor = snapshot.to_play
    enemy = engine.opponent(actor)
    cells = snapshot.board.cells
    owners = snapshot.ownership
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    for move in snapshot.legal_moves():
        mask[move_to_action(move)] = True
    planes = [
        [cell == actor for cell in cells],
        [cell == enemy for cell in cells],
        [cell == engine.Cell.Neutral for cell in cells],
        [owner == actor for owner in owners],
        [owner == enemy for owner in owners],
        [actor == engine.Cell.Black] * PASS_ACTION,
        [snapshot.remaining_stones(actor) / 41.0] * PASS_ACTION,
        [snapshot.remaining_stones(enemy) / 41.0] * PASS_ACTION,
        [snapshot.consecutive_passes / 2.0] * PASS_ACTION,
        mask[:PASS_ACTION].tolist(),
    ]
    features = torch.tensor(planes, dtype=torch.float32, device=device)
    return EncodedState(
        features.reshape(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        mask.to(device=device), actor,
    )


def terminal_value(state: engine.State) -> float:
    if not state.result.finished():
        raise ValueError("The game has not finished")
    return 1.0 if state.result.winner == state.to_play else -1.0


def visit_policy(search: engine.SearchResult, state: engine.State) -> torch.Tensor:
    """Normalize actual C++ root visits, leaving unexpanded actions at zero."""
    snapshot = state.copy()
    _check_rules(snapshot)
    if snapshot.result.finished():
        raise ValueError("A terminal state has no policy training target")
    legal = {move_to_action(move) for move in snapshot.legal_moves()}
    visits = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    seen = set()
    total = 0
    for item in search.moves:
        if (isinstance(item.visits, bool) or not isinstance(item.visits, Integral)
                or item.visits < 0):
            raise ValueError("Visit counts must be non-negative integers")
        action = move_to_action(item.move)
        if action not in legal or action in seen:
            raise ValueError("Search statistics must contain distinct legal moves")
        seen.add(action)
        visits[action] = item.visits
        total += item.visits
    if total < 1 or total != search.simulations:
        raise ValueError("Search visit counts must sum to the completed simulations")
    return visits / visits.sum()
