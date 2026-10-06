"""One-move tactical checks against the verified C++ state transitions.

These checks narrow root choices without changing the game's legal moves. They
take immediate wins and avoid immediate losses when a safe alternative exists;
they do not claim to solve threats that require more than one opposing move.
"""

from dataclasses import dataclass

import my_board_engine as engine

from .encoding import BOARD_SIZE, _check_rules, action_to_move, move_to_action


_NEIGHBOURS = tuple(
    tuple(r * BOARD_SIZE + c for r, c in (
        (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)
    ) if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE)
    for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
)


@dataclass(frozen=True)
class TacticalChoices:
    """Root action sets; immediate wins are also members of ``safe_actions``."""

    legal_actions: frozenset[int]
    winning_actions: frozenset[int]
    safe_actions: frozenset[int]

    @property
    def preferred_actions(self) -> frozenset[int]:
        return self.winning_actions or self.safe_actions or self.legal_actions


def _has_winning_reply(state: engine.State) -> bool:
    """Identify capture liberties, then verify only these replies in the engine."""
    actor = state.to_play
    if state.consecutive_passes == 1:
        reply = state.copy()
        outcome = reply.play(engine.Move.pass_turn())
        if outcome.accepted() and reply.result.finished() and reply.result.winner == actor:
            return True
    if state.remaining_stones(actor) < 1:
        return False
    cells = state.board.cells
    enemy = engine.opponent(actor)
    examined = set()
    checked_liberties = set()
    for point, cell in enumerate(cells):
        if cell != enemy or point in examined:
            continue
        group = [point]
        examined.add(point)
        liberties = set()
        for member in group:
            for adjacent in _NEIGHBOURS[member]:
                if cells[adjacent] == engine.Cell.Empty:
                    liberties.add(adjacent)
                elif cells[adjacent] == enemy and adjacent not in examined:
                    examined.add(adjacent)
                    group.append(adjacent)
        if len(liberties) != 1:
            continue
        liberty = next(iter(liberties))
        if liberty in checked_liberties:
            continue
        checked_liberties.add(liberty)
        reply = state.copy()
        outcome = reply.play(action_to_move(liberty))
        if (outcome.accepted() and reply.result.reason == engine.EndReason.Capture
                and reply.result.winner == actor):
            return True
    return False


def analyze_tactics(state: engine.State) -> TacticalChoices:
    """Classify legal root moves without mutating the caller's state.

    A safe action neither loses immediately nor lets the opponent win in one
    move by capture or consecutive passes. When every action loses, the full
    legal set remains selectable, leaving the original search as the fallback.
    """
    if not isinstance(state, engine.State):
        raise TypeError("Expected an engine State")
    snapshot = state.copy()
    _check_rules(snapshot)
    legal, winning, safe = set(), set(), set()
    actor = snapshot.to_play
    for move in snapshot.legal_moves():
        action = move_to_action(move)
        legal.add(action)
        after = snapshot.copy()
        outcome = after.play(move)
        if not outcome.accepted():
            raise RuntimeError("An engine legal move was rejected")
        if after.result.finished():
            if after.result.winner == actor:
                winning.add(action)
                safe.add(action)
        elif not _has_winning_reply(after):
            safe.add(action)
    return TacticalChoices(frozenset(legal), frozenset(winning), frozenset(safe))


def select_tactical_move(result, choices: TacticalChoices) -> engine.Move:
    """Use the search's highest-visit safe move, preserving a losing fallback."""
    if not isinstance(choices, TacticalChoices):
        raise TypeError("Expected TacticalChoices")
    allowed = choices.preferred_actions
    if result.best_move is None or not allowed:
        raise ValueError("A terminal or unsearched state has no move to select")
    if move_to_action(result.best_move) in allowed:
        return result.best_move
    candidates = [item for item in result.moves if move_to_action(item.move) in allowed]
    if not candidates:
        raise ValueError("Search statistics do not contain the preferred legal actions")
    best = max(candidates, key=lambda item: (
        item.visits, item.value, -move_to_action(item.move)
    ))
    return best.move
