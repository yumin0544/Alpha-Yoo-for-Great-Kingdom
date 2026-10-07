"""Immediate checks and opt-in bounded proof search using verified C++ rules.

These checks narrow root choices without changing the game's legal moves. They
take immediate wins and avoid immediate losses when a safe alternative exists;
The immediate checker does not solve deeper threats. The separate deep checker
accepts only C++ solver certificates, while cut-off branches remain UNKNOWN.
"""

from dataclasses import dataclass
import math

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


@dataclass(frozen=True)
class DeepTacticalChoices:
    """Depth/budget-limited proof sets, never a claim that UNKNOWN is safe.

    A ply is one actual placement or pass by either player. Proofs concern the
    exact Kingdom terminal winner, not ordinary Go life/death or territory value.
    A winning search can stop at one proved winning action; other winning actions
    may consequently remain unknown.
    """

    legal_actions: frozenset[int]
    winning_actions: frozenset[int]
    losing_actions: frozenset[int]
    unknown_actions: frozenset[int]
    outcome: object
    nodes: int
    proof_depth: int
    completed_depth: int
    elapsed_ms: float
    budget_exhausted: bool
    principal_variation: tuple[engine.Move, ...]

    @property
    def preferred_actions(self) -> frozenset[int]:
        # When every root action is proved losing, retain the ordinary search
        # fallback instead of pretending that there is a safe legal alternative.
        return self.winning_actions or self.unknown_actions or self.legal_actions


def analyze_deep_tactics(state: engine.State, *, max_depth=12, max_nodes=20000,
                         time_limit_ms=250) -> DeepTacticalChoices:
    """Ask the bounded C++ solver for certified root action classifications.

    Incomplete/depth-cut branches remain UNKNOWN; no policy/value label or pruning
    proof is inferred from a promising principal variation. ``time_limit_ms=0``
    disables the timer, but the positive node budget still bounds the search.
    """
    if not isinstance(state, engine.State):
        raise TypeError("Expected an engine State")
    for name, value, minimum in (("max_depth", max_depth, 1), ("max_nodes", max_nodes, 1),
                                 ("time_limit_ms", time_limit_ms, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_depth > 256:
        raise ValueError("max_depth must not exceed the engine limit of 256 plies")
    if not hasattr(engine, "solve_tactics") or not hasattr(engine, "TacticalSolverOptions"):
        raise RuntimeError("The installed engine binding lacks solve_tactics; rebuild/install this project")
    snapshot = state.copy()
    _check_rules(snapshot)
    legal = frozenset(move_to_action(move) for move in snapshot.legal_moves())
    result = engine.solve_tactics(snapshot, engine.TacticalSolverOptions(
        max_depth=max_depth, max_nodes=max_nodes, time_limit_ms=time_limit_ms))

    def action_set(moves, name):
        actions = [move_to_action(move) for move in moves]
        if len(set(actions)) != len(actions) or not set(actions) <= legal:
            raise RuntimeError(f"Solver returned invalid {name}")
        return frozenset(actions)

    winning = action_set(result.winning_moves, "winning moves")
    losing = action_set(result.losing_moves, "losing moves")
    unknown = action_set(result.unknown_moves, "unknown moves")
    if (winning & losing or winning & unknown or losing & unknown
            or winning | losing | unknown != legal):
        raise RuntimeError("Solver action classes must partition all legal root moves")
    if (result.nodes < 0 or result.proof_depth < 0 or result.completed_depth < 0
            or not math.isfinite(result.elapsed_ms) or result.elapsed_ms < 0):
        raise RuntimeError("Solver returned invalid proof metadata")
    return DeepTacticalChoices(legal, winning, losing, unknown, result.outcome,
                               result.nodes, result.proof_depth, result.completed_depth,
                               result.elapsed_ms, result.budget_exhausted,
                               tuple(result.principal_variation))


def _deep_preferred_actions(choices, immediate_choices):
    if not isinstance(choices, DeepTacticalChoices):
        raise TypeError("Expected DeepTacticalChoices")
    if immediate_choices is None:
        return choices.preferred_actions
    if not isinstance(immediate_choices, TacticalChoices):
        raise TypeError("Expected TacticalChoices for immediate_choices")
    if immediate_choices.legal_actions != choices.legal_actions:
        raise ValueError("Immediate and deep choices must describe the same legal root actions")
    winning = choices.winning_actions | immediate_choices.winning_actions
    losing = choices.losing_actions | (choices.legal_actions - immediate_choices.safe_actions)
    if winning & losing:
        raise RuntimeError("Immediate and deep tactical proofs contradict one another")
    return winning or (choices.legal_actions - losing) or choices.legal_actions


def select_deep_tactical_move(result, choices: DeepTacticalChoices, *,
                              immediate_choices=None) -> engine.Move:
    """CPU PUCT selection excluding proved losses, with an honest losing fallback."""
    allowed = _deep_preferred_actions(choices, immediate_choices)
    if result.best_move is None or not allowed:
        raise ValueError("A terminal or unsearched state has no move to select")
    if move_to_action(result.best_move) in allowed:
        return result.best_move
    candidates = [item for item in result.moves if move_to_action(item.move) in allowed]
    if not candidates:
        raise ValueError("Search statistics do not contain the preferred legal actions")
    best = max(candidates, key=lambda item: (
        item.visits, item.value, -move_to_action(item.move)))
    return best.move


def select_deep_tactical_action(result, choices: DeepTacticalChoices, *, lane=0,
                                immediate_choices=None) -> int:
    """CUDA result selection using its real [N,82] visits/priors fields.

    ``GpuSearchResult.values`` has shape [N] (root values), not per-action values.
    It must not be indexed or broadcast as an edge-value tiebreaker.
    """
    allowed = _deep_preferred_actions(choices, immediate_choices)
    if type(lane) is not int or lane < 0:
        raise ValueError("lane must be a non-negative integer")
    if (result.visits.ndim != 2 or result.visits.shape[1] != BOARD_SIZE ** 2 + 1
            or result.priors.shape != result.visits.shape
            or result.actions.ndim != 1 or result.actions.shape[0] != result.visits.shape[0]
            or lane >= result.visits.shape[0]):
        raise ValueError("Invalid GPU root statistics shapes")
    if not allowed:
        raise ValueError("A terminal state has no move to select")
    original = int(result.actions[lane].item())
    if original in allowed:
        return original
    visits = result.visits[lane].detach().cpu().tolist()
    priors = result.priors[lane].detach().cpu().tolist()
    if any(value < 0 or int(value) != value for value in visits):
        raise ValueError("GPU root visits must be non-negative integers")
    if any(not math.isfinite(value) or value < 0 for value in priors):
        raise ValueError("GPU root priors must be finite and non-negative")
    return max(allowed, key=lambda action: (visits[action], priors[action], -action))
