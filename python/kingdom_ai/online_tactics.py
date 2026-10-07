"""Mine bounded, replayable tactical cases from the current self-play cycle.

Only action histories are retained while self-play is running. A uniform game
reservoir is replayed afterwards by the authoritative engine, so GPU tensor
features are never mistaken for complete states. Low-liberty chains are merely
a scheduling heuristic: they are not a proof, a training target, or solver
pruning. A separate full-width solver must certify every training label.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
import my_board_engine as engine

from .encoding import ACTION_SIZE, PASS_ACTION, action_to_move
from .training import GameData


@dataclass(frozen=True)
class _RecordedGame:
    index: int
    history: tuple[int, ...]
    winner: engine.Cell
    reason: engine.EndReason


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


_SLOT_ORDER = ("threat_onset", "threat_follow_up", "best_overall", "atari_defense",
               "pre_atari_defense")


def _threat_profile(state, ply):
    """Return ordering-only priority and large-chain threat metadata."""
    board = state.board
    cells = board.cells
    actor = state.to_play
    enemy = engine.opponent(actor)
    visited = set()
    own_one = own_two = enemy_one = enemy_two = critical_stones = 0
    large_enemy_two = []
    large_own_one = []
    large_own_two = []
    for index, cell in enumerate(cells):
        if cell not in (actor, enemy) or index in visited:
            continue
        point = engine.Board.position(index)
        group = board.group_at(point)
        members = frozenset(engine.Board.index(member) for member in group)
        visited.update(members)
        liberties = len(board.liberties(point))
        if liberties > 2:
            continue
        if liberties == 0:
            raise ValueError("Nonterminal replay contains a captured chain")
        critical_stones += len(group)
        if cell == actor:
            own_one += liberties == 1
            own_two += liberties == 2
            if liberties == 1 and len(group) >= 2:
                large_own_one.append((cell, members))
            if liberties == 2 and len(group) >= 2:
                large_own_two.append((cell, members))
        else:
            enemy_one += liberties == 1
            enemy_two += liberties == 2
            if liberties == 2 and len(group) >= 2:
                large_enemy_two.append((cell, members))
    if not (own_one or own_two or enemy_one or enemy_two):
        return None
    # Prefer positions before the final atari: immediate captures are already
    # covered by the shallow guard and add little to long-reading supervision.
    if enemy_two and not enemy_one:
        tier = 3
    elif own_one and not enemy_one:
        tier = 2
    elif enemy_two:
        tier = 1
    else:
        tier = 0
    return ((tier, min(own_two + enemy_two, 4), min(critical_stones, 12), -ply),
            large_enemy_two, large_own_one, large_own_two)


class TacticalPositionMiner:
    """Uniformly sample fresh games, then select diverse low-liberty prefixes.

    ``max_cases`` is the final solver-call bound. Retained histories are bounded
    by ``2 * max_cases`` and candidate metadata by ``5 * max_cases``.
    Five constant-size slots per game cover a multi-stone threat's onset,
    follow-up, a high-priority later position, an early atari defense and a
    two-liberty own group before atari. Their
    first-use order rotates across game ids and cycles, so full-size cycles do
    not always solve only the strongest late positions.
    No engine replay is performed by :meth:`add_game`; :meth:`cases` validates
    and replays only the reservoir after self-play. Repeated ``cases()`` calls
    return detached copies without consuming random numbers or mutating games.

    The caller owns the dedicated CPU generator and checkpoints its state.
    Recreating a miner for every cycle prevents old positions being advertised
    as newly generated cases.
    """

    def __init__(self, max_cases, *, generator: torch.Generator, iteration=0):
        _nonnegative_integer(max_cases, "max_cases")
        if max_cases == 0:
            raise ValueError("max_cases must be positive")
        _nonnegative_integer(iteration, "iteration")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Mining requires a dedicated CPU torch.Generator")
        self.max_cases = max_cases
        self.max_sampled_games = 2 * max_cases
        self.candidate_capacity = len(_SLOT_ORDER) * max_cases
        self.iteration = iteration
        self.generator = generator
        self.seen_games = 0
        self.candidate_count = 0
        self._games: list[_RecordedGame] = []
        self._cached_cases: list[dict] | None = None

    @property
    def sampled_games(self):
        return len(self._games)

    @property
    def sampled_game_indices(self):
        return tuple(record.index for record in self._games)

    def add_game(self, game: GameData, game_index: int):
        """Reservoir-sample a completed game without decoding its observations."""
        _nonnegative_integer(game_index, "game_index")
        if not isinstance(game, GameData):
            raise TypeError("Expected completed self-play GameData")
        history = getattr(game, "action_history", ())
        if (not isinstance(history, tuple) or not history
                or len(history) > 2 * engine.CELL_COUNT + 2
                or any(type(action) is not int or not 0 <= action < ACTION_SIZE
                       for action in history)):
            raise ValueError("Online mining requires a bounded tuple of integer actions")
        if len(game.samples) != len(history):
            raise ValueError("Game action history must align with every pre-move sample")
        if game.winner not in (engine.Cell.Black, engine.Cell.White):
            raise ValueError("Online mining requires a completed game winner")
        if game.reason not in (engine.EndReason.Capture, engine.EndReason.Suicide,
                               engine.EndReason.TwoPasses):
            raise ValueError("Online mining requires a completed game end reason")
        record = _RecordedGame(game_index, history, game.winner, game.reason)
        self.seen_games += 1
        if len(self._games) < self.max_sampled_games:
            self._games.append(record)
        else:
            slot = int(torch.randint(self.seen_games, (), generator=self.generator).item())
            if slot < self.max_sampled_games:
                self._games[slot] = record
        self._cached_cases = None
        self.candidate_count = 0

    def _game_candidates(self, record):
        state = engine.State()
        history = []
        slots = {}
        onset = None
        for ply, action in enumerate(record.history, 1):
            if not state.play(action_to_move(action)).accepted():
                raise ValueError(f"Game {record.index} action {ply} is rejected by the engine")
            history.append("pass" if action == PASS_ACTION else
                           [action // 9 + 1, action % 9 + 1])
            if state.result.finished():
                if ply != len(record.history):
                    raise ValueError(f"Game {record.index} continues after termination")
                continue
            profile = _threat_profile(state, ply)
            if profile is None:
                continue
            priority, enemy_two, own_one, own_two = profile
            update_slots = []
            if onset is None and enemy_two:
                color, members = max(enemy_two, key=lambda item: len(item[1]))
                onset = (color, min(members), ply)
                update_slots.append("threat_onset")
            elif (onset is not None and "threat_follow_up" not in slots
                  and ply >= onset[2] + 2
                  and any(color == onset[0] and onset[1] in members
                          for color, members in enemy_two)):
                # Track a stone anchor rather than the group's current minimum:
                # a connection can merge groups and change that minimum index.
                update_slots.append("threat_follow_up")
            if own_one and "atari_defense" not in slots:
                update_slots.append("atari_defense")
            if own_two and "pre_atari_defense" not in slots and not update_slots:
                # A heuristic selection only: escaping atari is NOT itself a
                # WIN label. The solver must prove the entire game's outcome.
                update_slots.append("pre_atari_defense")
            if "best_overall" not in slots or priority > slots["best_overall"][0]:
                update_slots.append("best_overall")
            if not update_slots:
                continue
            family = f"online_cycle_{self.iteration}_game_{record.index}"
            case = {
                "format_version": 1,
                "id": f"{family}_ply_{ply}",
                "family_id": family,
                "motif": "self_play_low_liberty",
                "source": {
                    "kind": "fresh_self_play_replay",
                    "iteration": self.iteration,
                    "game_index": record.index,
                    "ply": ply,
                    "selection": "liberty heuristic orders candidates; outcome unproved",
                },
                "position": {"history": deepcopy(history), "neutral": [5, 5]},
            }
            for slot in update_slots:
                slots[slot] = (priority, case)
        if (not state.result.finished() or state.result.winner != record.winner
                or state.result.reason != record.reason):
            raise ValueError(f"Game {record.index} terminal result disagrees with its history")
        # A position may fill several slots. Prefer its temporal role over the
        # generic best slot and never spend multiple calls on the same prefix.
        candidates, seen = [], set()
        for slot in _SLOT_ORDER:
            if slot not in slots:
                continue
            priority, case = slots[slot]
            if case["id"] in seen:
                continue
            seen.add(case["id"])
            case = deepcopy(case)
            case["source"]["selection_slot"] = slot
            if slot in ("atari_defense", "pre_atari_defense"):
                case["motif"] = f"self_play_{slot}"
            candidates.append((priority, case))
        if candidates:
            offset = (self.iteration + record.index) % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
        return candidates

    def cases(self):
        """Return at most max_cases replay prefixes; never return assumed labels."""
        if self._cached_cases is not None:
            return deepcopy(self._cached_cases)
        candidates = []
        for record in self._games:
            candidates.extend((rank, priority, case) for rank, (priority, case)
                              in enumerate(self._game_candidates(record)))
            # Retain every game's first rotated slot before second slots, etc.
            # A late-position priority must not erase all onset slots here.
            candidates.sort(key=lambda item: (item[0], *(-value for value in item[1])))
            del candidates[self.candidate_capacity:]
        self.candidate_count = len(candidates)
        # At most one case per game on the first pass, then a second from each
        # remaining game, etc. This keeps a single forced chase from occupying
        # the entire cycle's solver budget with neighbouring positions.
        families = {}
        for _, priority, case in candidates:
            families.setdefault(case["family_id"], []).append((priority, case))
        selected = []
        offset = self.iteration % len(_SLOT_ORDER)
        slot_order = _SLOT_ORDER[offset:] + _SLOT_ORDER[:offset]
        for depth in range(len(_SLOT_ORDER)):
            round_candidates = [rows[depth] for rows in families.values() if len(rows) > depth]
            # Keep scheduling strata diverse while ordering each stratum by
            # the actual tactical heuristic. Otherwise 32 calls for 64 sampled
            # games can still discard every rotated onset as lower-scoring.
            buckets = {slot: deque(sorted(
                (item for item in round_candidates
                 if item[1]["source"]["selection_slot"] == slot),
                key=lambda item: item[0], reverse=True)) for slot in _SLOT_ORDER}
            while any(buckets.values()):
                for slot in slot_order:
                    if not buckets[slot]:
                        continue
                    _, case = buckets[slot].popleft()
                    selected.append(case)
                    if len(selected) == self.max_cases:
                        self._cached_cases = selected
                        return deepcopy(selected)
        self._cached_cases = selected
        return deepcopy(selected)
