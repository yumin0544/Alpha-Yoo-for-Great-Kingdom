"""Mine actual losing-side decisions and certify individual bad actions.

The game record is an observation, never a minimax target.  A played action is
penalized only when the authoritative solver proves that its successor is a
WIN for the opponent.  That does not prove the parent is lost, or that remaining
unclassified actions are safe.  Proof depth counts the played action too.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter

import torch
import my_board_engine as engine

from .encoding import ACTION_SIZE, PASS_ACTION, action_to_move, encode_state, move_to_action
from .tactical_positions import load_position as _load_position
from .tactics import analyze_tactics
from .training import GameData


_SLOTS = ("pre_trap", "earlier_defense", "early_error", "last_decision")
_DEPTH_LADDER = (1, 3, 5, 9, 15, 21, 29)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _name(value):
    return str(getattr(value, "name", value)).split(".")[-1].upper()


def _history(actions):
    return ["pass" if action == PASS_ACTION else [action // 9 + 1, action % 9 + 1]
            for action in actions]


@dataclass(frozen=True)
class _RecordedLoss:
    index: int
    history: tuple[int, ...]
    winner: object
    reason: object


class CaptureDefenseMiner:
    """Uniform reservoir of capture/suicide games, then stratified decisions.

    ``add_game`` never replays a state or decodes feature tensors.  ``cases``
    checks the complete authoritative replay and records pre-move histories of
    the losing colour within the final ``lookback_plies`` individual turns.
    Decisions before the final unavoidable atari are scheduled before the last
    decision; their role rotates across games/cycles to retain temporal diversity.
    At most ``2 * max_cases`` game histories and four candidates per game remain.
    """

    def __init__(self, max_cases, *, generator, iteration=0, lookback_plies=30):
        _integer(max_cases, "max_cases", 1)
        _integer(iteration, "iteration")
        _integer(lookback_plies, "lookback_plies", 1)
        if lookback_plies > 30:
            raise ValueError("lookback_plies cannot exceed 30 individual turns")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Capture mining requires a dedicated CPU torch.Generator")
        self.max_cases = max_cases
        self.max_sampled_games = 2 * max_cases
        self.candidate_capacity = len(_SLOTS) * self.max_sampled_games
        self.generator = generator
        self.iteration = iteration
        self.lookback_plies = lookback_plies
        self.seen_games = 0
        self.seen_losses = 0
        self.candidate_count = 0
        self._games = []
        self._cached_cases = None

    @property
    def sampled_games(self):
        return len(self._games)

    @property
    def sampled_game_indices(self):
        return tuple(row.index for row in self._games)

    def add_game(self, game, game_index):
        _integer(game_index, "game_index")
        if not isinstance(game, GameData):
            raise TypeError("Expected completed GameData")
        history = game.action_history
        if (not isinstance(history, tuple) or not history or len(history) > 164
                or any(type(action) is not int or not 0 <= action < ACTION_SIZE
                       for action in history)
                or len(game.samples) != len(history)):
            raise ValueError("Capture mining requires an aligned, bounded integer action history")
        if game.winner not in (engine.Cell.Black, engine.Cell.White):
            raise ValueError("Capture mining requires a completed game winner")
        if game.reason not in (engine.EndReason.Capture, engine.EndReason.Suicide,
                               engine.EndReason.TwoPasses):
            raise ValueError("Capture mining requires a completed game end reason")
        self.seen_games += 1
        if game.reason == engine.EndReason.TwoPasses:
            return
        self.seen_losses += 1
        record = _RecordedLoss(game_index, history, game.winner, game.reason)
        if len(self._games) < self.max_sampled_games:
            self._games.append(record)
        else:
            slot = int(torch.randint(self.seen_losses, (), generator=self.generator).item())
            if slot < self.max_sampled_games:
                self._games[slot] = record
        self._cached_cases = None
        self.candidate_count = 0

    def _game_candidates(self, record):
        state = engine.State()
        loser = engine.opponent(record.winner)
        decisions = []
        first_ply = max(0, len(record.history) - self.lookback_plies)
        for ply, action in enumerate(record.history):
            if state.result.finished():
                raise ValueError(f"Game {record.index} continues after termination")
            if ply >= first_ply and state.to_play == loser:
                decisions.append((ply, action))
            if not state.play(action_to_move(action)).accepted():
                raise ValueError(f"Game {record.index} action {ply + 1} is rejected")
        if (not state.result.finished() or state.result.winner != record.winner
                or state.result.reason != record.reason):
            raise ValueError(f"Game {record.index} terminal result disagrees with its history")
        if not decisions:
            return []
        # Actual decisions, not guessed danger labels. Prefer the preceding
        # move (a common four-ply trap) and a still earlier defensive decision.
        selected = {"pre_trap": max(0, len(decisions) - 2),
                    "earlier_defense": max(0, len(decisions) - 4),
                    "early_error": 0, "last_decision": len(decisions) - 1}
        candidates, seen = [], set()
        family = f"capture_cycle_{self.iteration}_game_{record.index}"
        for slot in _SLOTS:
            ply, action = decisions[selected[slot]]
            if ply in seen:
                continue
            seen.add(ply)
            candidates.append({
                "format_version": 1, "id": f"{family}_before_ply_{ply + 1}",
                "family_id": family, "motif": f"capture_defense_{slot}",
                "position": {"history": _history(record.history[:ply]), "neutral": [5, 5]},
                "source": {"kind": "fresh_capture_loss_replay", "iteration": self.iteration,
                           "game_index": record.index, "ply": ply, "decision_ply": ply + 1,
                           "plies_to_end": len(record.history) - ply, "bad_action": action,
                           "loser": loser.name, "recorded_end_reason": record.reason.name,
                           "selection_slot": slot,
                           "selection": "Observed losing-side decision; action outcome unproved"},
            })
        offset = (self.iteration + record.index) % len(candidates)
        return candidates[offset:] + candidates[:offset]

    def cases(self):
        if self._cached_cases is not None:
            return deepcopy(self._cached_cases)
        families = [self._game_candidates(record) for record in self._games]
        self.candidate_count = sum(map(len, families))
        selected = []
        for rank in range(len(_SLOTS)):
            buckets = {slot: deque(rows[rank] for rows in families
                                   if len(rows) > rank
                                   and rows[rank]["source"]["selection_slot"] == slot)
                       for slot in _SLOTS}
            offset = self.iteration % len(_SLOTS)
            order = _SLOTS[offset:] + _SLOTS[:offset]
            while any(buckets.values()):
                for slot in order:
                    if buckets[slot]:
                        selected.append(buckets[slot].popleft())
                    if len(selected) == self.max_cases:
                        self._cached_cases = selected
                        return deepcopy(selected)
        self._cached_cases = selected
        return deepcopy(selected)


def collect_capture_defense_samples(cases, options, *, max_cases=32,
                                   generation_seconds=30.0, bound_remaining_time=True,
                                   load_position=None, solver=None):
    """Certify observed actions, returning negative-policy rows and an audit.

    Each case has one total node budget and one total wall-clock solver budget,
    shared by every iterative-depth call. The original action counts as one ply,
    so the successor's search never exceeds ``max_depth - 1`` (at most 29).
    Remaining overall time bounds each solver call by default. State loading and
    OS scheduling overhead are not a hard real-time guarantee. Fully immediate-
    losing parent states are excluded: their final move is not a useful defense
    choice to penalize. UNKNOWN never creates a sample or a parent value label.
    """
    from .capture_defense import CertifiedDefenseSample

    _integer(max_cases, "max_cases", 1)
    if not isinstance(options, engine.TacticalSolverOptions):
        raise TypeError("Capture certification requires engine TacticalSolverOptions")
    _integer(options.max_depth, "max_depth", 1)
    _integer(options.max_nodes, "max_nodes", 1)
    _integer(options.time_limit_ms, "time_limit_ms")
    if options.max_depth > 30:
        raise ValueError("Capture certification max_depth cannot exceed 30 individual turns")
    if type(bound_remaining_time) is not bool:
        raise ValueError("bound_remaining_time must be a bool")
    if (isinstance(generation_seconds, bool)
            or not isinstance(generation_seconds, (int, float))
            or not math.isfinite(generation_seconds) or generation_seconds <= 0):
        raise ValueError("generation_seconds must be finite and positive")
    load_position = _load_position if load_position is None else load_position
    solver = engine.solve_tactics if solver is None else solver
    started = perf_counter()
    samples, records, seen = [], [], set()
    for case in list(cases)[:max_cases]:
        if perf_counter() - started >= generation_seconds:
            break
        case_started = perf_counter()
        if not isinstance(case, dict):
            raise TypeError("Each capture case must be a dictionary")
        case_id, family = case.get("id"), case.get("family_id", case.get("id"))
        if (type(case_id) is not str or not case_id or case_id in seen
                or type(family) is not str or not family):
            raise ValueError("Capture cases require distinct IDs and nonempty family IDs")
        seen.add(case_id)
        source = case.get("source")
        if not isinstance(source, dict):
            raise ValueError("Capture cases require replay provenance")
        action = source.get("bad_action")
        if type(action) is not int or not 0 <= action < ACTION_SIZE:
            raise ValueError("Capture case bad_action must be a legal integer action")
        if source.get("loser") not in ("Black", "White"):
            raise ValueError("Capture case loser must be Black or White")
        if not isinstance(case.get("position"), dict) or "history" not in case["position"]:
            raise ValueError("Capture cases require complete replay histories")
        state = load_position(case)
        if not isinstance(state, engine.State):
            raise TypeError("Position loader must return an engine State")
        if state.result.finished() or state.to_play.name != source["loser"]:
            raise ValueError("Capture prefix must be nonterminal and the losing colour's turn")
        encoded = encode_state(state)
        if not bool(encoded.legal_mask[action]):
            raise ValueError("Recorded capture action is not legal at its prefix")
        record = {"id": case_id, "family_id": family, "source": deepcopy(source),
                  "position": deepcopy(case["position"]), "actor": state.to_play.name,
                  "selected_action": action, "outcome": "UNKNOWN", "training_label": False,
                  "max_depth": options.max_depth, "nodes": 0, "calls": [],
                  "budget_exhausted": False}
        if not analyze_tactics(state).safe_actions:
            record["outcome"] = "ALREADY_FORCED_LOSS"
            records.append(record)
            continue
        after = state.copy()
        if not after.play(action_to_move(action)).accepted():
            raise ValueError("Engine rejected the recorded legal capture action")
        certified_depth = None
        if after.result.finished():
            record["terminal_reason"] = after.result.reason.name
            if after.result.winner != state.to_play:
                certified_depth = 1
                record["successor_outcome"] = "WIN"
                record["certificate_source"] = "terminal_opponent_win"
            else:
                record["outcome"] = "ACTION_WIN"
        else:
            limit = options.max_depth - 1
            ladder = sorted(set(min(depth, limit) for depth in _DEPTH_LADDER)) if limit else []
            for depth in ladder:
                used = record["nodes"]
                remaining_nodes = options.max_nodes - used
                remaining_case_ms = (int(options.time_limit_ms
                                         - (perf_counter() - case_started) * 1000)
                                     if options.time_limit_ms else 2 ** 31 - 1)
                remaining_global_ms = int((generation_seconds - (perf_counter() - started)) * 1000)
                remaining_ms = min(remaining_case_ms, remaining_global_ms
                                   if bound_remaining_time else 2 ** 31 - 1)
                if remaining_nodes < 1 or remaining_ms < 1:
                    record["budget_exhausted"] = True
                    break
                stage_options = engine.TacticalSolverOptions(
                    max_depth=depth, max_nodes=remaining_nodes, time_limit_ms=remaining_ms)
                result = solver(after.copy(), stage_options)
                outcome = _name(result.outcome)
                if outcome not in ("WIN", "LOSS", "UNKNOWN"):
                    raise ValueError("Unsupported capture proof outcome")
                nodes = result.nodes
                proof_depth = result.proof_depth
                if (type(nodes) is not int or not 0 <= nodes <= remaining_nodes
                        or type(proof_depth) is not int or not 0 <= proof_depth <= depth
                        or not math.isfinite(result.elapsed_ms) or result.elapsed_ms < 0):
                    raise ValueError("Invalid capture proof budget metadata")
                record["nodes"] += nodes
                record["calls"].append({"max_depth": depth, "max_nodes": remaining_nodes,
                                        "time_limit_ms": remaining_ms, "outcome": outcome,
                                        "nodes": nodes, "proof_depth": proof_depth,
                                        "elapsed_ms": result.elapsed_ms,
                                        "budget_exhausted": bool(result.budget_exhausted)})
                record["successor_outcome"] = outcome
                record["budget_exhausted"] |= bool(result.budget_exhausted)
                if outcome in ("WIN", "LOSS"):
                    variation = list(result.principal_variation)
                    if not variation or proof_depth < 1 or len(variation) != proof_depth:
                        raise ValueError("A nonterminal proof needs a continuation matching its depth")
                    replayed = after.copy()
                    for move in variation:
                        if not replayed.play(move).accepted():
                            raise ValueError("Capture proof continuation contains an illegal move")
                    if (not replayed.result.finished()
                            or (replayed.result.winner == after.to_play) != (outcome == "WIN")):
                        raise ValueError("Capture proof continuation contradicts its outcome")
                    record["principal_variation"] = [move_to_action(move) for move in variation]
                    record["pv_end_reason"] = replayed.result.reason.name
                    if outcome == "WIN":
                        certified_depth = proof_depth + 1
                    else:
                        record["outcome"] = "ACTION_WIN"
                    break
        if certified_depth is not None:
            losing_mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
            losing_mask[action] = True
            if not bool((encoded.legal_mask & ~losing_mask).any()):
                record["outcome"] = "ALL_LEGAL_BAD"
            else:
                record.update({"outcome": "ACTION_LOSS", "training_label": True,
                               "losing_actions": [action], "proof_depth": certified_depth,
                               "target_kind": "negative_policy_only"})
                samples.append(CertifiedDefenseSample(case_id, family,
                    encoded.features.detach().cpu().clone(), encoded.legal_mask.cpu().clone(),
                    losing_mask, deepcopy(record)))
        records.append(record)
    return samples, {"cases_available": len(cases), "cases_solved": len(records),
                     "certified_samples": len(samples),
                     "excluded_unknown": sum(row["outcome"] == "UNKNOWN" for row in records),
                     "excluded_already_forced": sum(row["outcome"] == "ALREADY_FORCED_LOSS"
                                                    for row in records),
                     "excluded_action_win": sum(row["outcome"] == "ACTION_WIN" for row in records),
                     "generation_seconds": perf_counter() - started, "records": records}
