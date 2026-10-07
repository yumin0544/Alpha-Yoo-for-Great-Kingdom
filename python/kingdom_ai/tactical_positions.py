"""Strict, replayable tactical positions for the confirmed Kingdom rules.

Motif names describe an exercise, not a proved result. Neither console moves nor
Go terminology are accepted as policy/value targets; a solver must prove those
separately. Coordinates in JSON are one-based, unlike the engine's public API.
"""

from copy import deepcopy
import json
from pathlib import Path

import my_board_engine as engine

POSITION_FORMAT_VERSION = 1
_PLAYERS = {"black": engine.Cell.Black, "white": engine.Cell.White}
_CELLS = {".": engine.Cell.Empty, "x": engine.Cell.Black,
          "o": engine.Cell.White, "#": engine.Cell.Neutral,
          "v": engine.Cell.Neutral}
_META_KEYS = {"format_version", "id", "family_id", "motif", "source", "notes",
              "position", "expected", "checkpoints"}


def _mapping(value, allowed, required, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError(f"Invalid {label} keys")
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer between {low} and {high}")
    return value


def _point(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain [row, col]")
    row = _integer(value[0], 1, 9, f"{label} row")
    col = _integer(value[1], 1, 9, f"{label} col")
    return engine.Position(row - 1, col - 1)


def _rows(value, label, *, owners=False):
    alphabet = ".BW" if owners else ".xo#v"
    if (not isinstance(value, (list, tuple)) or len(value) != 9
            or any(not isinstance(row, str) or len(row) != 9
                   or any(char not in alphabet for char in row) for row in value)):
        raise ValueError(f"{label} must contain nine nine-character rows ({alphabet})")
    if not owners and sum(row.count("#") + row.count("v") for row in value) > 1:
        raise ValueError("A position may contain at most one neutral stone")
    return value


def _read_spec(source):
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8-sig") as stream:
            source = json.load(stream)
    spec = _mapping(source, _META_KEYS, {"format_version", "id", "position"},
                    "position specification")
    if type(spec["format_version"]) is not int or spec["format_version"] != 1:
        raise ValueError("Unsupported tactical position format version")
    for key in ("id", "family_id", "motif", "notes"):
        if key in spec and (not isinstance(spec[key], str) or not spec[key].strip()):
            raise ValueError(f"{key} must be a non-empty string")
    if "source" in spec and not isinstance(spec["source"], (str, dict)):
        raise ValueError("source must be text or an object")
    return spec


def _initial_state(position):
    _mapping(position, {"history", "neutral", "board", "to_play"}, set(), "position")
    if "history" not in position and "board" not in position:
        raise ValueError("position requires history or an analysis board")
    history = position.get("history", [])
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    if "board" in position:
        if "neutral" in position or "to_play" not in position:
            raise ValueError("Analysis boards require to_play and cannot also set neutral")
        rows = _rows(position["board"], "board")
        player = position["to_play"]
        if not isinstance(player, str) or player not in _PLAYERS:
            raise ValueError("to_play must be black or white")
        board = engine.Board([_CELLS[char] for row in rows for char in row])
        # This constructor computes initial stock/ownership. It is only for
        # newly composed exercises, never for restoring a captured game state.
        state = engine.State(board, _PLAYERS[player])
        visited = set()
        for index, cell in enumerate(board.cells):
            if cell not in (engine.Cell.Black, engine.Cell.White) or index in visited:
                continue
            point = engine.Board.position(index)
            visited.update(engine.Board.index(member) for member in board.group_at(point))
            if not board.liberties(point):
                raise ValueError("Analysis setup contains an already captured group")
    else:
        if "to_play" in position:
            raise ValueError("History positions start with Black; do not set to_play")
        neutral = position.get("neutral", [5, 5])
        state = engine.State(neutral=None if neutral is None else _point(neutral, "neutral"))
    return state, history


def _assert_state(state, expected, label):
    _mapping(expected, {"board", "ownership", "to_play", "remaining_stones",
                        "consecutive_passes", "score", "result"}, set(), label)
    if "board" in expected:
        rows = _rows(expected["board"], "expected board")
        cells = [_CELLS[char] for row in rows for char in row]
        if state.board.cells != cells:
            raise ValueError(f"{label}: board mismatch")
    if "ownership" in expected:
        rows = _rows(expected["ownership"], "expected ownership", owners=True)
        values = {".": engine.Cell.Empty, "B": engine.Cell.Black, "W": engine.Cell.White}
        if state.ownership != [values[char] for row in rows for char in row]:
            raise ValueError(f"{label}: ownership mismatch")
    if "to_play" in expected:
        player = expected["to_play"]
        if (not isinstance(player, str) or player not in _PLAYERS
                or state.to_play != _PLAYERS[player]):
            raise ValueError(f"{label}: to_play mismatch")
    if "remaining_stones" in expected:
        stocks = _mapping(expected["remaining_stones"], {"black", "white"},
                          {"black", "white"}, "remaining_stones")
        for player, cell in _PLAYERS.items():
            _integer(stocks[player], 0, 41, f"{player} stock")
            if state.remaining_stones(cell) != stocks[player]:
                raise ValueError(f"{label}: {player} stock mismatch")
    if "consecutive_passes" in expected:
        passes = _integer(expected["consecutive_passes"], 0, 2, "consecutive_passes")
        if state.consecutive_passes != passes:
            raise ValueError(f"{label}: consecutive_passes mismatch")
    if "score" in expected:
        score = _mapping(expected["score"], {"black", "white"}, {"black", "white"}, "score")
        for player in _PLAYERS:
            _integer(score[player], 0, 81, f"{player} score")
            if getattr(state.score(), player) != score[player]:
                raise ValueError(f"{label}: {player} score mismatch")
    if "result" in expected:
        result = _mapping(expected["result"], {"winner", "reason", "captured_stones"},
                          {"winner", "reason", "captured_stones"}, "result")
        reasons = {"none": engine.EndReason.None_, "capture": engine.EndReason.Capture,
                   "suicide": engine.EndReason.Suicide, "two_passes": engine.EndReason.TwoPasses}
        winner = result["winner"]
        winner_cell = (engine.Cell.Empty if winner is None else
                       _PLAYERS.get(winner) if isinstance(winner, str) else None)
        captured = _integer(result["captured_stones"], 0, 81, "captured_stones")
        if (not isinstance(result["reason"], str) or result["reason"] not in reasons
                or winner_cell is None
                or state.result.reason != reasons[result["reason"]]
                or state.result.winner != winner_cell
                or state.result.captured_stones != captured):
            raise ValueError(f"{label}: result mismatch")


def replay_position(source):
    """Return independent full-state snapshots, checking every supplied checkpoint.

    ``history`` contains one-based [row, col] pairs or the literal ``"pass"``.
    Checkpoints use a zero-based count of completed plies (initial is 0).
    A history may end in a terminal state, but no move may follow termination.
    """
    spec = _read_spec(source)
    state, history = _initial_state(spec["position"])
    checkpoints = spec.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        raise ValueError("checkpoints must be a list")
    checks = {}
    for item in checkpoints:
        _mapping(item, {"ply", "expected"}, {"ply", "expected"}, "checkpoint")
        ply = _integer(item["ply"], 0, len(history), "checkpoint ply")
        if ply in checks:
            raise ValueError("Duplicate checkpoint ply")
        checks[ply] = item["expected"]
    snapshots = [state.copy()]
    if 0 in checks:
        _assert_state(state, checks[0], "checkpoint 0")
    for ply, value in enumerate(history, 1):
        if value == "pass":
            move = engine.Move.pass_turn()
        else:
            point = _point(value, f"history ply {ply}")
            move = engine.Move.place(point.row, point.col)
        if not state.play(move).accepted():
            raise ValueError(f"History ply {ply} is rejected by the engine")
        snapshots.append(state.copy())
        if ply in checks:
            _assert_state(state, checks[ply], f"checkpoint {ply}")
    if "expected" in spec:
        _assert_state(state, spec["expected"], "final expected state")
    return snapshots


def load_position(source):
    """Load a strict position dict/JSON path without losing game history."""
    return replay_position(source)[-1]


def load_casebook(path):
    """Validate a casebook and return detached specs, not assumed training labels."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        book = json.load(stream)
    _mapping(book, {"format_version", "positions", "notes"},
             {"format_version", "positions"}, "casebook")
    if type(book["format_version"]) is not int or book["format_version"] != 1:
        raise ValueError("Unsupported tactical casebook version")
    if not isinstance(book["positions"], list) or not book["positions"]:
        raise ValueError("positions must be a non-empty list")
    ids = set()
    for spec in book["positions"]:
        _read_spec(spec)
        if spec["id"] in ids:
            raise ValueError("Duplicate tactical position id")
        ids.add(spec["id"])
        load_position(spec)
    return deepcopy(book["positions"])


def _exercise(name, motif, black, white, *, to_play="black", neutral=(5, 5), family=None):
    board = [["."] * 9 for _ in range(9)]
    if neutral is not None:
        board[neutral[0] - 1][neutral[1] - 1] = "#"
    for char, points in (("x", black), ("o", white)):
        for row, col in points:
            if board[row - 1][col - 1] != ".":
                raise ValueError("Composed exercise has overlapping stones")
            board[row - 1][col - 1] = char
    return {"format_version": 1, "id": name, "family_id": family or name,
            "motif": motif, "source": "Original composed Kingdom exercise; outcome unproved",
            "position": {"board": ["".join(row) for row in board], "to_play": to_play}}


def curriculum_positions():
    """Fresh, original Go-inspired exercises evaluated only under Kingdom rules.

    There are no copied Go problem solutions. Capture ends Kingdom immediately;
    sacrifice, ko and ordinary Go scoring are deliberately not imported.
    """
    cases = [
        _exercise("capture_corner", "atari_capture", [(1, 2)], [(1, 1)]),
        _exercise("capture_edge", "atari_capture", [(1, 3), (1, 5)], [(1, 4)]),
        _exercise("capture_interior", "atari_capture", [(1, 2), (2, 1), (3, 2)], [(2, 2)]),
        _exercise("capture_chain", "atari_capture", [(1, 2), (1, 3), (2, 1), (3, 2), (3, 3)],
                  [(2, 2), (2, 3)]),
        _exercise("extend_from_atari", "atari_defense", [(1, 2), (2, 1), (3, 2)],
                  [(2, 2)], to_play="white"),
        _exercise("counter_capture", "atari_defense", [(2, 3), (3, 2), (4, 3)],
                  [(3, 3), (2, 2), (4, 2)],
                  to_play="white"),
        _exercise("double_atari_shared_liberty", "double_atari", [(2, 2), (3, 1), (4, 2),
                  (2, 4), (3, 5), (4, 4)], [(3, 2), (3, 4)]),
        _exercise("double_atari_shared_liberty_edge", "double_atari", [(1, 2), (2, 1), (3, 2),
                  (1, 4), (2, 5), (3, 4)], [(2, 2), (2, 4)]),
        _exercise("ladder_open_board", "ladder", [(2, 3), (3, 2), (4, 2)], [(3, 3)], neutral=None,
                  family="ladder_from_upper_left"),
        _exercise("ladder_neutral_wall", "ladder", [(2, 3), (3, 2), (4, 2)], [(3, 3)],
                  family="ladder_from_upper_left"),
        _exercise("ladder_edge_finish", "ladder", [(6, 7), (7, 6), (8, 6)], [(7, 7)]),
        _exercise("ladder_lower_right", "ladder", [(5, 6), (6, 5), (7, 5)], [(6, 6)]),
        _exercise("ladder_breaker", "ladder_breaker", [(2, 3), (3, 2), (4, 2)], [(3, 3), (6, 6)],
                  neutral=None, family="ladder_from_upper_left"),
        _exercise("net_enclosure", "net", [(2, 3), (3, 2), (3, 5), (5, 3), (5, 5)],
                  [(3, 3)], neutral=None),
        _exercise("connection_escape", "connection", [(1, 2), (2, 1), (3, 2)],
                  [(2, 2), (2, 4)], to_play="white"),
        _exercise("cut_connection", "connection", [(3, 3), (3, 5)],
                  [(2, 3), (4, 3), (3, 2), (2, 5), (4, 5), (3, 6)], to_play="black"),
        _exercise("connect_before_capture", "connection", [(2, 2), (2, 4)],
                  [(1, 2), (2, 1), (3, 2)], to_play="black"),
    ]
    # Validate composed setups as well as caller-supplied cases. Constructing a
    # setup is not a claim that this arrangement arose in a legal full game.
    for case in cases:
        load_position(case)
    return cases


def user_game_two_positions():
    """Critical replay prefixes of the user's complete 26-ply capture game.

    Adjacent positions share one family id, so train/held-out splitting must keep
    the entire game together. Recorded moves are observations, not optimal labels.
    """
    history = [[3, 2], [7, 2], [2, 7], [7, 8], [3, 5], [2, 4], [1, 3], [2, 5],
               [2, 6], [1, 4], [1, 6], [3, 4], [2, 3], [3, 6], [4, 5], [4, 6],
               [4, 4], [4, 3], [5, 4], [6, 4], [5, 3], [5, 2], [6, 3], [7, 3],
               [3, 3], [6, 2]]
    return [{"format_version": 1, "id": f"user_game_2_after_ply_{ply:02d}",
             "family_id": "user_capture_2026_10_07_game_2", "motif": "ladder_chase",
             "source": "User console record, second game; continuation not minimax-proved",
             "position": {"history": deepcopy(history[:ply]), "neutral": [5, 5]}}
            for ply in range(14, 26)]
