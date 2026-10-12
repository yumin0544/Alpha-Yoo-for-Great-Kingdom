"""Read-only, participant-specific loss accounting for model-match JSONL.

An ending is not a loss count: every Capture ending is a win for one model
and a loss for the other. Keep capture, suicide and territory/pass losses
separate, and retain both denominators so a safer-looking pass policy cannot
hide a deterioration in total wins. Only complete, internally consistent
schema-1 match logs are accepted.
"""

from copy import deepcopy
import json
import math
from pathlib import Path


COLORS = ("Black", "White")
REASONS = ("Capture", "Suicide", "TwoPasses")
LOSS_FIELDS = {"Capture": "capture_losses", "Suicide": "suicide_losses",
               "TwoPasses": "pass_losses"}


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _choice(value, choices, field):
    if type(value) is not str or value not in choices:
        raise ValueError(f"{field} must be one of {', '.join(choices)}")
    return value


def _real(value, field):
    if (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite number")
    return value


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_game(game, expected_games, seed):
    index = _integer(game.get("index"), "game.index", 1)
    if index > expected_games:
        raise ValueError("game.index exceeds protocol.games")
    pair_index = _integer(game.get("pair_index"), "game.pair_index", 1)
    if pair_index != (index + 1) // 2:
        raise ValueError("game.pair_index is inconsistent with game.index")
    game_seed = _integer(game.get("seed"), "game.seed")
    if game_seed != (seed + pair_index - 1) % (2 ** 64):
        raise ValueError("game.seed is inconsistent with the paired protocol")
    color = _choice(game.get("model_a_color"), COLORS, "game.model_a_color")
    if color != ("Black" if index % 2 else "White"):
        raise ValueError("game.model_a_color is inconsistent with color pairing")
    winner = _choice(game.get("winner"), ("a", "b"), "game.winner")
    winner_color = _choice(game.get("winner_color"), COLORS, "game.winner_color")
    expected_winner_color = color if winner == "a" else COLORS[1 - COLORS.index(color)]
    if winner_color != expected_winner_color:
        raise ValueError("game.winner_color is inconsistent with game.winner")
    _choice(game.get("reason"), REASONS, "game.reason")
    plies = _integer(game.get("plies"), "game.plies", 1)
    actions = game.get("actions")
    if not isinstance(actions, list) or len(actions) != plies:
        raise ValueError("game.actions must contain exactly game.plies actions")
    if any(type(action) is not int or not 0 <= action <= 81 for action in actions):
        raise ValueError("game.actions must contain integer actions in [0, 81]")
    territory = game.get("territory")
    if not isinstance(territory, dict) or set(territory) != {"black", "white"}:
        raise ValueError("game.territory must contain black and white counts")
    for key in ("black", "white"):
        _integer(territory[key], f"game.territory.{key}")
    board = game.get("board")
    if (not isinstance(board, list) or len(board) != 9
            or any(type(row) is not str or len(row) != 9
                   or set(row) - set("xo#BW.") for row in board)):
        raise ValueError("game.board must contain nine valid nine-point rows")
    return index


def _validate_summary(summary, games, expected_games):
    if summary.get("status") != "complete":
        raise ValueError("Only a complete match summary is accepted")
    if len(games) != expected_games:
        raise ValueError("Incomplete match: game count differs from protocol.games")
    wins_a = sum(game["winner"] == "a" for game in games)
    expected = {
        "games": expected_games, "wins_a": wins_a, "wins_b": expected_games - wins_a,
        "wins_a_as_black": sum(game["winner"] == "a" and game["model_a_color"] == "Black"
                               for game in games),
        "wins_a_as_white": sum(game["winner"] == "a" and game["model_a_color"] == "White"
                               for game in games),
        "total_plies": sum(game["plies"] for game in games),
    }
    for field, value in expected.items():
        if _integer(summary.get(field), f"summary.{field}") != value:
            raise ValueError(f"summary.{field} disagrees with the game records")
    if not math.isclose(_real(summary.get("win_rate_a"), "summary.win_rate_a"),
                        wins_a / expected_games, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("summary.win_rate_a disagrees with the game records")
    endings = summary.get("endings")
    if not isinstance(endings, dict) or set(endings) - set(REASONS):
        raise ValueError("summary.endings must contain known ending reasons")
    expected_endings = {reason: sum(game["reason"] == reason for game in games)
                        for reason in REASONS}
    for reason in REASONS:
        if _integer(endings.get(reason, 0), f"summary.endings.{reason}") != expected_endings[reason]:
            raise ValueError(f"summary.endings.{reason} disagrees with the game records")


def load_match_records(path):
    """Validate a completed schema-1 log without loading models or replaying games.

    Parallel match workers can emit games in any order. Their indices must still
    cover the complete paired series once each; duplicated/truncated records,
    aborted matches and contradictory aggregate summaries are errors.
    """
    source = Path(path)
    session = summary = None
    games = []
    seen = set()
    expected_games = seed = None
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                event = json.loads(line, object_pairs_hook=_unique_keys)
                if not isinstance(event, dict):
                    raise ValueError("Every JSONL record must be an object")
                kind = event.get("type")
                if summary is not None:
                    raise ValueError("Records after the complete summary are not allowed")
                if kind == "session":
                    if session is not None or games:
                        raise ValueError("A single session must be the first record")
                    if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
                        raise ValueError("Only model-match schema_version 1 is supported")
                    participants = event.get("participants")
                    if (not isinstance(participants, list) or len(participants) != 2
                            or any(not isinstance(item, dict) for item in participants)
                            or [item.get("id") for item in participants] != ["a", "b"]
                            or any(type(item.get("name")) is not str or not item["name"].strip()
                                   for item in participants)):
                        raise ValueError("session.participants must identify named models a and b")
                    protocol = event.get("protocol")
                    if not isinstance(protocol, dict):
                        raise ValueError("session.protocol must be an object")
                    expected_games = _integer(protocol.get("games"), "protocol.games", 2)
                    if expected_games % 2:
                        raise ValueError("protocol.games must be even for color pairing")
                    seed = _integer(protocol.get("seed"), "protocol.seed")
                    if seed >= 2 ** 64:
                        raise ValueError("protocol.seed must be an unsigned 64-bit integer")
                    session = event
                elif kind == "game":
                    if session is None:
                        raise ValueError("Game records require a preceding session")
                    index = _validate_game(event, expected_games, seed)
                    if index in seen:
                        raise ValueError(f"Duplicate game.index: {index}")
                    seen.add(index)
                    games.append(event)
                elif kind == "summary":
                    if session is None:
                        raise ValueError("Summary requires a preceding session")
                    _validate_summary(event, games, expected_games)
                    summary = event
                elif kind == "aborted":
                    raise ValueError("Aborted match is not a complete strength evaluation")
                else:
                    raise ValueError(f"Unknown model-match record type: {kind!r}")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    if session is None or summary is None:
        raise ValueError(f"{source}: A session and complete summary are required")
    # Unique in-range indices plus the expected count imply full coverage.
    return deepcopy(session), sorted(games, key=lambda game: game["index"])


def _counts(games, model):
    total = len(games)
    wins = sum(game["winner"] == model for game in games)
    losses = total - wins
    counts = {field: sum(game["winner"] != model and game["reason"] == reason
                         for game in games) for reason, field in LOSS_FIELDS.items()}
    counts["capture_or_suicide_losses"] = counts["capture_losses"] + counts["suicide_losses"]
    return {"games": total, "wins": wins, "losses": losses,
            "win_rate": wins / total if total else None, **counts,
            "loss_rates_all_games": {key: value / total if total else None
                                     for key, value in counts.items()},
            "loss_shares_all_losses": {key: value / losses if losses else None
                                       for key, value in counts.items()}}


def analyze_capture_losses(path):
    """Return both models' outcome metrics and color-separated counts/rates.

    Rates are fractions in [0, 1], not percentages. A share with a zero loss
    denominator is None rather than a misleading zero. This report measures
    observed outcomes only and cannot establish that future capture risk is zero.
    """
    session, games = load_match_records(path)
    models = {}
    for model, participant in zip(("a", "b"), session["participants"]):
        metrics = _counts(games, model)
        metrics["id"] = model
        metrics["name"] = participant["name"]
        metrics["colors"] = {
            color: _counts([game for game in games if
                            (game["model_a_color"] == color) == (model == "a")], model)
            for color in COLORS
        }
        models[model] = metrics
    return {"schema_version": 1, "source": str(Path(path).resolve()),
            "status": "complete", "participants": session["participants"],
            "protocol": session["protocol"], "models": models}
