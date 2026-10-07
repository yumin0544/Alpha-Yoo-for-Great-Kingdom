"""Summarize old and current Trainer JSONL metrics without loading checkpoints."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median


def compact_metric_row(row):
    """Keep learning metrics, replacing duplicated teacher histories with counts.

    The caller's iteration/checkpoint payload is untouched. Full accepted proof
    certificates remain in the certified replay; this is only a JSONL view.
    """
    compact = deepcopy(row)
    report = compact.get("online_tactics")
    if not isinstance(report, dict) or not report.get("records"):
        return compact
    records = report.pop("records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Online tactics records must be a list of objects")
    outcomes, accepted, exhausted, reasons = Counter(), Counter(), Counter(), Counter()
    depths, accepted_depths = defaultdict(Counter), defaultdict(Counter)
    motifs, actors = defaultdict(Counter), defaultdict(Counter)
    for record in records:
        outcome = record.get("outcome", "UNKNOWN")
        depth = str(record.get("proof_depth", 0))
        outcomes[outcome] += 1
        depths[outcome][depth] += 1
        if record.get("budget_exhausted", False):
            exhausted[outcome] += 1
        reason = record.get("excluded") or record.get("excluded_reason")
        if reason:
            reasons[reason] += 1
        if record.get("training_label", False):
            accepted[outcome] += 1
            accepted_depths[outcome][depth] += 1
            motifs[outcome][record.get("motif", "unknown")] += 1
            actors[outcome][record.get("actor", "unknown")] += 1
    report["record_summary"] = {
        "records": len(records), "outcomes": dict(outcomes),
        "accepted": dict(accepted), "proof_depths": dict(depths),
        "accepted_proof_depths": dict(accepted_depths),
        "accepted_motifs": dict(motifs), "accepted_actors": dict(actors),
        "budget_exhausted": dict(exhausted), "exclusion_reasons": dict(reasons),
    }
    return compact


def load_metric_rows(path) -> tuple[list[dict], int]:
    """Load iteration rows, keeping the last copy of a repeated iteration."""
    source = Path(path)
    by_iteration = {}
    parsed_rows = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Metrics line {line_number} must contain an object")
            iteration = row.get("iteration")
            if type(iteration) is not int or iteration < 1:
                # Benchmark JSONL files can contain environment/summary records.
                continue
            if not _finite_number(row.get("generated_samples"), positive=True):
                raise ValueError(f"Metrics line {line_number} has invalid generated_samples")
            if type(row.get("replay_size")) is not int or row["replay_size"] < 1:
                raise ValueError(f"Metrics line {line_number} has invalid replay_size")
            if not _finite_number(row.get("mean_self_play_plies"), positive=True):
                raise ValueError(f"Metrics line {line_number} has invalid mean_self_play_plies")
            evaluation = row.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError(f"Metrics line {line_number} has invalid evaluation")
            evaluation_games = evaluation.get("games")
            integer_fields = ("wins", "losses", "wins_as_black", "wins_as_white")
            if (type(evaluation_games) is not int or evaluation_games < 2
                    or evaluation_games % 2
                    or any(type(evaluation.get(name)) is not int
                           or evaluation[name] < 0 for name in integer_fields)
                    or evaluation["wins"] + evaluation["losses"] != evaluation_games
                    or (evaluation["wins_as_black"] + evaluation["wins_as_white"]
                        != evaluation["wins"])):
                raise ValueError(f"Metrics line {line_number} has invalid evaluation counts")
            configured_games = row.get("games_per_iteration")
            if configured_games is not None and (
                    type(configured_games) is not int or configured_games < 1):
                raise ValueError(f"Metrics line {line_number} has invalid games_per_iteration")
            inferred_games = row["generated_samples"] / row["mean_self_play_plies"]
            if configured_games is None and not math.isclose(
                    inferred_games, round(inferred_games), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Metrics line {line_number} cannot infer completed games")
            for name in ("self_play_seconds", "elapsed_seconds"):
                if not _finite_number(row.get(name), positive=True):
                    raise ValueError(f"Metrics line {line_number} has invalid {name}")
            for name in ("training_seconds", "evaluation_seconds"):
                if not _finite_number(row.get(name), positive=False):
                    raise ValueError(f"Metrics line {line_number} has invalid {name}")
            for name in ("self_play_compute_seconds", "replay_store_seconds",
                         "checkpoint_seconds", "initial_checkpoint_seconds"):
                if name in row and not _finite_number(row[name], positive=False):
                    raise ValueError(f"Metrics line {line_number} has invalid {name}")
            if ("evaluation_workers" in row
                    and (type(row["evaluation_workers"]) is not int
                         or row["evaluation_workers"] < 1)):
                raise ValueError(
                    f"Metrics line {line_number} has invalid evaluation_workers"
                )
            if "checkpoint_written" in row and type(row["checkpoint_written"]) is not bool:
                raise ValueError(f"Metrics line {line_number} has invalid checkpoint_written")
            _validate_online_tactics(row, prefix=f"Metrics line {line_number}")
            by_iteration[iteration] = row
            parsed_rows += 1
    if not by_iteration:
        raise ValueError("No Trainer iteration rows were found")
    rows = [by_iteration[index] for index in sorted(by_iteration)]
    return rows, parsed_rows - len(rows)


def _finite_number(value, *, positive):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _validate_online_tactics(row, *, prefix="Metrics row"):
    if ("online_tactics_seconds" in row
            and not _finite_number(row["online_tactics_seconds"], positive=False)):
        raise ValueError(f"{prefix} has invalid online_tactics_seconds")
    if ("online_tactics_enabled" in row
            and type(row["online_tactics_enabled"]) is not bool):
        raise ValueError(f"{prefix} has invalid online_tactics_enabled")
    if "online_tactics" not in row:
        return
    report = row["online_tactics"]
    if not isinstance(report, dict):
        raise ValueError(f"{prefix} has invalid online_tactics report")
    if "enabled" in report and type(report["enabled"]) is not bool:
        raise ValueError(f"{prefix} has invalid online_tactics.enabled")
    for name in ("added_samples", "excluded_shallow", "replay_size", "mixed_updates",
                 "tactical_rows_per_batch", "replay_rows_per_batch"):
        if name in report and (type(report[name]) is not int or report[name] < 0):
            raise ValueError(f"{prefix} has invalid online_tactics.{name}")
    if "seconds" in report and not _finite_number(report["seconds"], positive=False):
        raise ValueError(f"{prefix} has invalid online_tactics.seconds")


def _online_tactics_summary(rows):
    reports = [row["online_tactics"] for row in rows if "online_tactics" in row]
    latest = rows[-1].get("online_tactics", {})
    return {
        "timing_covered_iterations": sum("online_tactics_seconds" in row for row in rows),
        "report_covered_iterations": len(reports),
        "enabled_iterations": sum(report.get("enabled", False) for report in reports),
        "added_samples": sum(report.get("added_samples", 0) for report in reports),
        "excluded_shallow": sum(report.get("excluded_shallow", 0) for report in reports),
        "mixed_updates": sum(report.get("mixed_updates", 0) for report in reports),
        "tactical_training_draws": sum(report.get("mixed_updates", 0)
                                       * report.get("tactical_rows_per_batch", 0)
                                       for report in reports),
        "latest_replay_size": latest.get("replay_size"),
        "latest_tactical_rows_per_batch": latest.get("tactical_rows_per_batch"),
        "latest_replay_rows_per_batch": latest.get("replay_rows_per_batch"),
    }


def _games(row):
    configured = row.get("games_per_iteration")
    if type(configured) is int and configured > 0:
        return configured
    mean_plies = row.get("mean_self_play_plies")
    if _finite_number(mean_plies, positive=True):
        return int(round(row["generated_samples"] / mean_plies))
    # load_metric_rows validates mean_self_play_plies for file-backed data.
    return int(round(row["generated_samples"] / row["mean_self_play_plies"]))


def _window_summary(rows):
    samples = sum(row["generated_samples"] for row in rows)
    games = sum(_games(row) for row in rows)
    self_play_seconds = sum(row["self_play_seconds"] for row in rows)
    elapsed_seconds = sum(row["elapsed_seconds"] for row in rows)
    position_rates = [row["generated_samples"] / row["self_play_seconds"] for row in rows]
    game_rates = [_games(row) / row["self_play_seconds"] for row in rows if _games(row)]
    measured_with_checkpoint = [
        row.get("elapsed_with_checkpoint_seconds", row["elapsed_seconds"]) for row in rows
    ]
    return {
        "iterations": len(rows),
        "games": games,
        "generated_samples": samples,
        "mean_self_play_plies": samples / games if games else None,
        "positions_per_self_play_second": samples / self_play_seconds,
        "median_iteration_positions_per_second": median(position_rates),
        "games_per_self_play_second": games / self_play_seconds if games else None,
        "median_iteration_games_per_second": median(game_rates) if game_rates else None,
        "measured_seconds": elapsed_seconds,
        "median_iteration_seconds": median(measured_with_checkpoint),
    }


def _evaluation_summary(rows):
    games = sum(row.get("evaluation", {}).get("games", 0) for row in rows)
    wins = sum(row.get("evaluation", {}).get("wins", 0) for row in rows)
    wins_as_black = sum(row.get("evaluation", {}).get("wins_as_black", 0) for row in rows)
    wins_as_white = sum(row.get("evaluation", {}).get("wins_as_white", 0) for row in rows)
    games_per_color = games / 2 if games else 0
    promotions = sum(bool(row.get("promoted")) for row in rows)
    return {
        "games": games,
        "candidate_wins": wins,
        "candidate_win_rate": wins / games if games else None,
        "wins_as_black": wins_as_black,
        "wins_as_white": wins_as_white,
        "black_win_rate": wins_as_black / games_per_color if games_per_color else None,
        "white_win_rate": wins_as_white / games_per_color if games_per_color else None,
        "black_white_gap": ((wins_as_black - wins_as_white) / games_per_color
                            if games_per_color else None),
        "promotions": promotions,
        "promotion_rate": promotions / len(rows),
    }


def _replay_capacity(rows):
    explicit = {row["replay_capacity"] for row in rows
                if type(row.get("replay_capacity")) is int and row["replay_capacity"] > 0}
    if len(explicit) > 1:
        raise ValueError("Metrics rows contain conflicting replay capacities")
    if explicit:
        return explicit.pop(), "metrics"
    previous_size = None
    inferred = set()
    for row in rows:
        size = row.get("replay_size")
        generated = row.get("generated_samples")
        if type(size) is not int or size < 1:
            previous_size = None
            continue
        expected = generated if previous_size is None else previous_size + generated
        if size < expected:
            inferred.add(size)
        previous_size = size
    if len(inferred) > 1:
        raise ValueError("Metrics rows imply conflicting replay capacities")
    return ((inferred.pop(), "inferred_from_eviction") if inferred else (None, None))


def summarize_metrics(rows, *, recent=20, target_iterations=None, duplicate_rows=0):
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    if type(recent) is not int or recent < 1:
        raise ValueError("recent must be a positive integer")
    if target_iterations is not None and (
            type(target_iterations) is not int or target_iterations < 1):
        raise ValueError("target_iterations must be a positive integer")
    ordered = sorted(rows, key=lambda row: row["iteration"])
    for row in ordered:
        _validate_online_tactics(row)
    recent_rows = ordered[-recent:]
    phase_seconds = {
        "self_play": sum(row["self_play_seconds"] for row in ordered),
        "online_tactics": sum(row.get("online_tactics_seconds", 0.0) for row in ordered),
        "training": sum(row["training_seconds"] for row in ordered),
        "evaluation": sum(row["evaluation_seconds"] for row in ordered),
    }
    phase_total = sum(phase_seconds.values())
    checkpoint_rows = [
        row for row in ordered
        if row.get("checkpoint_written", bool(row.get("checkpoint_bytes")))
    ]
    checkpoint_seconds = sum(
        row.get("checkpoint_seconds", 0.0) + row.get("initial_checkpoint_seconds", 0.0)
        for row in checkpoint_rows
    )
    self_play_detail_rows = [
        row for row in ordered
        if "self_play_compute_seconds" in row and "replay_store_seconds" in row
    ]
    detail_compute_seconds = sum(
        row["self_play_compute_seconds"] for row in self_play_detail_rows
    )
    detail_replay_seconds = sum(
        row["replay_store_seconds"] for row in self_play_detail_rows
    )
    detail_self_play_seconds = sum(
        row["self_play_seconds"] for row in self_play_detail_rows
    )
    winner_rows = [row for row in ordered if isinstance(row.get("self_play_winners"), dict)]
    self_play_black = sum(row["self_play_winners"].get("Black", 0) for row in winner_rows)
    self_play_white = sum(row["self_play_winners"].get("White", 0) for row in winner_rows)
    latest = ordered[-1]
    recent_summary = _window_summary(recent_rows)
    remaining = None
    if target_iterations is not None:
        count = max(0, target_iterations - latest["iteration"])
        remaining = {
            "target_iterations": target_iterations,
            "remaining_iterations": count,
            "estimated_seconds": count * recent_summary["median_iteration_seconds"],
            "includes_checkpoint_where_available": bool(checkpoint_rows),
        }
    replay_capacity, capacity_source = _replay_capacity(ordered)
    generated_recent = median(row["generated_samples"] for row in recent_rows)
    replay_size = latest.get("replay_size")
    replay_window = (replay_size / generated_recent
                     if _finite_number(replay_size, positive=True) else None)
    return {
        "first_iteration": ordered[0]["iteration"],
        "last_iteration": latest["iteration"],
        "rows": len(ordered),
        "duplicate_rows_replaced": duplicate_rows,
        "overall": _window_summary(ordered),
        "recent": recent_summary,
        "phase_seconds": phase_seconds,
        "online_tactics": _online_tactics_summary(ordered),
        "phase_fractions": {
            **{name: seconds / sum(row["elapsed_seconds"] for row in ordered)
               for name, seconds in phase_seconds.items()},
            "unattributed": max(
                0.0, sum(row["elapsed_seconds"] for row in ordered) - phase_total
            ) / sum(row["elapsed_seconds"] for row in ordered),
        },
        "checkpoint": {
            "rows_with_timing": len(checkpoint_rows),
            "seconds": checkpoint_seconds,
            "latest_bytes": latest.get("checkpoint_bytes"),
        },
        "self_play_detail": {
            "covered_iterations": len(self_play_detail_rows),
            "compute_seconds": detail_compute_seconds,
            "replay_store_seconds": detail_replay_seconds,
            "compute_fraction": (
                detail_compute_seconds / detail_self_play_seconds
                if detail_self_play_seconds else None
            ),
            "replay_store_fraction": (
                detail_replay_seconds / detail_self_play_seconds
                if detail_self_play_seconds else None
            ),
        },
        "evaluation": {**_evaluation_summary(ordered),
                       "recent": _evaluation_summary(recent_rows),
                       "latest_workers": latest.get("evaluation_workers"),
                       "worker_timing_covered_iterations": sum(
                           "evaluation_workers" in row for row in ordered
                       )},
        "self_play_colors": {
            "covered_iterations": len(winner_rows),
            "black_wins": self_play_black,
            "white_wins": self_play_white,
        },
        "replay": {
            "latest_size": replay_size,
            "latest_capacity": replay_capacity,
            "capacity_source": capacity_source,
            "estimated_recent_iteration_window": replay_window,
            "latest_turnover": latest.get("replay_turnover"),
            "latest_training_draws_per_generated_sample": latest.get(
                "training_draws_per_generated_sample"),
            "latest_training_draws_per_replay_sample": latest.get(
                "training_draws_per_replay_sample"),
        },
        "projection": remaining,
    }
