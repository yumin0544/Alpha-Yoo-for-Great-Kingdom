"""Read-only historical opponents and a two-stage champion promotion gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

from .checkpoint import _file_digest, _weights_digest, load_model
from .evaluation import EvaluationResult


class PromotionLeague:
    """Pin best.pt files; compare both actors against exactly the same pool.

    No archive files are written. Incumbent results are reusable only for the
    same weights, opponents, seeds and complete evaluation/search settings.
    """

    def __init__(self, directory, *, games=100, seed=42, state=None):
        if type(games) is not int or games < 2 or games % 2:
            raise ValueError("Promotion games must be even and at least two")
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise ValueError("Promotion seed must be an unsigned 64-bit integer")
        self.directory = Path(directory).resolve() if directory is not None else None
        self.games, self.seed = games, seed
        self.opponents, self.models = [], []
        self.reference_cache = None
        if self.directory is not None:
            self.opponents = self._discover()
            for opponent in self.opponents:
                self.models.append(load_model(opponent["path"]))
            self.check_unchanged()
        if state is not None:
            self._restore(state)

    def _discover(self):
        opponents = []
        for folder in self.directory.iterdir():
            match = re.fullmatch(r"champion_(\d+)v", folder.name)
            if not folder.is_dir() or match is None:
                continue
            path = (folder / "best.pt").resolve()
            opponents.append({"version": int(match.group(1)), "path": str(path),
                              "sha256": _file_digest(path)})
        opponents.sort(key=lambda row: row["version"])
        if not opponents:
            raise ValueError("Promotion archive needs at least one champion_<version>v/best.pt")
        if len({row["version"] for row in opponents}) != len(opponents):
            raise ValueError("Promotion archive contains duplicate champion versions")
        return opponents

    def check_unchanged(self):
        if self.directory is not None and self._discover() != self.opponents:
            raise ValueError("Promotion archive changed; explicitly reconfigure a new run to accept it")

    def state_dict(self):
        return deepcopy({"version": 1, "opponents": self.opponents,
                         "reference_cache": self.reference_cache})

    def _restore(self, state):
        if (not isinstance(state, dict)
                or set(state) != {"version", "opponents", "reference_cache"}
                or type(state["version"]) is not int or state["version"] != 1
                or not isinstance(state["opponents"], list)
                or json.dumps(state["opponents"], sort_keys=True) != json.dumps(self.opponents, sort_keys=True)):
            raise ValueError("Invalid or changed checkpoint promotion archive")
        cache = state["reference_cache"]
        if cache is not None:
            if (self.directory is None or not isinstance(cache, dict)
                    or set(cache) != {"signature", "results"}
                    or type(cache["signature"]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", cache["signature"]) is None
                    or not isinstance(cache["results"], list)
                    or len(cache["results"]) != len(self.opponents)):
                raise ValueError("Invalid checkpoint promotion reference cache")
            for opponent, row in zip(self.opponents, cache["results"]):
                if (not isinstance(row, dict) or set(row) != {"version", "seed", "evaluation"}
                        or type(row["version"]) is not int or row["version"] != opponent["version"]
                        or type(row["seed"]) is not int or row["seed"] != self._seed(opponent)):
                    raise ValueError("Invalid checkpoint promotion reference result")
                self._validate_result(row["evaluation"])
        self.reference_cache = deepcopy(cache)

    def _seed(self, opponent):
        return (self.seed + opponent["version"]) % (2 ** 64)

    def _validate_result(self, row):
        fields = {"games", "wins", "losses", "wins_as_black", "wins_as_white",
                  "total_plies", "endings"}
        if (not isinstance(row, dict) or set(row) != fields
                or any(type(row[key]) is not int or row[key] < 0 for key in fields - {"endings"})
                or row["games"] != self.games or row["wins"] + row["losses"] != self.games
                or row["wins_as_black"] + row["wins_as_white"] != row["wins"]
                or max(row["wins_as_black"], row["wins_as_white"]) > self.games // 2
                or not isinstance(row["endings"], dict)
                or any(type(key) is not str or type(value) is not int or value < 0
                       for key, value in row["endings"].items())
                or sum(row["endings"].values()) != self.games):
            raise ValueError("Invalid promotion evaluation result")

    def skipped_report(self, gate_passed):
        return {"enabled": self.directory is not None, "gate_passed": gate_passed,
                "rule": "candidate_mean_gt_reference_mean", "evaluated": False,
                "games_per_opponent": self.games, "opponent_count": len(self.opponents),
                "actual_games_played": 0, "seconds": 0.0,
                "reason": "disabled" if self.directory is None else "head_to_head_rejected"}

    def compare(self, candidate, reference, *, evaluator, **options):
        if self.directory is None:
            raise ValueError("Promotion archive is disabled")
        started = perf_counter()
        self.check_unchanged()
        # Do not share the primary duel's mutable diagnostics with archive duels.
        options = {key: value for key, value in options.items() if key != "diagnostics"}
        reference_hash = _weights_digest(reference)
        signature_payload = {"reference": reference_hash, "opponents": self.opponents,
                             "games": self.games, "seed": self.seed, "options": options,
                             "device": str(next(reference.parameters()).device)}
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True,
            allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_hit = (self.reference_cache is not None
                     and self.reference_cache["signature"] == signature)
        device = next(candidate.parameters()).device

        def evaluate(actor):
            results = []
            for opponent, model in zip(self.opponents, self.models):
                result = evaluator(actor, model.to(device), games=self.games,
                                   seed=self._seed(opponent), **options)
                if not isinstance(result, EvaluationResult):
                    raise TypeError("Promotion evaluator must return EvaluationResult")
                row = asdict(result)
                self._validate_result(row)
                results.append({"version": opponent["version"], "seed": self._seed(opponent),
                                "evaluation": row})
            return results

        reference_results = (deepcopy(self.reference_cache["results"]) if cache_hit
                             else evaluate(reference))
        candidate_results = evaluate(candidate)
        self.check_unchanged()
        # Failed or interrupted comparisons must not commit a partial cache.
        self.reference_cache = {"signature": signature, "results": deepcopy(reference_results)}
        candidate_wins = sum(row["evaluation"]["wins"] for row in candidate_results)
        reference_wins = sum(row["evaluation"]["wins"] for row in reference_results)
        games = self.games * len(self.opponents)
        improved = candidate_wins > reference_wins
        results = []
        for opponent, candidate_row, reference_row in zip(
                self.opponents, candidate_results, reference_results):
            candidate_result, reference_result = candidate_row["evaluation"], reference_row["evaluation"]
            results.append({**opponent, "seed": candidate_row["seed"],
                "candidate": {**candidate_result, "win_rate": candidate_result["wins"] / self.games},
                "reference": {**reference_result, "win_rate": reference_result["wins"] / self.games}})
        return {"enabled": True, "gate_passed": True, "evaluated": True,
                "rule": "candidate_mean_gt_reference_mean", "games_per_opponent": self.games,
                "opponent_count": len(self.opponents), "candidate_games": games,
                "reference_games": games, "actual_games_played": games * (1 if cache_hit else 2),
                "candidate_wins": candidate_wins, "reference_wins": reference_wins,
                "candidate_mean_win_rate": candidate_wins / games,
                "reference_mean_win_rate": reference_wins / games,
                "mean_win_rate_delta": (candidate_wins - reference_wins) / games,
                "reference_cache_hit": cache_hit, "reference_cache_signature": signature,
                "candidate_weights_sha256": _weights_digest(candidate),
                "reference_weights_sha256": reference_hash, "results": results,
                "passed": improved, "seconds": perf_counter() - started,
                "reason": "archive_average_improved" if improved else "archive_average_not_improved"}
