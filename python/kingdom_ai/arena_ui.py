"""Durable match dashboard over the existing JSONL match protocol.

The CLI runs in a separate process. The server tails complete JSONL lines,
keeps polls small, and never changes or resumes an imported match.
"""

from copy import deepcopy
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from time import monotonic
from uuid import uuid4

import my_board_engine as engine

from .match import MatchOptions, board_rows, series_ratings


REASONS = {"Capture": "포획", "Suicide": "자충수", "TwoPasses": "연속 패스"}


def run_id(path, root):
    return hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:20]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class MatchLog:
    def __init__(self, path, root, request=None):
        self.path, self.id = path, run_id(path, root)
        self.request = request or {}
        self.offset = 0
        self.session = None
        self.records = []
        self.by_index = {}
        self.terminal = None
        self.error = self.request.get("error")
        self.wins = {"a": 0, "b": 0}
        self.colors = {key: {color: {"games": 0, "wins": 0} for color in ("Black", "White")}
                       for key in ("a", "b")}
        self.endings = {key: 0 for key in REASONS}
        self.total_plies = 0
        self.curve = []

    def refresh(self):
        if not self.path.exists() or self.error:
            return
        if self.path.stat().st_size < self.offset:
            self.error = "결과 파일이 실행 중 변경되었습니다. 원본 기록을 확인해 주세요."
            return
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for line in handle:
                if not line.endswith(b"\n"):
                    break  # A writer may still be flushing the last line.
                try:
                    row = json.loads(line)
                    self._accept(row)
                except (ValueError, TypeError, KeyError) as error:
                    self.error = f"결과 기록을 읽지 못했습니다: {error}"
                    break
                self.offset = handle.tell()

    def _accept(self, row):
        if not isinstance(row, dict) or self.terminal is not None:
            raise ValueError("올바른 대결 기록이 아닙니다.")
        kind = row.get("type")
        if self.session is None:
            if kind != "session" or row.get("schema_version") != 1:
                raise ValueError("지원하지 않는 결과 형식입니다.")
            if {p["id"] for p in row["participants"]} != {"a", "b"}:
                raise ValueError("두 참가자 정보가 필요합니다.")
            options = MatchOptions(**{key: value for key, value in row["protocol"].items()
                                      if key in MatchOptions.__dataclass_fields__})
            row["protocol"] = {**asdict(options), **row["protocol"]}
            self.session = row
            return
        if kind == "game":
            protocol = self.session["protocol"]
            index = row["index"]
            if (type(index) is not int or not 1 <= index <= protocol["games"]
                    or index in self.by_index or row["winner"] not in self.wins
                    or row["model_a_color"] not in ("Black", "White")
                    or row["reason"] not in REASONS or type(row["plies"]) is not int
                    or not 1 <= row["plies"] <= 164
                    or len(row["actions"]) != row["plies"]
                    or any(type(action) is not int or not 0 <= action <= 81 for action in row["actions"])):
                raise ValueError("판별 승패 또는 수순이 올바르지 않습니다.")
            color_a = row["model_a_color"]
            color_b = "White" if color_a == "Black" else "Black"
            if (color_a != ("Black" if index % 2 else "White")
                    or row["pair_index"] != (index + 1) // 2
                    or row["seed"] != (protocol["seed"] + (index - 1) // 2) % (2 ** 64)):
                raise ValueError("판 번호와 흑백·시드 배정이 일치하지 않습니다.")
            if row["winner_color"] != (color_a if row["winner"] == "a" else color_b):
                raise ValueError("승자와 흑백 배정이 일치하지 않습니다.")
            self.records.append(row)
            self.by_index[index] = row
            self.wins[row["winner"]] += 1
            for key, color in (("a", color_a), ("b", color_b)):
                self.colors[key][color]["games"] += 1
                self.colors[key][color]["wins"] += int(row["winner"] == key)
            self.endings[row["reason"]] += 1
            self.total_plies += row["plies"]
            self.curve.append({"game": len(self.records), "rate": self.wins["a"] / len(self.records)})
        elif kind == "summary":
            protocol = self.session["protocol"]
            if (row.get("status") != "complete" or row["games"] != len(self.records)
                    or row["games"] != protocol["games"]
                    or row["wins_a"] != self.wins["a"] or row["wins_b"] != self.wins["b"]):
                raise ValueError("완료 통계와 저장된 판별 결과가 일치하지 않습니다.")
            ratings = protocol["initial_ratings"]
            expected = series_ratings(ratings["a"], ratings["b"], self.wins["a"], row["games"],
                                      protocol["k_per_series"])
            if row["ratings"] != expected:
                raise ValueError("완료 레이팅이 대결 설정과 일치하지 않습니다.")
            self.terminal = row
        elif kind == "aborted":
            if row["status"] not in ("cancelled", "failed") or row["completed_games"] != len(self.records):
                raise ValueError("중단 기록의 완료 판수가 일치하지 않습니다.")
            self.terminal = row
        else:
            raise ValueError("알 수 없는 결과 항목입니다.")

    def view(self, live=False, stopping=False, process_error=None):
        protocol = self.session["protocol"] if self.session else self.request.get("protocol", {})
        participants = self.session["participants"] if self.session else self.request.get("participants", [])
        status = ("failed" if self.error else self.terminal["status"] if self.terminal else
                  "stopping" if stopping and live else "running" if live and self.session else
                  "loading" if live else "failed" if process_error else "incomplete")
        completed = len(self.records)
        elapsed = self.terminal.get("elapsed_seconds") if self.terminal else None
        progress = None
        progress_path = self.path.parent / "progress.json"
        if live and not self.terminal and progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                elapsed = progress["elapsed_seconds"]
            except (OSError, ValueError, KeyError):
                pass
        target = protocol.get("games", completed)
        stride = max(1, math.ceil(completed / 200))
        curve = self.curve[::stride]
        if self.curve and (not curve or curve[-1] != self.curve[-1]):
            curve = curve + [self.curve[-1]]
        display_protocol = deepcopy(protocol)
        if "seed" in display_protocol:
            display_protocol["seed"] = str(display_protocol["seed"])
        return {"id": self.id, "status": status, "participants": deepcopy(participants),
                "protocol": display_protocol, "started_at_utc":
                self.session["started_at_utc"] if self.session else self.request.get("started_at_utc"),
                "completed": completed, "target": target, "wins": dict(self.wins),
                "win_rate_a": self.wins["a"] / completed if completed else None,
                "colors": deepcopy(self.colors), "endings": dict(self.endings),
                "mean_plies": self.total_plies / completed if completed else None,
                "elapsed_seconds": elapsed, "eta_seconds":
                elapsed / completed * (target - completed) if live and not self.terminal and elapsed and completed >= 2 else None,
                "games_per_minute": completed / elapsed * 60 if elapsed and completed else None,
                "progress": progress, "curve": deepcopy(curve),
                "ratings": deepcopy(self.terminal.get("ratings")) if self.terminal and not self.error else None,
                "error": self.error or (self.terminal.get("error") if self.terminal else process_error),
                "file": str(self.path), "can_stop": live and not stopping and not self.terminal,
                "can_download": self.path.exists()}


class MatchManager:
    def __init__(self, root, models, popen=subprocess.Popen, program=None):
        self.root = Path(root).resolve()
        self.runs_root = self.root / "runs"
        self.models = dict(models)
        self.popen = popen
        self.program = Path(program) if program else self.root / "examples/model_match.py"
        self.lock = RLock()
        self.logs = {}
        self.processes = {}
        self.last_scan = -math.inf
        self.closed = False

    def discover(self, force=False):
        if not force and monotonic() - self.last_scan < 5:
            return
        self.last_scan = monotonic()
        for identifier, log in list(self.logs.items()):
            process = self.processes.get(identifier)
            if (not log.path.exists() and not (log.path.parent / "request.json").exists()
                    and (process is None or process.poll() is not None)):
                del self.logs[identifier]
        if not self.runs_root.exists():
            return
        for path in self.runs_root.rglob("*.jsonl"):
            if not path.resolve().is_relative_to(self.runs_root.resolve()):
                continue
            identifier = run_id(path, self.root)
            if identifier in self.logs:
                continue
            try:
                with path.open(encoding="utf-8") as handle:
                    first = json.loads(handle.readline(262144))
                if first.get("type") == "session" and first.get("schema_version") == 1:
                    self.logs[identifier] = MatchLog(path, self.root)
            except (OSError, ValueError, AttributeError):
                continue
        # A model load can fail before the CLI creates its JSONL.
        for path in (self.runs_root / "matches").glob("*/request.json"):
            output = path.parent / "results.jsonl"
            identifier = run_id(output, self.root)
            if identifier not in self.logs:
                try:
                    self.logs[identifier] = MatchLog(output, self.root, json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue

    def _get(self, identifier):
        self.discover()
        if identifier not in self.logs:
            raise ValueError("대결 기록을 찾을 수 없습니다.")
        log = self.logs[identifier]
        log.refresh()
        return log

    def _view(self, log):
        process = self.processes.get(log.id)
        live = process is not None and process.poll() is None
        error = None
        if process is not None and not live and not log.terminal:
            error = self._log_error(log.path.parent)
            self._persist_error(log, error)
        return log.view(live, (log.path.parent / "stop.request").exists(), error)

    @staticmethod
    def _persist_error(log, message):
        log.error = message
        log.request["error"] = message
        request_path = log.path.parent / "request.json"
        if request_path.exists():
            request_path.write_text(json.dumps(log.request, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _log_error(directory):
        try:
            with (directory / "worker.log").open("rb") as handle:
                handle.seek(max(0, handle.seek(0, 2) - 3000))
                text = handle.read().decode("utf-8", errors="replace").strip()
            return text.splitlines()[-1] if text else "대결 작업이 완료 결과 없이 종료되었습니다."
        except OSError:
            return "대결 작업이 완료 결과 없이 종료되었습니다."

    def list_runs(self):
        with self.lock:
            self.discover()
            results = []
            for log in self.logs.values():
                log.refresh()
                view = self._view(log)
                results.append({key: view[key] for key in ("id", "status", "participants", "started_at_utc",
                                                          "completed", "target", "wins", "win_rate_a")})
            return sorted(results, key=lambda row: row["started_at_utc"] or "", reverse=True)

    def snapshot(self, identifier):
        with self.lock:
            return self._view(self._get(identifier))

    def start(self, values):
        if not isinstance(values, dict):
            raise ValueError("대결 설정을 확인해 주세요.")
        allowed = {"model_a", "model_b", "games", "simulations", "device", "seed", "tactical_checks",
                   "rating_a", "rating_b", "k"}
        execution = {"workers", "backend", "leaf_batch_size", "reuse_tree", "inference_wait_ms"}
        if not allowed.issubset(values) or set(values) - allowed - execution:
            raise ValueError("대결 설정 항목을 확인해 주세요.")
        if any(values[key] not in self.models for key in ("model_a", "model_b")):
            raise ValueError("목록에서 두 모델을 선택해 주세요.")
        if values["model_a"] == values["model_b"]:
            raise ValueError("서로 다른 모델 두 개를 선택해 주세요.")
        options = MatchOptions(games=values["games"], simulations=values["simulations"], seed=values["seed"],
                               tactical_checks=values["tactical_checks"],
                               **{key: values[key] for key in execution if key in values})
        if options.games > 100000 or options.simulations > 100000:
            raise ValueError("대국 수와 수당 탐색은 각각 100,000 이하로 설정해 주세요.")
        if options.workers > 64 or options.leaf_batch_size > 64 or options.inference_wait_ms > 10:
            raise ValueError("동시 대국·추론 배치는 각각 64 이하, 추론 대기는 10ms 이하로 설정해 주세요.")
        if values["device"] not in ("cpu", "cuda"):
            raise ValueError("CPU 또는 GPU를 선택해 주세요.")
        series_ratings(values["rating_a"], values["rating_b"], 0, options.games, values["k"])
        with self.lock:
            if self.closed:
                raise ValueError("서버가 종료 중입니다.")
            if any(process.poll() is None for process in self.processes.values()):
                raise ValueError("진행 중인 대결을 완료하거나 중단한 뒤 새 대결을 시작해 주세요.")
            folder = self.runs_root / "matches" / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
            folder.mkdir(parents=True)
            output = folder / "results.jsonl"
            names = {key: Path(values["model_" + key]).parent.as_posix().removeprefix("runs/") for key in ("a", "b")}
            request = {"started_at_utc": timestamp(), "participants": [{"id": key, "name": names[key]} for key in names],
                       "protocol": {**options.__dict__, "device": values["device"], "initial_ratings":
                                    {"a": values["rating_a"], "b": values["rating_b"]}, "k_per_series": values["k"]}}
            (folder / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            log = MatchLog(output, self.root, request)
            self.logs[log.id] = log
            arguments = [sys.executable, "-u", str(self.program)]
            for key in ("a", "b"):
                arguments += ["--model-" + key, str(Path(self.models[values["model_" + key]]).resolve()),
                              "--name-" + key, names[key]]
            for name in ("games", "simulations", "seed", "device", "rating_a", "rating_b", "k"):
                arguments += ["--" + name.replace("_", "-"), str(values[name])]
            for name in ("workers", "backend", "leaf_batch_size", "inference_wait_ms"):
                arguments += ["--" + name.replace("_", "-"), str(getattr(options, name))]
            arguments += ["--reuse-tree" if options.reuse_tree else "--no-reuse-tree"]
            arguments += ["--tactical-checks" if options.tactical_checks else "--no-tactical-checks",
                          "--output", str(output), "--progress-file", str(folder / "progress.json"),
                          "--stop-file", str(folder / "stop.request")]
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            try:
                with (folder / "worker.log").open("wb") as handle:
                    self.processes[log.id] = self.popen(arguments, cwd=self.root, env=environment,
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as error:
                self._persist_error(log, f"대결 작업을 시작하지 못했습니다: {error}")
                raise ValueError(log.error) from error
            return self._view(log)

    def stop(self, identifier):
        with self.lock:
            log = self._get(identifier)
            process = self.processes.get(identifier)
            if process is None or process.poll() is not None or log.terminal:
                raise ValueError("현재 실행 중인 대결이 아닙니다.")
            (log.path.parent / "stop.request").touch(exist_ok=True)
            return self._view(log)

    def games(self, identifier, *, page=1, winner="all", reason="all", color="all"):
        if type(page) is not int or page < 1 or winner not in ("all", "a", "b") or reason not in ("all", *REASONS) or color not in ("all", "Black", "White"):
            raise ValueError("검색 조건을 확인해 주세요.")
        with self.lock:
            log = self._get(identifier)
            rows = [row for row in sorted(log.records, key=lambda row: row["index"], reverse=True)
                    if (winner == "all" or row["winner"] == winner)
                    and (reason == "all" or row["reason"] == reason)
                    and (color == "all" or row["model_a_color"] == color)]
            pages = max(1, math.ceil(len(rows) / 25))
            page = min(page, pages)
            return {"page": page, "pages": pages, "total": len(rows), "items":
                    deepcopy([{key: row[key] for key in ("index", "model_a_color", "winner", "winner_color", "reason", "plies", "territory")}
                     for row in rows[(page - 1) * 25:page * 25]])}

    def replay(self, identifier, index):
        with self.lock:
            log = self._get(identifier)
            if type(index) is not int or index not in log.by_index:
                raise ValueError("해당 판의 기보를 찾을 수 없습니다.")
            row = deepcopy(log.by_index[index])
        state = engine.State()
        frames = []
        for action in [None, *row["actions"]]:
            if action is not None:
                move = engine.Move.pass_turn() if action == 81 else engine.Move.place(action // 9, action % 9)
                if not state.play(move).accepted():
                    raise ValueError("저장된 기보에 거부된 착수가 있습니다.")
            score = state.score()
            frames.append({"cells": [cell.name for cell in state.board.cells],
                           "ownership": [cell.name for cell in state.ownership], "action": action,
                           "score": {"black": score.black, "white": score.white}})
        if (not state.result.finished() or state.result.winner.name != row["winner_color"]
                or state.result.reason.name != row["reason"] or board_rows(state) != row["board"]):
            raise ValueError("기보 재생 결과와 저장된 종료 결과가 일치하지 않습니다.")
        return {"record": row, "frames": frames}

    def download(self, identifier, format):
        with self.lock:
            log = self._get(identifier)
            if format == "jsonl":
                if not log.path.exists():
                    raise ValueError("모델 준비 후 기록이 생성됩니다.")
                # Export only complete, validated lines, including during a write.
                with log.path.open("rb") as handle:
                    return handle.read(log.offset), "application/x-ndjson; charset=utf-8"
            if format != "csv":
                raise ValueError("CSV 또는 JSONL을 선택해 주세요.")
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            participants = self.session_names(log)
            def csv_name(key):
                name = participants.get(key, key.upper())
                return "'" + name if name.startswith(("=", "+", "-", "@", "\t", "\r")) else name
            writer.writerow(["판", "모델 A", "모델 B", "매 수 탐색", "시드", "A의 돌", "승자", "승자 돌", "종료 사유", "수순 수", "흑 집", "백 집"])
            for row in sorted(log.records, key=lambda row: row["index"]):
                writer.writerow([row["index"], csv_name("a"), csv_name("b"), log.session["protocol"]["simulations"], row["seed"], row["model_a_color"], row["winner"].upper(), row["winner_color"],
                                 REASONS[row["reason"]], row["plies"], row["territory"]["black"], row["territory"]["white"]])
            return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8"

    @staticmethod
    def session_names(log):
        return {item["id"]: item["name"] for item in log.session["participants"]} if log.session else {}

    def close(self):
        with self.lock:
            self.closed = True
            for identifier, process in self.processes.items():
                if process.poll() is None:
                    (self.logs[identifier].path.parent / "stop.request").touch(exist_ok=True)
