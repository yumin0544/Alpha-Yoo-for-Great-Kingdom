"""Local browser play, with all real moves adjudicated by the existing engine."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

import my_board_engine as engine


ERRORS = {"Occupied": "이미 돌이 놓인 자리입니다.", "OwnTerritory": "완성된 자기 집에는 놓을 수 없습니다.",
          "OpponentTerritory": "완성된 상대 집에는 놓을 수 없습니다.",
          "NoStones": "남은 돌이 없습니다. 패스를 선택해 주세요.",
          "OutOfBounds": "보드 안의 교차점을 선택해 주세요.", "GameOver": "대국이 끝났습니다."}
REASONS = {"Capture": "상대 돌 포획", "Suicide": "자충수", "TwoPasses": "연속 패스"}


@dataclass(frozen=True)
class PlaySettings:
    mode: str = "human_ai"
    human: str = "Black"
    model: str = "mcts"
    simulations: int = 64
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self):
        if self.mode not in ("human_ai", "human_human") or self.human not in ("Black", "White"):
            raise ValueError("대국 방식과 내 돌을 다시 선택해 주세요.")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("상대 AI를 선택해 주세요.")
        if type(self.simulations) is not int or not 1 <= self.simulations <= 4096:
            raise ValueError("탐색 횟수는 1~4096 사이여야 합니다.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("AI 장치는 CPU 또는 GPU를 선택해 주세요.")
        if self.model == "mcts" and self.device != "cpu":
            raise ValueError("기본 AI는 CPU에서 실행합니다. GPU에는 저장된 모델을 선택해 주세요.")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 64:
            raise ValueError("시드 범위를 확인해 주세요.")


class NeuralPlayer:
    """Match the CPU console opponent's tactical move selection."""
    def __init__(self, checkpoint, settings):
        import torch
        from .checkpoint import load_model
        from .puct import PUCT, PUCTOptions
        from .tactics import analyze_tactics, select_tactical_move
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("GPU를 사용할 수 없습니다. CPU를 선택해 주세요.")
        model = load_model(checkpoint, device=settings.device)
        self.searcher = PUCT(model, PUCTOptions(simulations=settings.simulations,
                                              seed=settings.seed, dirichlet_epsilon=0.0))
        self.analyze = analyze_tactics
        self.select = select_tactical_move

    def choose(self, state):
        from .encoding import action_to_move
        choices = self.analyze(state)
        if choices.winning_actions:
            return action_to_move(min(choices.winning_actions))
        return self.select(self.searcher.search(state), choices)


class MCTSPlayer:
    def __init__(self, settings):
        self.searcher = engine.MCTS(engine.MCTSOptions(simulations=settings.simulations,
                                                     seed=settings.seed, time_limit_ms=0))

    def choose(self, state):
        return self.searcher.search(state).best_move


class PlaySession:
    """One local game; revision checks reject double clicks and stale tabs.

    Loading and search run on one worker. Restart invalidates earlier tasks;
    their results cannot alter the new board. Engine states are never shared
    with a worker: the worker searches a copy and the UI owns actual play().
    """
    def __init__(self, models=None, player_factory=None):
        self.session_id = uuid4().hex
        self.models = dict(models or {})
        self.player_factory = player_factory or self._make_player
        self.lock = RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kingdom-player")
        self.state = engine.State()
        self.settings = PlaySettings(mode="human_human")
        self.player = None
        self.history = []
        self.generation = 0
        self.revision = 0
        self.busy = False
        self.error = None
        self.closed = False
        self.pending = None

    def _make_player(self, settings):
        if settings.model == "mcts":
            return MCTSPlayer(settings)
        return NeuralPlayer(self.models[settings.model], settings)

    def _check_revision(self, revision):
        if type(revision) is not int or revision != self.revision:
            raise ValueError("화면이 갱신되었습니다. 현재 보드를 확인하고 다시 선택해 주세요.")

    def new_game(self, settings, revision):
        if not isinstance(settings, PlaySettings):
            raise TypeError("PlaySettings required")
        if settings.model != "mcts" and settings.model not in self.models:
            raise ValueError("모델 목록에서 상대를 선택해 주세요.")
        with self.lock:
            self._check_revision(revision)
            self.generation += 1
            if self.pending is not None:
                self.pending.cancel()
            self.state = engine.State()
            self.settings = settings
            self.player = None
            self.history = []
            self.error = None
            self.revision += 1
            self.busy = settings.mode == "human_ai"
            if self.busy:
                self.pending = self.pool.submit(self._prepare, self.generation, settings)
            return self._snapshot()

    def _prepare(self, generation, settings):
        try:
            player = self.player_factory(settings)
            with self.lock:
                if self.closed or generation != self.generation:
                    return
                self.player = player
                self.busy = False
                self.revision += 1
                self._schedule_ai()
        except Exception as error:
            self._fail(generation, error)

    def _ai_turn(self):
        return (self.settings.mode == "human_ai"
                and self.state.to_play.name != self.settings.human)

    def _schedule_ai(self):
        if self.state.result.finished() or not self._ai_turn():
            return
        self.busy = True
        self.revision += 1
        self.pending = self.pool.submit(self._think, self.generation,
                                        self.state.copy(), self.player)

    def _think(self, generation, state, player):
        try:
            move = player.choose(state)
            with self.lock:
                if self.closed or generation != self.generation:
                    return
                self._apply(move)
                self.busy = False
        except Exception as error:
            self._fail(generation, error)

    def _fail(self, generation, error):
        with self.lock:
            if not self.closed and generation == self.generation:
                self.busy = False
                self.error = f"AI를 실행하지 못했습니다: {error}. 새 게임에서 모델이나 장치를 바꿔 주세요."
                self.revision += 1

    def _apply(self, move):
        if not isinstance(move, engine.Move):
            raise RuntimeError("AI가 수를 반환하지 않았습니다.")
        actor = self.state.to_play.name
        outcome = self.state.play(move)
        if not outcome.accepted():
            raise ValueError(ERRORS.get(outcome.error.name, "이 자리에 놓을 수 없습니다."))
        self.history.append({"number": len(self.history) + 1, "player": actor,
                             "action": 81 if move.is_pass() else move.point.row * 9 + move.point.col,
                             "label": "패스" if move.is_pass() else f"{move.point.row + 1}행 {move.point.col + 1}열"})
        self.revision += 1

    def move(self, action, revision):
        if type(action) is not int or not 0 <= action <= 81:
            raise ValueError("올바른 착수 지점을 선택해 주세요.")
        with self.lock:
            self._check_revision(revision)
            if self.busy or self._ai_turn() or self.error:
                raise ValueError("지금은 내 차례가 아닙니다.")
            move = engine.Move.pass_turn() if action == 81 else engine.Move.place(action // 9, action % 9)
            self._apply(move)
            self._schedule_ai()
            return self._snapshot()

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        result = self.state.result
        can_play = not (self.busy or result.finished() or self._ai_turn() or self.error)
        score = self.state.score()
        legal = [81 if move.is_pass() else move.point.row * 9 + move.point.col
                 for move in self.state.legal_moves()] if can_play else []
        return {"session_id": self.session_id, "revision": self.revision, "generation": self.generation,
                "settings": asdict(self.settings), "busy": self.busy,
                "ready": self.player is not None or self.settings.mode == "human_human",
                "can_play": bool(can_play), "error": self.error,
                "cells": [cell.name for cell in self.state.board.cells],
                "ownership": [cell.name for cell in self.state.ownership],
                "to_play": self.state.to_play.name, "legal_actions": legal,
                "remaining": {"Black": self.state.remaining_stones(engine.Cell.Black),
                              "White": self.state.remaining_stones(engine.Cell.White)},
                "score": {"Black": score.black, "White": score.white},
                "consecutive_passes": self.state.consecutive_passes,
                "finished": result.finished(), "winner": result.winner.name,
                "reason": REASONS.get(result.reason.name, ""),
                "history": deepcopy(self.history)}

    def close(self):
        with self.lock:
            self.closed = True
            self.generation += 1
        self.pool.shutdown(wait=False, cancel_futures=True)


def make_server(session, assets, port=0, matches=None):
    """Serve only loopback with exact static paths and same-origin JSON writes."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit

    assets = Path(assets)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, data, status=200, content_type="application/json; charset=utf-8", filename=None):
            payload = (json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
                       if isinstance(data, dict) else data)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return self.headers.get("Host") == expected

        def do_GET(self):
            if not self.allowed():
                self.send_data({"error": "잘못된 주소입니다."}, 403)
                return
            route = urlsplit(self.path)
            parts = route.path.strip("/").split("/")
            if matches is not None and parts[:2] == ["api", "matches"]:
                try:
                    if len(parts) == 2:
                        self.send_data({"runs": matches.list_runs(), "session_id": session.session_id})
                    elif len(parts) == 3:
                        self.send_data(matches.snapshot(parts[2]))
                    elif len(parts) == 4 and parts[3] == "games":
                        query = parse_qs(route.query)
                        self.send_data(matches.games(parts[2], page=int(query.get("page", ["1"])[0]),
                            winner=query.get("winner", ["all"])[0], reason=query.get("reason", ["all"])[0],
                            color=query.get("color", ["all"])[0]))
                    elif len(parts) == 5 and parts[3] == "replay":
                        self.send_data(matches.replay(parts[2], int(parts[4])))
                    elif len(parts) == 4 and parts[3] in ("csv", "jsonl"):
                        data, mime = matches.download(parts[2], parts[3])
                        self.send_data(data, content_type=mime,
                                       filename=f"match-{parts[2]}.{parts[3]}")
                    else:
                        self.send_data({"error": "페이지를 찾을 수 없습니다."}, 404)
                except (ValueError, TypeError, KeyError, OSError) as error:
                    self.send_data({"error": str(error)}, 400)
                return
            if self.path == "/api/state":
                self.send_data(session.snapshot())
            elif self.path == "/api/models":
                self.send_data({"models": [{"id": key, "name": Path(key).parent.as_posix()}
                                          for key in sorted(session.models, reverse=True)]})
            elif self.path in ("/", "/style.css", "/app.js", "/matches", "/match.css", "/match.js"):
                file, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/style.css": ("style.css", "text/css; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                              "/matches": ("match.html", "text/html; charset=utf-8"),
                              "/match.css": ("match.css", "text/css; charset=utf-8"),
                              "/match.js": ("match.js", "text/javascript; charset=utf-8")}[self.path]
                self.send_data((assets / file).read_bytes(), content_type=mime)
            else:
                self.send_data({"error": "페이지를 찾을 수 없습니다."}, 404)

        def do_POST(self):
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if not self.allowed() or self.headers.get("Origin") != expected:
                self.send_data({"error": "이 게임 화면에서만 조작할 수 있습니다."}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("요청 크기를 확인해 주세요.")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON 요청이 필요합니다.")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("요청 형식을 확인해 주세요.")
                if body.get("session_id") != session.session_id:
                    raise ValueError("대국 연결이 바뀌었습니다. 새 보드를 확인해 주세요.")
                if self.path == "/api/new":
                    response = session.new_game(PlaySettings(**body["settings"]), body["revision"])
                elif self.path == "/api/move":
                    response = session.move(body["action"], body["revision"])
                elif matches is not None and self.path == "/api/matches/start":
                    response = matches.start(body["settings"])
                elif matches is not None and self.path.startswith("/api/matches/") and self.path.endswith("/stop"):
                    parts = self.path.strip("/").split("/")
                    if len(parts) != 4:
                        raise ValueError("대결 주소를 확인해 주세요.")
                    response = matches.stop(parts[2])
                else:
                    self.send_data({"error": "동작을 찾을 수 없습니다."}, 404)
                    return
                self.send_data(response)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send_data({"error": str(error), "state": session.snapshot()}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
