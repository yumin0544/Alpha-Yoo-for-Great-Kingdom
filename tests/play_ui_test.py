"""Actual rules, asynchronous restarts and local browser transport."""

import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import my_board_engine as engine

from kingdom_ai.play_ui import PlaySession, PlaySettings, make_server


ROOT = Path(__file__).resolve().parents[1]


def wait_ready(session):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        state = session.snapshot()
        if not state["busy"]:
            return state
        sleep(.005)
    raise AssertionError("AI worker did not finish")


class PassPlayer:
    def choose(self, state):
        return engine.Move.pass_turn()


class PlaySessionTest(unittest.TestCase):
    def setUp(self):
        self.session = PlaySession(player_factory=lambda settings: PassPlayer())
        self.addCleanup(self.session.close)

    def move(self, action):
        return self.session.move(action, self.session.snapshot()["revision"])

    def test_human_game_moves_and_immutable_snapshots(self):
        initial = self.session.snapshot()
        self.assertEqual(initial["cells"][40], "Neutral")
        self.assertNotIn(40, initial["legal_actions"])
        moved = self.move(20)
        self.assertEqual(moved["cells"][20], "Black")
        self.assertEqual(moved["to_play"], "White")
        self.assertEqual(moved["remaining"], {"Black": 40, "White": 41})
        moved["cells"][40] = "Empty"
        moved["history"][0]["label"] = "changed"
        fresh = self.session.snapshot()
        self.assertEqual(fresh["cells"][40], "Neutral")
        self.assertEqual(fresh["history"][0]["label"], "3행 3열")

    def test_illegal_and_duplicate_clicks_do_not_change_game(self):
        initial = self.session.snapshot()
        with self.assertRaisesRegex(ValueError, "이미 돌"):
            self.move(40)
        self.assertEqual(initial, self.session.snapshot())
        self.move(0)
        after = self.session.snapshot()
        with self.assertRaisesRegex(ValueError, "갱신"):
            self.session.move(1, initial["revision"])
        self.assertEqual(after, self.session.snapshot())
        for action in (True, -1, 82, "1", None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.move(action)

    def test_two_passes_finish_with_real_engine_winner(self):
        self.move(81)
        result = self.move(81)
        self.assertTrue(result["finished"])
        self.assertEqual(result["winner"], "White")
        self.assertEqual(result["reason"], "연속 패스")
        self.assertFalse(result["can_play"])
        with self.assertRaisesRegex(ValueError, "끝났습니다"):
            self.move(0)

    def test_suicide_is_accepted_and_not_mistaken_for_illegal_move(self):
        board = engine.Board()
        for player, points in ((engine.Cell.Black, [(0,1)]),
                               (engine.Cell.White, [(0,0),(0,2),(1,0),(1,2),(2,1)])):
            for row,col in points:
                board.place(engine.Position(row,col), player)
        self.session.state = engine.State(board, engine.Cell.Black)
        before = self.session.snapshot()
        self.assertIn(10, before["legal_actions"])
        result = self.move(10)
        self.assertTrue(result["finished"])
        self.assertEqual(result["winner"], "White")
        self.assertEqual(result["reason"], "자충수")
        self.assertEqual(result["history"][-1]["action"], 10)

    def test_ai_reply_and_white_human_start(self):
        self.session.new_game(PlaySettings(), self.session.revision)
        wait_ready(self.session)
        self.move(20)
        result = wait_ready(self.session)
        self.assertEqual([move["action"] for move in result["history"]], [20,81])
        self.assertEqual(result["to_play"], "Black")
        self.assertTrue(result["can_play"])
        self.session.new_game(PlaySettings(human="White"), result["revision"])
        result = wait_ready(self.session)
        self.assertEqual(result["history"][0]["player"], "Black")
        self.assertEqual(result["to_play"], "White")
        self.assertEqual(result["remaining"], {"Black": 41, "White": 41})

    def test_restart_discards_inflight_ai_result(self):
        started, release, finished = Event(), Event(), Event()

        class SlowPlayer:
            def choose(self, state):
                started.set()
                release.wait(3)
                finished.set()
                return engine.Move.place(0,0)

        self.addCleanup(release.set)
        self.session.player_factory = lambda settings: SlowPlayer()
        self.session.new_game(PlaySettings(human="White"), self.session.revision)
        self.assertTrue(started.wait(3))
        old = self.session.snapshot()
        self.assertTrue(old["busy"])
        with self.assertRaisesRegex(ValueError, "차례"):
            self.session.move(1, old["revision"])
        restarted = self.session.new_game(PlaySettings(mode="human_human"), old["revision"])
        release.set()
        self.assertTrue(finished.wait(3))
        # Drain the single worker so the old result has attempted its commit.
        self.session.pool.submit(lambda: None).result(timeout=3)
        self.assertEqual(self.session.snapshot(), restarted)
        self.assertEqual(restarted["history"], [])
        self.assertEqual(restarted["cells"][0], "Empty")

    def test_old_model_load_failure_cannot_affect_restarted_game(self):
        started, release = Event(), Event()
        def fail(settings):
            started.set()
            release.wait(3)
            raise RuntimeError("model missing")
        self.addCleanup(release.set)
        self.session.player_factory = fail
        self.session.new_game(PlaySettings(), self.session.revision)
        self.assertTrue(started.wait(3))
        restarted = self.session.new_game(PlaySettings(mode="human_human"), self.session.revision)
        release.set()
        self.session.pool.submit(lambda: None).result(timeout=3)
        self.assertEqual(restarted, self.session.snapshot())

    def test_load_failure_is_visible_and_new_game_recovers(self):
        def fail(settings):
            raise ValueError("bad checkpoint")
        self.session.player_factory = fail
        self.session.new_game(PlaySettings(), self.session.revision)
        failed = wait_ready(self.session)
        self.assertIn("bad checkpoint", failed["error"])
        self.assertFalse(failed["can_play"])
        recovered = self.session.new_game(PlaySettings(mode="human_human"), failed["revision"])
        self.assertIsNone(recovered["error"])
        self.assertTrue(recovered["can_play"])

    def test_settings_and_unregistered_models_rejected_before_reset(self):
        for values in ({"mode":"unknown"},{"human":"Neutral"},{"simulations":True},
                       {"simulations":0},{"simulations":4097},{"seed":-1},{"device":"mps"},
                       {"device":"cuda","model":"mcts"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PlaySettings(**values)
        initial = self.session.snapshot()
        with self.assertRaisesRegex(ValueError, "모델 목록"):
            self.session.new_game(PlaySettings(model="../secret.pt"), initial["revision"])
        self.assertEqual(initial, self.session.snapshot())

    def test_real_mcts_moves_on_actual_board(self):
        session = PlaySession()
        self.addCleanup(session.close)
        session.new_game(PlaySettings(human="White",simulations=1), session.revision)
        result = wait_ready(session)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["cells"][40], "Neutral")
        self.assertEqual(result["to_play"], "White")


class PlayHTTPTest(unittest.TestCase):
    def setUp(self):
        self.session = PlaySession()
        self.server = make_server(self.session, ROOT / "python/kingdom_ai/web")
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.session.close()

    def request(self, path, body=None, origin=None, host=None):
        headers = {}
        if body is not None:
            headers = {"Content-Type":"application/json", "Origin":origin or self.origin}
            body = {"session_id":self.session.session_id, **body}
        if host:
            headers["Host"] = host
        request = Request(self.origin + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers=headers)
        with urlopen(request, timeout=3) as response:
            return response.status, response.read()

    def test_assets_and_actual_move_roundtrip(self):
        for asset in ("/","/app.js","/style.css"):
            status, content = self.request(asset)
            self.assertEqual(status,200)
            self.assertTrue(content)
        _, payload = self.request("/api/state")
        state = json.loads(payload)
        _, payload = self.request("/api/move", {"action":20,"revision":state["revision"]})
        moved = json.loads(payload)
        self.assertEqual(moved["cells"][20], "Black")
        self.assertEqual(moved["history"][0]["label"], "3행 3열")

    def test_external_origin_host_and_unknown_paths_are_rejected(self):
        for kwargs in ({"origin":"https://unrelated.example"}, {"host":"unrelated.example"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(HTTPError) as caught:
                self.request("/api/move", {"action":0,"revision":0}, **kwargs)
            self.assertEqual(caught.exception.code,403)
        with self.assertRaises(HTTPError) as caught:
            self.request("/../AGENTS.md")
        self.assertEqual(caught.exception.code,404)
        self.assertEqual(self.session.snapshot()["history"], [])

    def test_bad_json_operation_returns_current_state(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/api/move", {"action":40,"revision":0})
        self.assertEqual(caught.exception.code,400)
        error = json.loads(caught.exception.read())
        self.assertIn("이미 돌", error["error"])
        self.assertEqual(error["state"]["cells"][40], "Neutral")

    def test_previous_server_session_cannot_apply_moves(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/api/move", {"action":0,"revision":0,"session_id":"previous-server"})
        self.assertEqual(caught.exception.code,400)
        self.assertEqual(self.session.snapshot()["history"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
