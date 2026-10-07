"""Large catalogs, persistent results, cooperative cancellation and real CLI runs."""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
import my_board_engine as engine

from kingdom_ai.arena_ui import MatchManager, run_id
from kingdom_ai.checkpoint import save_model
from kingdom_ai.match import MatchOptions, board_rows, series_ratings
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.play_ui import PlaySession, make_server


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "examples/model_match.py"


class Fixture(unittest.TestCase):
    def setUp(self):
        fixture_root = ROOT / "build-parallel/arena-ui-tests"
        fixture_root.mkdir(parents=True,exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="arena_ui_test_", dir=fixture_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.models = {f"runs/{key}/best.pt": self.root / f"runs/{key}/best.pt" for key in ("a", "b")}
        self.manager = MatchManager(self.root, self.models, program=PROGRAM)
        self.addCleanup(self.manager.close)

    def settings(self, games=2):
        return {"model_a":"runs/a/best.pt", "model_b":"runs/b/best.pt", "games":games,
                "simulations":1, "seed":42, "device":"cpu", "tactical_checks":False,
                "rating_a":1500.0, "rating_b":1500.0, "k":32.0}

    def write_series(self, games, *, complete=True):
        path = self.root / "runs/legacy-match.jsonl"
        path.parent.mkdir(exist_ok=True)
        session = {"type":"session", "schema_version":1, "started_at_utc":"2026-10-07T00:00:00Z",
                   "participants":[{"id":"a","name":"TEST A"},{"id":"b","name":"TEST B"}],
                   "protocol":{**asdict(MatchOptions(games=games)), "device":"cpu",
                               "initial_ratings":{"a":1500.0,"b":1500.0},"k_per_series":32.0}}
        state = engine.State()
        state.play(engine.Move.pass_turn());state.play(engine.Move.pass_turn())
        records = [{"type":"game", "index":index, "pair_index":(index+1)//2, "seed":42+(index-1)//2,
                    "model_a_color":"Black" if index%2 else "White", "winner":"b" if index%2 else "a",
                    "winner_color":"White", "reason":"TwoPasses", "plies":2, "actions":[81,81],
                    "territory":{"black":0,"white":0}, "board":board_rows(state)} for index in range(1,games+1)]
        summary = {"type":"summary", "status":"complete", "games":games,"wins_a":games//2,"wins_b":games//2,
                   "ratings":series_ratings(1500.0,1500.0,games//2,games),"elapsed_seconds":10.0}
        events = [session,*records,*([summary] if complete else [])]
        path.write_text("".join(json.dumps(row)+"\n" for row in events),encoding="utf-8")
        return path, events


class CatalogTest(Fixture):
    def test_1024_results_statistics_filters_pagination_replay_and_export(self):
        path,_ = self.write_series(1024)
        catalog=self.manager.list_runs()
        self.assertEqual(len(catalog),1)
        identifier=catalog[0]["id"]
        view=self.manager.snapshot(identifier)
        self.assertEqual((view["status"],view["completed"],view["target"]),("complete",1024,1024))
        self.assertEqual(view["wins"],{"a":512,"b":512})
        self.assertEqual(view["win_rate_a"],.5)
        self.assertEqual(view["colors"]["a"]["Black"],{"games":512,"wins":0})
        self.assertEqual(view["colors"]["b"]["Black"],{"games":512,"wins":0})
        self.assertEqual(view["colors"]["b"]["White"],{"games":512,"wins":512})
        self.assertLessEqual(len(view["curve"]),201)
        self.assertEqual(view["curve"][-1],{"game":1024,"rate":.5})
        page=self.manager.games(identifier)
        self.assertEqual((page["pages"],page["items"][0]["index"],len(page["items"])),(41,1024,25))
        filtered=self.manager.games(identifier,winner="a",reason="TwoPasses",color="White")
        self.assertEqual(filtered["total"],512)
        self.assertEqual(self.manager.games(identifier,winner="a",color="Black")["total"],0)
        last=self.manager.games(identifier,page=999)
        self.assertEqual((last["page"],last["items"][-1]["index"]),(41,1))
        replay=self.manager.replay(identifier,1024)
        self.assertEqual(len(replay["frames"]),3)
        self.assertEqual(replay["frames"][0]["cells"][40],"Neutral")
        data,mime=self.manager.download(identifier,"jsonl")
        self.assertEqual(data,path.read_bytes())
        self.assertIn("ndjson",mime)
        csv,_=self.manager.download(identifier,"csv")
        self.assertEqual(len(csv.decode("utf-8-sig").splitlines()),1025)
        # Opening a fresh server keeps the completed results and ratings.
        reopened=MatchManager(self.root,self.models)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.snapshot(identifier)["ratings"],view["ratings"])

    def test_partial_lines_do_not_invent_completed_games_or_ratings(self):
        path,events=self.write_series(4)
        path.write_text(json.dumps(events[0])+"\n"+json.dumps(events[1])+"\n"+json.dumps(events[2])[:18],encoding="utf-8")
        identifier=self.manager.list_runs()[0]["id"]
        view=self.manager.snapshot(identifier)
        self.assertEqual((view["completed"],view["status"]),(1,"incomplete"))
        self.assertIsNone(view["ratings"])
        self.assertEqual(view["colors"]["a"]["White"]["games"],0)
        exported,_=self.manager.download(identifier,"jsonl")
        self.assertEqual(len(exported.splitlines()),2)
        with path.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(events[2])[18:]+"\n")
        self.assertEqual(self.manager.snapshot(identifier)["completed"],2)

    def test_inconsistent_completion_is_rejected(self):
        path,events=self.write_series(2)
        events[-1]["wins_a"]=2
        path.write_text("".join(json.dumps(row)+"\n" for row in events),encoding="utf-8")
        identifier=self.manager.list_runs()[0]["id"]
        view=self.manager.snapshot(identifier)
        self.assertEqual(view["status"],"failed")
        self.assertIsNone(view["ratings"])
        self.assertIn("일치",view["error"])

    def test_out_of_order_parallel_results_use_stable_game_ids(self):
        path, events = self.write_series(1024)
        # #1024 can finish while #1 is still playing. Legacy session headers
        # omit the new parallel fields and must remain readable.
        for key in ("workers", "backend", "leaf_batch_size", "reuse_tree", "inference_wait_ms"):
            events[0]["protocol"].pop(key)
        reordered = [events[0], *reversed(events[1:-1]), events[-1]]
        path.write_text("".join(json.dumps(row)+"\n" for row in reordered), encoding="utf-8")
        identifier = self.manager.list_runs()[0]["id"]
        view = self.manager.snapshot(identifier)
        self.assertEqual((view["status"], view["completed"]), ("complete", 1024))
        self.assertEqual(view["protocol"]["workers"], 1)
        self.assertEqual(self.manager.replay(identifier, 1)["record"]["index"], 1)
        self.assertEqual(self.manager.replay(identifier, 1024)["record"]["index"], 1024)
        self.assertEqual(self.manager.games(identifier)["items"][0]["index"], 1024)

    def test_partial_parallel_results_reject_duplicate_and_out_of_range_ids(self):
        path, events = self.write_series(4)
        path.write_text("".join(json.dumps(row)+"\n" for row in (events[0], events[4])), encoding="utf-8")
        identifier = self.manager.list_runs()[0]["id"]
        self.assertEqual(self.manager.replay(identifier, 4)["record"]["index"], 4)
        with self.assertRaises(ValueError): self.manager.replay(identifier, 1)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(events[4])+"\n")
        self.assertEqual(self.manager.snapshot(identifier)["status"], "failed")
        invalid_path = self.root / "runs/out-of-range.jsonl"
        invalid = {**events[4], "index": 5}
        invalid_path.write_text(json.dumps(events[0])+"\n"+json.dumps(invalid)+"\n", encoding="utf-8")
        self.manager.discover(force=True)
        self.assertEqual(self.manager.snapshot(run_id(invalid_path, self.root))["status"], "failed")

    def test_public_views_are_independent_and_ids_cannot_access_other_files(self):
        _,_=self.write_series(2)
        identifier=self.manager.list_runs()[0]["id"]
        view=self.manager.snapshot(identifier);view["curve"][0]["rate"]=.91;view["colors"]["a"]["Black"]["wins"]=10
        page=self.manager.games(identifier);page["items"][0]["territory"]["black"]=100
        self.assertEqual(self.manager.snapshot(identifier)["curve"][0]["rate"],0)
        self.assertEqual(self.manager.games(identifier)["items"][0]["territory"]["black"],0)
        for invalid in ("../AGENTS.md","unknown"):
            with self.assertRaises(ValueError):self.manager.snapshot(invalid)
        with self.assertRaises(ValueError):self.manager.games(identifier,page=0)
        with self.assertRaises(ValueError):self.manager.replay(identifier,3)

    def test_settings_are_validated_before_job_or_files_are_created(self):
        for changes in ({"games":1001},{"games":100002},{"simulations":True},{"device":"mps"},
                        {"model_a":"../other.pt"},{"model_b":"runs/a/best.pt"},{"k":float("nan")},
                        {"workers":0},{"workers":65},{"workers":True},{"backend":"other"},
                        {"leaf_batch_size":65},{"reuse_tree":1},{"inference_wait_ms":11}):
            with self.subTest(changes=changes),self.assertRaises((ValueError,TypeError)):
                self.manager.start({**self.settings(),**changes})
        self.assertFalse((self.root/"runs/matches").exists())


class WorkerTest(Fixture):
    def save_models(self):
        for index,path in enumerate(self.models.values()):
            path.parent.mkdir(parents=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(index+1)
                save_model(PolicyValueNet(channels=4,residual_blocks=0),path)

    def wait_job(self, identifier):
        process=self.manager.processes[identifier]
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        process.wait(timeout=60)
        return self.manager.snapshot(identifier)

    def test_real_saved_models_run_to_completion_through_existing_cli(self):
        self.save_models()
        view=self.manager.start(self.settings())
        self.assertEqual(view["status"],"loading")
        done=self.wait_job(view["id"])
        self.assertEqual(done["status"],"complete",done["error"])
        self.assertEqual(done["completed"],2)
        self.assertEqual(sum(done["wins"].values()),2)
        self.assertIsNotNone(done["ratings"])
        self.assertTrue((Path(done["file"]).parent/"progress.json").exists())
        self.assertGreaterEqual(len(self.manager.replay(view["id"],1)["frames"]),3)

    def test_parallel_saved_models_record_execution_and_progress(self):
        self.save_models()
        view = self.manager.start({**self.settings(games=8), "workers":4, "backend":"batched_cpp",
                                   "leaf_batch_size":4, "reuse_tree":True})
        done = self.wait_job(view["id"])
        self.assertEqual((done["status"], done["completed"]), ("complete", 8), done["error"])
        self.assertEqual(done["protocol"]["workers"], 4)
        self.assertEqual(done["protocol"]["backend"], "batched_cpp")
        progress = json.loads((Path(done["file"]).parent/"progress.json").read_text())
        self.assertLessEqual(len(progress["active_games"]), 4)
        self.assertGreater(done["games_per_minute"], 0)
        for index in range(1, 9): self.manager.replay(view["id"], index)

    def test_duplicate_jobs_are_blocked_and_stop_is_cooperative(self):
        self.save_models()
        view=self.manager.start(self.settings(games=1000))
        with self.assertRaisesRegex(ValueError,"진행 중"):
            self.manager.start(self.settings())
        stopped=self.manager.stop(view["id"])
        self.assertEqual(stopped["status"],"stopping")
        done=self.wait_job(view["id"])
        self.assertEqual(done["status"],"cancelled",done["error"])
        self.assertLess(done["completed"],1000)
        self.assertIsNone(done["ratings"])
        self.assertIsNone(done["eta_seconds"])
        self.assertGreaterEqual(done["elapsed_seconds"],0)

    def test_model_load_failure_is_visible_and_survives_reopening(self):
        for path in self.models.values():path.parent.mkdir(parents=True)
        view=self.manager.start(self.settings())
        failed=self.wait_job(view["id"])
        self.assertEqual(failed["status"],"failed")
        self.assertIsNotNone(failed["error"])
        self.assertIsNone(failed["ratings"])
        reopened=MatchManager(self.root,self.models)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.snapshot(view["id"])["status"],"failed")

    def test_cli_stop_preserves_completed_games_and_marks_cancelled(self):
        specification=importlib.util.spec_from_file_location("arena_cli",PROGRAM)
        program=importlib.util.module_from_spec(specification);specification.loader.exec_module(program)
        original_threads=torch.get_num_threads();self.addCleanup(torch.set_num_threads,original_threads)
        output=self.root/"partial.jsonl";stop=self.root/"stop.request";progress=self.root/"progress.json"
        class PassSearch:
            def __init__(self,model,options):self.options=options
            def search(self,state):
                move=engine.Move.pass_turn()
                return SimpleNamespace(best_move=move,simulations=self.options.simulations,
                                       moves=[SimpleNamespace(move=move,visits=self.options.simulations)])
        def observe_print(*args,**kwargs):
            if args and str(args[0]).startswith("[1/4]") and "승리" in args[0]:stop.touch()
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),patch("kingdom_ai.evaluation.PUCT",PassSearch),patch.object(program,"print",side_effect=observe_print,create=True):
            status=program.main(["--demo","--games","4","--simulations","1","--output",str(output),
                                 "--stop-file",str(stop),"--progress-file",str(progress)])
        events=[json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(status,130)
        self.assertEqual([event["type"] for event in events],["session","game","aborted"])
        self.assertEqual(events[-1]["status"],"cancelled")
        self.assertEqual(events[-1]["completed_games"],1)
        self.assertTrue(progress.exists())


class HTTPTest(Fixture):
    def setUp(self):
        super().setUp()
        self.write_series(2)
        self.session=PlaySession(self.models);self.addCleanup(self.session.close)
        self.server=make_server(self.session,ROOT/"python/kingdom_ai/web",matches=self.manager)
        self.thread=Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close_server)
        self.origin=f"http://127.0.0.1:{self.server.server_port}"

    def close_server(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)

    def request(self,path,body=None,origin=None):
        headers={}
        if body is not None:
            headers={"Content-Type":"application/json","Origin":origin or self.origin}
            body={"session_id":self.session.session_id,**body}
        with urlopen(Request(self.origin+path,data=json.dumps(body).encode() if body is not None else None,headers=headers),timeout=5) as response:
            return response.read(),response.headers

    def test_history_pages_replay_exports_and_dashboard_assets(self):
        for path in ("/matches","/match.js","/match.css"):
            data,_=self.request(path);self.assertTrue(data)
        data,_=self.request("/api/matches");identifier=json.loads(data)["runs"][0]["id"]
        data,_=self.request(f"/api/matches/{identifier}/games?winner=a")
        self.assertEqual(json.loads(data)["total"],1)
        data,_=self.request(f"/api/matches/{identifier}/replay/2")
        self.assertEqual(len(json.loads(data)["frames"]),3)
        data,headers=self.request(f"/api/matches/{identifier}/csv")
        self.assertIn("attachment",headers["Content-Disposition"])
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))

    def test_start_rejects_foreign_origins_and_old_server_sessions(self):
        for kwargs in ({"origin":"https://other.example"},{}):
            body={"settings":self.settings()}
            if not kwargs:body["session_id"]="old-server"
            with self.subTest(kwargs=kwargs),self.assertRaises(HTTPError) as caught:
                self.request("/api/matches/start",body,**kwargs)
            self.assertIn(caught.exception.code,(400,403))
        self.assertEqual(self.manager.processes,{})


if __name__=="__main__":
    unittest.main(verbosity=2)
