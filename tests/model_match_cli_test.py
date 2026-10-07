"""Exercise the visible two-model match program through its real CLI."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine

import torch

from kingdom_ai.checkpoint import save_model
from kingdom_ai.model import PolicyValueNet


REPOSITORY = Path(__file__).resolve().parents[1]
PROGRAM = REPOSITORY / "examples" / "model_match.py"


class ModelMatchCLITest(unittest.TestCase):
    def setUp(self):
        # Keep subprocess fixtures inside the shared writable workspace. Some
        # Windows sandbox profiles allow creating an OS temp directory but
        # deny opening files there from child processes.
        fixture_root = REPOSITORY / "runs"
        fixture_root.mkdir(exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="kingdom_match_test_", dir=fixture_root)
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)

    def run_cli(self, *arguments):
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, str(PROGRAM), *map(str, arguments)],
            cwd=REPOSITORY, env=environment, capture_output=True,
            text=True, encoding="utf-8", timeout=60,
        )

    def read_events(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def save_checkpoint(self, path, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = PolicyValueNet(channels=4, residual_blocks=0)
            save_model(model, path)

    def test_demo_records_final_boards_summary_and_refuses_overwrite(self):
        output = self.directory / "demo.jsonl"
        arguments = ("--demo", "--games", 2, "--simulations", 2, "--seed", 71,
                     "--show-board", "--output", output)
        process = self.run_cli(*arguments)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertIn("미학습", process.stdout)
        self.assertIn("대결 완료", process.stdout)
        self.assertIn("승률", process.stdout)
        self.assertIn("내부 시리즈 레이팅", process.stdout)
        self.assertEqual(process.stdout.count("x=흑, o=백, #=중립 돌"), 2)
        events = self.read_events(output)
        self.assertEqual([event["type"] for event in events],
                         ["session", "game", "game", "summary"])
        session, first, second, summary = events
        self.assertEqual(session["schema_version"], 1)
        self.assertEqual([item["source"] for item in session["participants"]],
                         ["untrained_demo", "untrained_demo"])
        self.assertEqual(session["protocol"]["simulations"], 2)
        self.assertEqual(session["protocol"]["rules"]["neutral"], [4, 4])
        self.assertEqual((first["model_a_color"], second["model_a_color"]),
                         ("Black", "White"))
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["wins_a"] + summary["wins_b"], 2)
        self.assertEqual(summary["total_plies"], first["plies"] + second["plies"])
        self.assertEqual(sum(summary["endings"].values()), 2)
        self.assertGreater(summary["elapsed_seconds"], 0)
        for game in (first, second):
            self.assertEqual(len(game["board"]), 9)
            self.assertTrue(all(len(row) == 9 for row in game["board"]))
            self.assertEqual(game["board"][4][4], "#")
            self.assertEqual(len(game["actions"]), game["plies"])
            self.assertIn(game["reason"], ("Capture", "Suicide", "TwoPasses"))
        original = output.read_bytes()
        repeated = self.run_cli(*arguments)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("이미 있습니다", repeated.stderr)
        self.assertEqual(output.read_bytes(), original)

    def test_saved_models_load_with_exact_checkpoint_fingerprints(self):
        model_a = self.directory / "기준 모델.pt"
        model_b = self.directory / "후보 모델.pt"
        self.save_checkpoint(model_a, 11)
        self.save_checkpoint(model_b, 29)
        output = self.directory / "checkpoints.jsonl"
        process = self.run_cli(
            "--model-a", model_a, "--model-b", model_b,
            "--name-a", "기준 모델", "--name-b", "후보 모델",
            "--games", 2, "--simulations", 2, "--output", output,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("기준 모델 vs 후보 모델", process.stdout)
        events = self.read_events(output)
        self.assertEqual([event["type"] for event in events],
                         ["session", "game", "game", "summary"])
        participants = events[0]["participants"]
        for entry, key, name, checkpoint in zip(
                participants, ("a", "b"), ("기준 모델", "후보 모델"), (model_a, model_b)):
            self.assertEqual(entry["id"], key)
            self.assertEqual(entry["name"], name)
            self.assertEqual(entry["path"], str(checkpoint.resolve()))
            self.assertEqual(entry["checkpoint_sha256"],
                             hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertRegex(entry["weights_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(entry["model_config"], {"channels": 4, "residual_blocks": 0})
        self.assertNotEqual(participants[0]["weights_sha256"], participants[1]["weights_sha256"])
        self.assertEqual(events[-1]["status"], "complete")

    def test_invalid_game_count_and_demo_model_combination_create_no_output(self):
        for index, arguments in enumerate((
                ("--demo", "--games", 3),
                ("--demo", "--model-a", self.directory / "unused.pt"),
        )):
            with self.subTest(arguments=arguments):
                output = self.directory / f"invalid_{index}.jsonl"
                process = self.run_cli(*arguments, "--output", output)
                self.assertEqual(process.returncode, 2)
                self.assertIn("error:", process.stderr)
                self.assertFalse(output.exists())

    def test_full_training_checkpoint_error_points_to_inference_model(self):
        checkpoint = self.directory / "latest.pt"
        torch.save({"checkpoint_version": 5, "learner": {}, "optimizer": {}}, checkpoint)
        output = self.directory / "unsupported.jsonl"
        process = self.run_cli(
            "--model-a", checkpoint, "--model-b", checkpoint,
            "--games", 2, "--simulations", 2, "--output", output,
        )
        self.assertEqual(process.returncode, 1)
        self.assertIn("best.pt", process.stderr)
        self.assertIn("latest.pt", process.stderr)
        self.assertIn("추론용", process.stderr)
        self.assertIn("완료 0판", process.stderr)
        self.assertFalse(output.exists())

    def test_watch_prints_one_based_moves_and_live_boards(self):
        output = self.directory / "watched.jsonl"
        process = self.run_cli(
            "--demo", "--games", 2, "--simulations", 1,
            "--opening-moves", 0, "--watch", "--output", output,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        games = [event for event in self.read_events(output) if event["type"] == "game"]
        total_plies = sum(game["plies"] for game in games)
        self.assertEqual(process.stdout.count("x=흑, o=백, #=중립 돌"), total_plies)
        coordinates = re.findall(r"(?:흑/선공|백/후공): (\d+)행 (\d+)열", process.stdout)
        self.assertTrue(coordinates)
        self.assertTrue(all(1 <= int(row) <= 9 and 1 <= int(col) <= 9
                            for row, col in coordinates))
        self.assertIn("[1/2] 대국 시작", process.stdout)
        self.assertIn("[2/2] 대국 시작", process.stdout)
        self.assertEqual(len(re.findall(r"^\s*5 \| .*#", process.stdout, re.MULTILINE)), total_plies)

    def test_output_interruption_after_summary_keeps_completed_result(self):
        specification = importlib.util.spec_from_file_location("kingdom_model_match_cli", PROGRAM)
        program = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(program)
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        output = self.directory / "completed_before_pipe_closed.jsonl"
        stdout, stderr = io.StringIO(), io.StringIO()

        class PassSearcher:
            def __init__(searcher, model, options):
                searcher.options = options

            def search(searcher, state):
                move = engine.Move.pass_turn()
                return SimpleNamespace(
                    best_move=move, simulations=searcher.options.simulations,
                    moves=[SimpleNamespace(move=move, visits=searcher.options.simulations)],
                )

        def interrupted_print(*args, **kwargs):
            if args and args[0] == "\n대결 완료":
                raise BrokenPipeError("result pipe closed")
            return builtins.print(*args, **kwargs)

        with redirect_stdout(stdout), redirect_stderr(stderr), \
                patch.object(program, "print", side_effect=interrupted_print, create=True), \
                patch("kingdom_ai.evaluation.PUCT", PassSearcher):
            status = program.main(["--demo", "--games", "2", "--simulations", "1",
                                   "--output", str(output)])
        self.assertEqual(status, 1)
        events = self.read_events(output)
        self.assertEqual([event["type"] for event in events],
                         ["session", "game", "game", "summary"])
        self.assertEqual(events[-1]["status"], "complete")
        self.assertIn("완료됐지만", stderr.getvalue())
        self.assertIn("레이팅은 유효", stderr.getvalue())
        self.assertNotIn("레이팅은 적용하지", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
