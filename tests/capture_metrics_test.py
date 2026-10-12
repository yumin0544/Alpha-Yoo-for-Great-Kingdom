"""Complete match accounting never confuses endings with a model's losses."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kingdom_ai.capture_metrics import analyze_capture_losses, load_match_records


REPOSITORY = Path(__file__).resolve().parents[1]
PROGRAM = REPOSITORY / "examples" / "analyze_capture_losses.py"


def match_events(outcomes):
    session = {"type": "session", "schema_version": 1,
               "participants": [{"id": "a", "name": "후보"}, {"id": "b", "name": "기준"}],
               "protocol": {"games": len(outcomes), "seed": 42}}
    games = []
    for index, (winner, reason) in enumerate(outcomes, 1):
        color = "Black" if index % 2 else "White"
        games.append({"type": "game", "index": index, "pair_index": (index + 1) // 2,
                      "seed": 42 + (index - 1) // 2, "model_a_color": color,
                      "winner": winner,
                      "winner_color": color if winner == "a" else
                      "White" if color == "Black" else "Black",
                      "reason": reason, "plies": 2, "actions": [81, 81],
                      "territory": {"black": 0, "white": 0}, "board": ["........."] * 9})
    wins_a = sum(game["winner"] == "a" for game in games)
    summary = {"type": "summary", "status": "complete", "games": len(games),
               "wins_a": wins_a, "wins_b": len(games) - wins_a,
               "win_rate_a": wins_a / len(games),
               "wins_a_as_black": sum(g["winner"] == "a" and g["model_a_color"] == "Black"
                                      for g in games),
               "wins_a_as_white": sum(g["winner"] == "a" and g["model_a_color"] == "White"
                                      for g in games),
               "total_plies": len(games) * 2,
               "endings": {reason: sum(game["reason"] == reason for game in games)
                           for reason in ("Capture", "Suicide", "TwoPasses")}}
    return [session, *games, summary]


class CaptureMetricsTest(unittest.TestCase):
    def setUp(self):
        fixture_root = REPOSITORY / "runs"
        fixture_root.mkdir(exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="kingdom_capture_metrics_test_", dir=fixture_root)
        self.addCleanup(workspace.cleanup)
        self.path = Path(workspace.name) / "results.jsonl"
        self.events = match_events([("a", "Capture"), ("b", "Capture"),
                                   ("b", "Suicide"), ("b", "TwoPasses"),
                                   ("a", "TwoPasses"), ("a", "Suicide")])

    def write_events(self, events=None):
        self.path.write_text("\n".join(json.dumps(event, ensure_ascii=False)
                                      for event in (self.events if events is None else events)) + "\n",
                             encoding="utf-8")

    def test_both_participants_have_separate_losses_and_explicit_denominators(self):
        self.write_events()
        report = analyze_capture_losses(self.path)
        for model in ("a", "b"):
            metrics = report["models"][model]
            self.assertEqual((metrics["games"], metrics["wins"], metrics["losses"]), (6, 3, 3))
            self.assertEqual(metrics["win_rate"], 0.5)
            for field in ("capture_losses", "suicide_losses", "pass_losses"):
                self.assertEqual(metrics[field], 1)
                self.assertEqual(metrics["loss_rates_all_games"][field], 1 / 6)
                self.assertEqual(metrics["loss_shares_all_losses"][field], 1 / 3)
            self.assertEqual(metrics["capture_or_suicide_losses"], 2)
        self.assertEqual(report["models"]["a"]["colors"]["Black"]["wins"], 2)
        self.assertEqual(report["models"]["a"]["colors"]["White"]["wins"], 1)
        self.assertEqual(report["models"]["b"]["colors"]["Black"]["wins"], 2)
        self.assertEqual(report["models"]["b"]["colors"]["White"]["wins"], 1)
        self.assertEqual(report["models"]["a"]["colors"]["Black"]["suicide_losses"], 1)
        self.assertEqual(report["models"]["a"]["colors"]["White"]["capture_losses"], 1)

    def test_zero_capture_can_still_mean_only_pass_losses(self):
        self.write_events(match_events([("b", "TwoPasses"), ("b", "TwoPasses")]))
        metrics = analyze_capture_losses(self.path)["models"]["a"]
        self.assertEqual(metrics["capture_or_suicide_losses"], 0)
        self.assertEqual(metrics["pass_losses"], 2)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["loss_rates_all_games"]["pass_losses"], 1.0)

    def test_zero_loss_denominator_is_null_not_zero(self):
        self.write_events(match_events([("a", "Capture"), ("a", "TwoPasses")]))
        metrics = analyze_capture_losses(self.path)["models"]["a"]
        self.assertEqual(metrics["win_rate"], 1.0)
        self.assertTrue(all(value is None for value in metrics["loss_shares_all_losses"].values()))
        self.assertTrue(all(value == 0 for value in metrics["loss_rates_all_games"].values()))

    def test_out_of_order_parallel_completion_is_accepted_and_inputs_unchanged(self):
        self.write_events([self.events[0], *reversed(self.events[1:-1]), self.events[-1]])
        before_bytes = self.path.read_bytes()
        before_modified = self.path.stat().st_mtime_ns
        session, games = load_match_records(self.path)
        self.assertEqual([game["index"] for game in games], list(range(1, 7)))
        self.assertEqual(session["protocol"]["games"], 6)
        analyze_capture_losses(self.path)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(self.path.stat().st_mtime_ns, before_modified)

    def test_invalid_winner_reason_and_colors_are_rejected(self):
        for field, value in (("winner", "draw"), ("winner", True), ("reason", "None"),
                             ("reason", None), ("model_a_color", "black"),
                             ("model_a_color", "White"), ("winner_color", "White")):
            with self.subTest(field=field, value=value):
                events = deepcopy(self.events)
                events[1][field] = value
                self.write_events(events)
                with self.assertRaisesRegex(ValueError, field):
                    analyze_capture_losses(self.path)

    def test_duplicate_missing_and_non_integer_indices_are_rejected(self):
        for mutation in ("duplicate", "missing", "bool", "out_of_range", "pair", "seed"):
            with self.subTest(mutation=mutation):
                events = deepcopy(self.events)
                if mutation == "duplicate":
                    events[2] = deepcopy(events[1])
                elif mutation == "missing":
                    events.pop(1)
                else:
                    field, value = {"bool": ("index", True), "out_of_range": ("index", 7),
                                    "pair": ("pair_index", 2), "seed": ("seed", 0)}[mutation]
                    events[1][field] = value
                self.write_events(events)
                with self.assertRaises(ValueError):
                    analyze_capture_losses(self.path)

    def test_schema_session_and_completion_requirements_are_strict(self):
        cases = [[], self.events[1:], self.events[:-1],
                 [self.events[0], self.events[0], *self.events[1:]],
                 [*self.events, self.events[-1]],
                 [*self.events[:-1], {"type": "aborted", "completed_games": 6}]]
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("participants", [{"id": "a", "name": "a"}]),
                             ("protocol", {"games": 3, "seed": 42})):
            events = deepcopy(self.events)
            events[0][field] = value
            cases.append(events)
        for events in cases:
            with self.subTest(events=events):
                self.write_events(events)
                with self.assertRaises(ValueError):
                    analyze_capture_losses(self.path)

    def test_summary_aggregate_cannot_contradict_individual_games(self):
        for field, value in (("games", 8), ("wins_a", 4), ("wins_b", True),
                             ("win_rate_a", 0.7), ("win_rate_a", float("nan")),
                             ("wins_a_as_black", 3), ("total_plies", 1),
                             ("endings", {"Capture": 6}), ("status", "cancelled")):
            with self.subTest(field=field, value=value):
                events = deepcopy(self.events)
                events[-1][field] = value
                self.write_events(events)
                with self.assertRaises(ValueError):
                    analyze_capture_losses(self.path)

    def test_malformed_json_unknown_record_and_duplicate_json_keys_are_rejected(self):
        for content in ('{bad json}\n', '[]\n', '{"type":"unknown"}\n',
                        '{"type":"session","type":"game"}\n', '\n'):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, ":1:"):
                    analyze_capture_losses(self.path)

    def test_game_payload_dimensions_and_action_types_are_validated(self):
        for field, value in (("plies", True), ("actions", [81]), ("actions", [True, 81]),
                             ("actions", [82, 81]), ("territory", {"black": -1, "white": 0}),
                             ("board", [".........."] * 9)):
            with self.subTest(field=field, value=value):
                events = deepcopy(self.events)
                events[1][field] = value
                self.write_events(events)
                with self.assertRaisesRegex(ValueError, field):
                    analyze_capture_losses(self.path)

    def test_cli_defaults_to_a_and_explicitly_warns_about_pass_loss_tradeoffs(self):
        self.write_events()
        specification = importlib.util.spec_from_file_location("capture_metrics_cli", PROGRAM)
        program = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(program)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(program.main([str(self.path)]), 0)
        text = stdout.getvalue()
        self.assertIn("모델 A (후보)", text)
        self.assertIn("포획+자충수 패배: 2판", text)
        self.assertIn("연속 패스/집 패배: 1판", text)
        self.assertIn("전체 경기 중 16.67% | 전체 패배 중 33.33%", text)
        self.assertIn("흑/선공", text)
        self.assertIn("백/후공", text)
        self.assertIn("미래의 포획 위험 0을 증명하지 않습니다", text)

    def test_cli_json_selection_b_and_real_subprocess_are_read_only(self):
        self.write_events()
        before = self.path.read_bytes()
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run([sys.executable, "-B", str(PROGRAM), str(self.path),
                                  "--model", "b", "--json"], cwd=REPOSITORY, env=environment,
                                 capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual(report["selected_model"], "b")
        self.assertEqual(report["models"]["b"]["name"], "기준")
        self.assertEqual(report["models"]["a"]["pass_losses"], 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_rejects_incomplete_log_with_no_report(self):
        self.write_events(self.events[:-1])
        specification = importlib.util.spec_from_file_location("capture_metrics_cli_error", PROGRAM)
        program = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(program)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            program.main([str(self.path), "--json"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("complete summary", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
