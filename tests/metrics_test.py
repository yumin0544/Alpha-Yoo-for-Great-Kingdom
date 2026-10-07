"""Training JSONL summary compatibility and projection tests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from kingdom_ai.metrics import load_metric_rows, summarize_metrics


def row(iteration, *, samples=200, games=10, self_play=2.0, elapsed=5.0):
    return {
        "iteration": iteration,
        "self_play_games": iteration * games,
        "training_steps": iteration * 2,
        "champion_version": iteration // 2,
        "generated_samples": samples,
        "replay_size": min(iteration * samples, 500),
        "mean_self_play_plies": samples / games,
        "self_play_endings": {"Capture": games},
        "loss": 1.0, "policy_loss": 0.5, "value_loss": 0.5,
        "evaluation": {"games": 2, "wins": 1, "losses": 1,
                       "wins_as_black": 1, "wins_as_white": 0,
                       "total_plies": 10, "endings": {"Capture": 2}, "win_rate": 0.5},
        "promoted": iteration % 2 == 0,
        "self_play_seconds": self_play,
        "training_seconds": 1.0,
        "evaluation_seconds": 2.0,
        "elapsed_seconds": elapsed,
        "self_play_backend": "cuda", "self_play_batch_size": games,
        "self_play_games_per_second": games / self_play,
        "iteration_games_per_second": games / elapsed,
    }


class MetricsTest(unittest.TestCase):
    def test_old_and_new_rows_are_summarized_and_duplicates_use_last(self):
        old = row(1)
        duplicate = row(1, samples=100, games=10, self_play=1.0)
        new = row(2, samples=300, games=10, self_play=3.0, elapsed=6.0)
        new.update({
            "games_per_iteration": 10, "replay_capacity": 500,
            "self_play_winners": {"Black": 4, "White": 6},
            "replay_turnover": 0.6,
            "training_draws_per_generated_sample": 0.5,
            "training_draws_per_replay_sample": 0.3,
            "self_play_compute_seconds": 2.25, "replay_store_seconds": 0.75,
            "evaluation_workers": 12,
            "checkpoint_written": True,
            "checkpoint_seconds": 1.0, "initial_checkpoint_seconds": 0.5,
            "checkpoint_bytes": 1234, "elapsed_with_checkpoint_seconds": 7.5,
        })
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("\n".join(json.dumps(value) for value in (
                {"kind": "environment"}, old, duplicate, new)), encoding="utf-8")
            rows, duplicates = load_metric_rows(path)
        self.assertEqual([value["iteration"] for value in rows], [1, 2])
        self.assertEqual(duplicates, 1)
        summary = summarize_metrics(rows, recent=1, target_iterations=4,
                                    duplicate_rows=duplicates)
        self.assertEqual(summary["overall"]["generated_samples"], 400)
        self.assertEqual(summary["recent"]["generated_samples"], 300)
        self.assertEqual(summary["recent"]["positions_per_self_play_second"], 100.0)
        self.assertEqual(summary["evaluation"]["candidate_win_rate"], 0.5)
        self.assertEqual(summary["evaluation"]["black_win_rate"], 1.0)
        self.assertEqual(summary["evaluation"]["white_win_rate"], 0.0)
        self.assertEqual(summary["self_play_colors"]["covered_iterations"], 1)
        self.assertEqual(summary["self_play_detail"]["covered_iterations"], 1)
        self.assertEqual(summary["self_play_detail"]["replay_store_fraction"], 0.25)
        self.assertEqual(summary["evaluation"]["latest_workers"], 12)
        self.assertEqual(summary["checkpoint"]["rows_with_timing"], 1)
        self.assertEqual(summary["projection"]["remaining_iterations"], 2)
        self.assertEqual(summary["projection"]["estimated_seconds"], 15.0)

    def test_invalid_or_empty_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"kind":"environment"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metric_rows(path)
            path.write_text('{bad json}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metric_rows(path)
            malformed = row(1)
            malformed.pop("evaluation")
            path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation"):
                load_metric_rows(path)
        with self.assertRaises(ValueError):
            summarize_metrics([], recent=1)
        with self.assertRaises(ValueError):
            summarize_metrics([row(1)], recent=0)

    def test_zero_checkpoint_fields_do_not_claim_a_checkpoint_write(self):
        value = row(1)
        value.update({
            "checkpoint_written": False,
            "checkpoint_seconds": 0.0,
            "initial_checkpoint_seconds": 0.0,
            "checkpoint_bytes": 0,
            "elapsed_with_checkpoint_seconds": value["elapsed_seconds"],
        })
        summary = summarize_metrics([value], target_iterations=2)
        self.assertEqual(summary["checkpoint"]["rows_with_timing"], 0)
        self.assertFalse(summary["projection"]["includes_checkpoint_where_available"])

    def test_legacy_rows_have_no_online_tactics_cost_or_claimed_report(self):
        summary = summarize_metrics([row(1)])
        self.assertEqual(summary["phase_seconds"]["online_tactics"], 0.0)
        self.assertEqual(summary["phase_fractions"]["online_tactics"], 0.0)
        online = summary["online_tactics"]
        self.assertEqual(online["timing_covered_iterations"], 0)
        self.assertEqual(online["report_covered_iterations"], 0)
        self.assertEqual(online["added_samples"], 0)
        self.assertEqual(online["mixed_updates"], 0)
        self.assertIsNone(online["latest_replay_size"])

    def test_online_tactics_timing_is_separate_and_uses_full_elapsed_denominator(self):
        value = row(1, elapsed=10.0)
        value.update({
            "online_tactics_seconds": 4.0,
            "online_tactics_enabled": True,
            "online_tactics": {
                "enabled": True, "seconds": 4.0, "added_samples": 3,
                "excluded_shallow": 2, "replay_size": 5, "mixed_updates": 2,
                "tactical_rows_per_batch": 4, "replay_rows_per_batch": 12,
            },
        })
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            rows, _ = load_metric_rows(path)
        summary = summarize_metrics(rows, target_iterations=3)
        self.assertEqual(summary["phase_seconds"]["online_tactics"], 4.0)
        fractions = summary["phase_fractions"]
        self.assertEqual(fractions["online_tactics"], 0.4)
        self.assertEqual(fractions["self_play"], 0.2)
        self.assertEqual(fractions["training"], 0.1)
        self.assertEqual(fractions["evaluation"], 0.2)
        self.assertEqual(fractions["unattributed"], 0.1)
        self.assertAlmostEqual(sum(fractions.values()), 1.0)
        self.assertEqual(summary["projection"]["estimated_seconds"], 20.0)
        online = summary["online_tactics"]
        self.assertEqual(online["timing_covered_iterations"], 1)
        self.assertEqual(online["enabled_iterations"], 1)
        self.assertEqual(online["added_samples"], 3)
        self.assertEqual(online["excluded_shallow"], 2)
        self.assertEqual(online["mixed_updates"], 2)
        self.assertEqual(online["tactical_training_draws"], 8)
        self.assertEqual(online["latest_replay_size"], 5)
        self.assertEqual(online["latest_tactical_rows_per_batch"], 4)

    def test_zero_online_tactics_seconds_are_valid_and_coverage_is_explicit(self):
        value = row(1)
        value.update({"online_tactics_seconds": 0.0,
                      "online_tactics": {"enabled": False, "seconds": 0.0,
                                         "added_samples": 0, "replay_size": 0,
                                         "mixed_updates": 0}})
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            rows, _ = load_metric_rows(path)
        summary = summarize_metrics(rows)
        self.assertEqual(summary["phase_seconds"]["online_tactics"], 0.0)
        self.assertEqual(summary["online_tactics"]["timing_covered_iterations"], 1)
        self.assertEqual(summary["online_tactics"]["enabled_iterations"], 0)
        self.assertEqual(summary["online_tactics"]["latest_replay_size"], 0)

    def test_invalid_online_tactics_timing_and_counts_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            for bad in (-1.0, float("nan"), float("inf"), True):
                with self.subTest(seconds=bad):
                    value = row(1)
                    value["online_tactics_seconds"] = bad
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "online_tactics_seconds"):
                        load_metric_rows(path)
                    with self.assertRaisesRegex(ValueError, "online_tactics_seconds"):
                        summarize_metrics([value])
            for field, bad in (("added_samples", -1), ("mixed_updates", True),
                               ("replay_size", 1.5), ("seconds", float("nan"))):
                with self.subTest(field=field, value=bad):
                    value = row(1)
                    value["online_tactics"] = {field: bad}
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "online_tactics"):
                        load_metric_rows(path)

    def test_legacy_and_online_rows_aggregate_only_known_teacher_reports(self):
        legacy = row(1)
        new = row(2, elapsed=9.0)
        new.update({"online_tactics_seconds": 4.0,
                    "online_tactics": {"enabled": True, "added_samples": 1,
                                       "replay_size": 7, "mixed_updates": 2,
                                       "tactical_rows_per_batch": 4}})
        summary = summarize_metrics([legacy, new])
        self.assertEqual(summary["phase_seconds"]["online_tactics"], 4.0)
        self.assertEqual(summary["phase_fractions"]["online_tactics"], 4.0 / 14.0)
        self.assertEqual(summary["online_tactics"]["report_covered_iterations"], 1)
        self.assertEqual(summary["online_tactics"]["added_samples"], 1)
        self.assertEqual(summary["online_tactics"]["latest_replay_size"], 7)

    def test_analyzer_keeps_original_phases_and_adds_online_teacher_output(self):
        value = row(1, elapsed=10.0)
        value.update({"online_tactics_seconds": 5.0,
                      "online_tactics": {"enabled": True, "added_samples": 3,
                                         "replay_size": 5, "mixed_updates": 2,
                                         "tactical_rows_per_batch": 4}})
        entry = Path(__file__).resolve().parents[1] / "examples" / "analyze_training.py"
        analyzer = runpy.run_path(str(entry))
        with tempfile.TemporaryDirectory(prefix="kingdom-metrics-") as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            output = io.StringIO()
            with patch("sys.argv", [str(entry), str(path)]), redirect_stdout(output):
                self.assertEqual(analyzer["main"](), 0)
        text = output.getvalue()
        self.assertIn("단계 비중: 자가대국 20.0%, 학습 10.0%, 평가 20.0%", text)
        self.assertIn("온라인 전술 teacher: 계산 시간 비중 50.0%", text)
        self.assertIn("전술 혼합: 새 채택 3개, 혼합 갱신 2회", text)
        self.assertIn("전술 표본 추출 8회, 최근 증명 버퍼 5개", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
