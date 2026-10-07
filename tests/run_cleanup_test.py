"""Exercise native archive/deletion safeguards with tiny workspace fixtures."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compact_logs", REPOSITORY / "examples/compact_training_logs.py")
PROGRAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROGRAM)


def metric():
    return {"iteration": 1, "generated_samples": 4, "replay_size": 4,
            "mean_self_play_plies": 2, "games_per_iteration": 2,
            "self_play_seconds": 1.0, "training_seconds": 0.1,
            "evaluation_seconds": 0.1, "elapsed_seconds": 1.2,
            "evaluation": {"games": 2, "wins": 1, "losses": 1,
                           "wins_as_black": 1, "wins_as_white": 0},
            "online_tactics": {"records": [{"outcome": "UNKNOWN", "proof_depth": 0,
                                           "budget_exhausted": True, "position": {"history": [[1, 2]]}}]}}


class RunCleanupTest(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="run_cleanup_test_", dir=REPOSITORY / "runs")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name) / "runs"
        self.root.mkdir()
        self.protected = self.root / "past-version"
        self.protected.mkdir()
        self.protected_file = self.protected / "metrics.jsonl"
        self.protected_file.write_text(json.dumps(metric()) + "\n", encoding="utf-8")

    def test_compaction_refuses_protected_and_outside_paths(self):
        original = self.protected_file.read_bytes()
        with self.assertRaisesRegex(ValueError, "outside past-version"):
            PROGRAM.compact_log(self.protected_file, runs_root=self.root, apply=True)
        outside = Path(self.fixture.name) / "metrics.jsonl"
        outside.write_bytes(original)
        with self.assertRaises(ValueError):
            PROGRAM.compact_log(outside, runs_root=self.root, apply=True)
        self.assertEqual(self.protected_file.read_bytes(), original)

    def test_dry_run_is_read_only_and_apply_keeps_performance_metrics(self):
        path = self.root / "metrics.jsonl"
        path.write_text(json.dumps(metric()) + "\n", encoding="utf-8")
        original = path.read_bytes()
        report = PROGRAM.compact_log(path, runs_root=self.root)
        self.assertFalse(report["applied"])
        self.assertEqual(path.read_bytes(), original)
        report = PROGRAM.compact_log(path, runs_root=self.root, apply=True)
        self.assertTrue(report["applied"])
        compact = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("records", compact["online_tactics"])
        self.assertEqual(compact["online_tactics"]["record_summary"]["budget_exhausted"], {"UNKNOWN": 1})
        self.assertEqual(list(self.root.glob('.compact-*')), [])

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 required")
    def test_native_cleanup_backs_up_and_checksums_before_deletion(self):
        old, active = self.root / "old", self.root / "active"
        old.mkdir(); active.mkdir()
        old_model = old / "latest.pt"
        old_model.write_bytes(b"tiny checkpoint fixture")
        log = active / "metrics.jsonl"
        log.write_text(json.dumps(metric()) + "\n", encoding="utf-8")
        original_log = log.read_bytes()
        protected_before = (self.protected_file.read_bytes(), self.protected_file.stat().st_mtime_ns)
        command = [shutil.which("pwsh"), "-NoProfile", "-File",
                   str(REPOSITORY / "examples/cleanup_runs.ps1"), "-RunsPath", str(self.root),
                   "-ArchiveRun", "old", "-CompactLogs", "-ArchiveName", "fixture.zip"]
        dry = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertTrue(old_model.exists())
        self.assertFalse((self.root / "archive").exists())
        done = subprocess.run([*command, "-Apply"], capture_output=True, text=True,
                              encoding="utf-8", timeout=45)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse(old.exists())
        self.assertEqual(protected_before,
                         (self.protected_file.read_bytes(), self.protected_file.stat().st_mtime_ns))
        with zipfile.ZipFile(self.root / "archive/fixture.zip") as archive:
            self.assertEqual(archive.read("old/latest.pt"), b"tiny checkpoint fixture")
            self.assertEqual(archive.read("active/metrics.jsonl"), original_log)
            self.assertFalse(any('past-version/' in name for name in archive.namelist()))
        report = json.loads((self.root / "archive/fixture.zip.json").read_text(encoding="utf-8"))
        self.assertTrue(report["protected_unchanged"])

    @unittest.skipUnless(shutil.which("pwsh"), "PowerShell 7 required")
    def test_native_cleanup_rejects_protected_and_parent_targets(self):
        for name in ("past-version", ".."):
            result = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File",
                str(REPOSITORY / "examples/cleanup_runs.ps1"), "-RunsPath", str(self.root),
                "-ArchiveRun", name, "-Apply"], capture_output=True, text=True,
                encoding="utf-8", timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(self.protected_file.exists())
            self.assertFalse((self.root / "archive").exists())


if __name__ == "__main__":
    unittest.main()
