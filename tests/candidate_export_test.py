"""Weight-only export validation, provenance and overwrite protection."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from kingdom_ai.checkpoint import export_training_model, load_model, save_model
from kingdom_ai.encoding import ACTION_SIZE, BOARD_SIZE, FEATURE_NAMES, FORMAT_VERSION
from kingdom_ai.model import PolicyValueNet


REPOSITORY = Path(__file__).resolve().parents[1]
PROGRAM = REPOSITORY / "examples" / "export_candidate.py"


class CandidateExportTest(unittest.TestCase):
    def setUp(self):
        fixture_root = REPOSITORY / "runs"
        fixture_root.mkdir(exist_ok=True)
        self.workspace = tempfile.TemporaryDirectory(prefix="candidate_export_test_", dir=fixture_root)
        self.addCleanup(self.workspace.cleanup)
        self.directory = Path(self.workspace.name)
        self.source = self.directory / "latest.pt"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.learner = PolicyValueNet(channels=4, residual_blocks=0)
            torch.manual_seed(29)
            self.champion = PolicyValueNet(channels=4, residual_blocks=0)
        # Other full-checkpoint state is intentionally not interpreted by this
        # weight-only utility: replay/optimizer restoration belongs to Trainer.
        self.payload = {
            "checkpoint_version": 7,
            "schema": {
                "format_version": FORMAT_VERSION, "board_size": BOARD_SIZE,
                "action_size": ACTION_SIZE, "feature_names": list(FEATURE_NAMES),
                "perspective": "to_play",
                "rules": {"suicide": "Loses", "own_territory_moves": False,
                          "single_edge_territory": True, "stones_per_player": 41},
            },
            "model_config": self.learner.model_config,
            "model": deepcopy(self.learner.state_dict()),
            "champion": deepcopy(self.champion.state_dict()),
            "progress": {"iteration": 515, "training_steps": 66176, "champion_version": 171},
            "config": {"self_play_backend": "cuda"},
            "optimizer": {}, "replay": {},
        }
        torch.save(self.payload, self.source)

    def assert_model_equal(self, actual, expected):
        self.assertEqual(actual.model_config, expected.model_config)
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(actual.state_dict()[name], value), name)
        self.assertEqual(next(actual.parameters()).device.type, "cpu")

    def test_default_exports_learner_and_separate_manifest_without_rng_or_cuda(self):
        source_before = self.source.read_bytes()
        rng_before = torch.get_rng_state().clone()
        output = self.directory / "candidate.pt"
        with patch("torch.cuda.init", side_effect=AssertionError("Must not initialize CUDA")):
            metadata = export_training_model(self.source, output)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assert_model_equal(load_model(output), self.learner)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(metadata, json.loads(output.with_suffix(".manifest.json").read_text("utf-8")))
        self.assertEqual(metadata["role"], "learner")
        self.assertEqual(metadata["iteration"], 515)
        self.assertEqual(metadata["training_steps"], 66176)
        self.assertEqual(metadata["source_checkpoint_sha256"], hashlib.sha256(source_before).hexdigest())
        self.assertEqual(metadata["output_checkpoint_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertRegex(metadata["weights_sha256"], r"^[a-f0-9]{64}$")
        portable = torch.load(output, weights_only=True)
        self.assertEqual(set(portable), {"format_version", "board_size", "action_size",
                                       "feature_names", "perspective", "model_config", "state_dict"})

    def test_explicit_champion_differs_from_learner(self):
        learner_meta = export_training_model(self.source, self.directory / "learner.pt")
        output = self.directory / "champion-copy.pt"
        champion_meta = export_training_model(self.source, output, role="champion")
        self.assert_model_equal(load_model(output), self.champion)
        self.assertNotEqual(learner_meta["weights_sha256"], champion_meta["weights_sha256"])

    def test_refuses_existing_paths_source_aliases_and_best(self):
        output = self.directory / "candidate.pt"
        manifest = output.with_suffix(".manifest.json")
        manifest.write_text("keep manifest", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_training_model(self.source, output)
        self.assertFalse(output.exists())
        self.assertEqual(manifest.read_text("utf-8"), "keep manifest")
        manifest.unlink()
        output.write_bytes(b"keep candidate")
        with self.assertRaises(FileExistsError):
            export_training_model(self.source, output)
        self.assertEqual(output.read_bytes(), b"keep candidate")
        with self.assertRaises(ValueError):
            export_training_model(self.source, self.source)
        with self.assertRaises(ValueError):
            export_training_model(self.source, self.directory / "best.pt")
        with self.assertRaises(ValueError):
            export_training_model(self.source, self.directory / "new.pt", manifest_path=self.source)
        with self.assertRaises(ValueError):
            export_training_model(self.source, self.directory / "new.pt",
                                  manifest_path=self.directory / "new.pt")

    def test_rejects_malformed_selected_weights_and_schema_without_outputs(self):
        key = next(iter(self.payload["model"]))
        for mutation in (
            lambda p: p["model"].pop(key),
            lambda p: p["model"].update({key: torch.zeros(1)}),
            lambda p: p["model"].update({key: p["model"][key].double()}),
            lambda p: p["model"][key].flatten().__setitem__(0, float("nan")),
            lambda p: p["schema"]["rules"].update(own_territory_moves=0),
            lambda p: p["schema"].update(perspective="black"),
            lambda p: p["progress"].update(training_steps=True),
            lambda p: p.update(checkpoint_version=99),
            lambda p: p.update(model_config={"channels": 4, "residual_blocks": 0, "extra": 1}),
        ):
            malformed = deepcopy(self.payload)
            mutation(malformed)
            torch.save(malformed, self.source)
            output = self.directory / "invalid.pt"
            with self.assertRaises(ValueError):
                export_training_model(self.source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.json").exists())

    def test_refuses_portable_source_and_bad_role(self):
        portable = self.directory / "portable.pt"
        save_model(self.learner, portable)
        with self.assertRaises(ValueError):
            export_training_model(portable, self.directory / "candidate.pt")
        with self.assertRaises(ValueError):
            export_training_model(self.source, self.directory / "candidate.pt", role="model")

    def test_v8_weight_only_export(self):
        self.payload["checkpoint_version"] = 8
        torch.save(self.payload, self.source)
        output = self.directory / "v8.pt"
        metadata = export_training_model(self.source, output)
        self.assertEqual(metadata["source_checkpoint_version"], 8)
        self.assert_model_equal(load_model(output), self.learner)

    def test_source_change_is_detected_before_output_creation(self):
        output = self.directory / "candidate.pt"
        with patch("kingdom_ai.checkpoint._file_digest", side_effect=("first", "changed")):
            with self.assertRaisesRegex(RuntimeError, "changed during extraction"):
                export_training_model(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".manifest.json").exists())

    def test_manifest_write_failure_removes_only_new_outputs(self):
        output = self.directory / "candidate.pt"
        source_before = self.source.read_bytes()
        with patch("kingdom_ai.checkpoint.json.dump", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                export_training_model(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".manifest.json").exists())
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_cli_emits_manifest_and_refuses_second_export(self):
        output = self.directory / "cli-candidate.pt"
        arguments = [sys.executable, str(PROGRAM), "--checkpoint", str(self.source),
                     "--output", str(output)]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(arguments, cwd=REPOSITORY, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["role"], "learner")
        result = subprocess.run(arguments, cwd=REPOSITORY, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
