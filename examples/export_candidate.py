"""Export latest learner (or champion) weights without resuming training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kingdom_ai.checkpoint import export_training_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract CPU inference weights and provenance from a full latest.pt")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Full saved training checkpoint; source is never modified")
    parser.add_argument("--output", type=Path, required=True,
                        help="New portable model path, not best.pt; existing paths are refused")
    parser.add_argument("--role", choices=("learner", "champion"), default="learner",
                        help="Selected weights (default: latest learner, not champion)")
    parser.add_argument("--manifest", type=Path,
                        help="New provenance JSON path (default: output stem + .manifest.json)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        provenance = export_training_model(
            args.checkpoint, args.output, role=args.role, manifest_path=args.manifest)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(provenance, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
