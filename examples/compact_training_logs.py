"""Compact backed-up Trainer JSONL files; never touch past-version or checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "python"))

from kingdom_ai.metrics import compact_metric_row, load_metric_rows, summarize_metrics


def compact_log(path, *, runs_root, apply=False):
    # OneDrive placeholders under restricted Windows sessions can deny strict
    # GetFinalPathNameByHandle even when normal stat/open work. Check existence
    # separately, and reject actual symlinks/junctions rather than cloud tags.
    root = Path(runs_root).resolve()
    if not root.is_dir() or not root.is_relative_to(REPOSITORY) or root.name != "runs":
        raise ValueError("runs root must be a runs directory inside this repository")
    source = Path(path).resolve()
    if (not source.is_file() or not source.is_relative_to(root)
            or "past-version" in [part.lower() for part in source.relative_to(root).parts]
            or source.name != "metrics.jsonl"):
        raise ValueError("Only metrics.jsonl outside past-version inside runs may be compacted")
    cursor = Path(path).absolute()
    while cursor != REPOSITORY:
        if cursor.is_symlink() or cursor.is_junction():
            raise ValueError("Linked log paths may not be compacted")
        if cursor.parent == cursor:
            raise ValueError("Log path escapes this repository")
        cursor = cursor.parent
    original = source.read_bytes()
    rows, duplicates = load_metric_rows(source)
    before = summarize_metrics(rows, duplicate_rows=duplicates)
    objects = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
    compacted = [compact_metric_row(row) for row in objects]
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")) + "\n" for row in compacted)
    encoded = text.encode("utf-8")
    report = {"path": str(source), "before_bytes": len(original),
              "after_bytes": len(encoded), "rows": len(objects), "applied": False}
    if not apply or encoded == original:
        return report
    # A validated same-directory temporary file prevents partial-log truncation.
    descriptor, name = tempfile.mkstemp(prefix=".compact-", suffix=".jsonl", dir=source.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        loaded, count = load_metric_rows(temporary)
        if summarize_metrics(loaded, duplicate_rows=count) != before:
            raise ValueError("Compaction changed the training performance summary")
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(original).digest():
            raise RuntimeError("Log changed during compaction; stop training before cleanup")
        os.replace(temporary, source)
        report["applied"] = True
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--runs-root", type=Path, default=REPOSITORY / "runs")
    parser.add_argument("--apply", action="store_true", help="requires an external backup first")
    args = parser.parse_args()
    try:
        for path in args.paths:
            print(json.dumps(compact_log(path, runs_root=args.runs_root, apply=args.apply),
                             ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
