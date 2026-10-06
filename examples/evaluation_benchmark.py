"""Benchmark sequential and batched parallel model evaluation on one device."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import torch

from kingdom_ai import load_model
from kingdom_ai.evaluation import evaluate_models


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=positive_integer, default=20)
    parser.add_argument("--simulations", type=positive_integer, default=32)
    parser.add_argument("--workers", type=positive_integer, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--repeats", type=positive_integer, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-moves", type=int, default=6)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--tactical-checks", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--threads", type=positive_integer, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; select a new JSONL path")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be even and at least two")
    if not 0 <= args.seed < 2 ** 64:
        parser.error("--seed must be an unsigned 64-bit integer")
    if args.opening_moves < 0 or args.opening_temperature < 0:
        parser.error("Opening settings must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("An available CUDA device is required for --device cuda")

    torch.set_num_threads(args.threads)
    candidate = load_model(args.model, device=device)
    reference = deepcopy(candidate).to(device).eval()
    original_digest = model_digest(candidate)
    workers = list(dict.fromkeys([1, *args.workers]))
    settings = {
        "games": args.games,
        "simulations": args.simulations,
        "seed": args.seed,
        "opening_moves": args.opening_moves,
        "opening_temperature": args.opening_temperature,
        "tactical_checks": args.tactical_checks,
    }

    # Exclude first-use model/device setup from every measured configuration.
    evaluate_models(candidate, reference, games=2, simulations=2, seed=args.seed,
                    opening_moves=min(args.opening_moves, 2),
                    opening_temperature=args.opening_temperature,
                    tactical_checks=args.tactical_checks, workers=max(workers))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        def emit(row):
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()

        emit({
            "kind": "evaluation_workers_environment",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "model": str(args.model.resolve()),
            "model_config": candidate.model_config,
            "model_parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
            "model_weight_sha256": original_digest,
            "workers": workers,
            "repeats": args.repeats,
            "cpu_threads": torch.get_num_threads(),
            "logical_processors": os.cpu_count(),
            **settings,
            "timing_includes": ["game rules", "PUCT", "model inference", "tactical checks"],
            "timing_excludes": ["process/model/device startup", "warm-up", "JSONL writes"],
        })
        for repeat in range(args.repeats):
            expected = None
            parallel_workers = workers[1:]
            offset = repeat % len(parallel_workers) if parallel_workers else 0
            measurement_order = [1] + parallel_workers[offset:] + parallel_workers[:offset]
            for worker_count in measurement_order:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
                result = evaluate_models(
                    candidate, reference, workers=worker_count,
                    **{**settings, "seed": (args.seed + repeat) % (2 ** 64)},
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                if expected is None:
                    expected = result
                if model_digest(candidate) != original_digest or model_digest(reference) != original_digest:
                    raise RuntimeError("Evaluation modified model weights")
                emit({
                    "kind": "completed_evaluation_workers_measurement",
                    "repeat": repeat + 1,
                    "workers": worker_count,
                    "effective_workers": min(worker_count, args.games),
                    "elapsed_seconds": elapsed,
                    "games_per_second": args.games / elapsed,
                    "matches_sequential_result": result == expected,
                    "result": {**asdict(result), "win_rate": result.win_rate},
                    "peak_torch_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
