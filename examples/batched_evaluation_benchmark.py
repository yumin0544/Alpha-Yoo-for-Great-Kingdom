"""Compare legacy evaluation against C++ leaf batches and retained subtrees.

The default runs nine configurations on the same immutable model pair: legacy
with 12 game workers, then leaf batches 1/4/8/16 with reuse off/on. Configured
simulations are NEW simulations per move, not an inherited-tree total. Batched
or retained searches can choose different moves: wall time and total plies must
both be reported rather than claiming identical-work acceleration.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter

import torch
import my_board_engine as engine

from kingdom_ai import Trainer, load_model
from kingdom_ai.encoding import encode_state
from kingdom_ai.evaluation import evaluate_models


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def nonnegative_real(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite, non-negative number")
    return number


def positive_real(value):
    number = nonnegative_real(value)
    if number == 0:
        raise argparse.ArgumentTypeError("Expected a finite, positive number")
    return number


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_models(args, device):
    """Load frozen evaluation models without advancing or saving a trainer.

    Training sources go through the public, strict resume validator. Keeping
    only model references releases the large replay and optimizer before any
    measured evaluation. The optional memory-mapped metadata read avoids a
    second full copy of the replay tensors just to record the source version.
    """
    if args.training_checkpoint is not None:
        trainer = Trainer.load_checkpoint(args.training_checkpoint, device=device)
        candidate, reference = trainer.model.eval(), trainer.champion.eval()
        metadata = {
            "source_kind": "training_checkpoint",
            "training_checkpoint": str(args.training_checkpoint.resolve()),
            "candidate_model": None,
            "reference_model": None,
            "source_model_roles": ["learner", "champion"],
            "source_progress": {
                "iteration": trainer.iteration,
                "self_play_games": trainer.self_play_games,
                "training_steps": trainer.training_steps,
                "champion_version": trainer.champion_version,
            },
            "source_training_config": asdict(trainer.config),
        }
        del trainer
        payload = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        metadata["source_checkpoint_version"] = payload["checkpoint_version"]
        del payload
        return candidate, reference, metadata
    candidate = load_model(args.model, device=device).eval()
    reference = (load_model(args.reference_model, device=device).eval()
                 if args.reference_model else deepcopy(candidate).to(device).eval())
    return candidate, reference, {
        "source_kind": "inference_models",
        "training_checkpoint": None,
        "candidate_model": str(args.model.resolve()),
        "reference_model": str(args.reference_model.resolve()) if args.reference_model else None,
        "source_model_roles": ["candidate", "reference"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="Candidate inference model")
    source.add_argument("--training-checkpoint", type=Path,
                        help="Strictly load the saved learner/champion pair; no training or save")
    parser.add_argument("--reference-model", type=Path,
                        help="Reference inference model; defaults to an identical frozen copy")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=positive_integer, default=20)
    parser.add_argument("--simulations", type=positive_integer, default=32)
    parser.add_argument("--c-puct", type=positive_real, default=1.5)
    parser.add_argument("--workers", type=positive_integer, default=12)
    parser.add_argument("--leaf-batch-sizes", type=positive_integer, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--reuse-modes", choices=("off", "on"), nargs="+", default=["off", "on"])
    parser.add_argument("--backends", choices=("legacy", "batched_cpp"), nargs="+",
                        default=["legacy", "batched_cpp"])
    parser.add_argument("--repeats", type=positive_integer, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-moves", type=int, default=6)
    parser.add_argument("--opening-temperature", type=nonnegative_real, default=1.0)
    parser.add_argument("--tactical-checks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=positive_integer, default=1)
    args = parser.parse_args()
    if args.training_checkpoint is not None and args.reference_model is not None:
        parser.error("--reference-model cannot be combined with --training-checkpoint")
    if args.output.exists():
        parser.error("Output already exists; select a new JSONL path")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be even and at least two")
    if not 0 <= args.seed < 2 ** 64:
        parser.error("--seed must be an unsigned 64-bit integer")
    if args.opening_moves < 0:
        parser.error("--opening-moves must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("An available CUDA device is required for --device cuda")

    torch.set_num_threads(args.threads)
    paths = tuple(dict.fromkeys(path.resolve() for path in (
                                   args.training_checkpoint, args.model, args.reference_model)
                               if path is not None))
    original_files = {str(path): file_digest(path) for path in paths}
    candidate, reference, source_metadata = load_source_models(args, device)
    models = (candidate, reference)
    original_digests = tuple(model_digest(model) for model in models)
    identical_models = original_digests[0] == original_digests[1]
    settings = dict(
        games=args.games, simulations=args.simulations, c_puct=args.c_puct, workers=args.workers,
        seed=args.seed, opening_moves=args.opening_moves,
        opening_temperature=args.opening_temperature, tactical_checks=args.tactical_checks,
    )
    configurations = []
    for backend in dict.fromkeys(args.backends):
        if backend == "legacy":
            configurations.append(dict(backend="legacy", leaf_batch_size=1, reuse_tree=False))
        else:
            for size in dict.fromkeys(args.leaf_batch_sizes):
                for reuse_mode in dict.fromkeys(args.reuse_modes):
                    configurations.append(dict(backend="batched_cpp", leaf_batch_size=size,
                                               reuse_tree=reuse_mode == "on"))

    # First-use CUDA libraries and shape-specific convolution setup are outside
    # measurements. No game/training/checkpoint is created by this warm-up.
    encoded = encode_state(engine.State())
    warmup_sizes = {1, min(args.workers, args.games)}
    warmup_sizes.update(min(args.workers, args.games) * config["leaf_batch_size"]
                        for config in configurations)
    with torch.inference_mode():
        for model in models:
            for size in sorted(warmup_sizes):
                features = encoded.features.unsqueeze(0).expand(size, -1, -1, -1).contiguous().to(device)
                model(features)
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
            "kind": "batched_evaluation_environment",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            **source_metadata,
            "model_configs": [model.model_config for model in models],
            "parameter_counts": [sum(parameter.numel() for parameter in model.parameters()) for model in models],
            "model_weight_sha256": list(original_digests), "model_files_sha256": original_files,
            "identical_models": identical_models,
            "configurations": configurations, "repeats": args.repeats,
            "cpu_threads": torch.get_num_threads(), "logical_processors": os.cpu_count(),
            **settings,
            "timing_includes": ["inference service setup", "game rules", "PUCT", "input encoding",
                                "model inference", "tactical checks", "service shutdown"],
            "timing_excludes": ["process/model/device startup", "model warm-up", "digests", "JSONL writes"],
            "comparison_note": (
                "Leaf batching/tree reuse may change trajectories. Same configured NEW simulation budget, "
                "models and paired seeds are held fixed; wall time alone is not equal-work speedup. "
                "Identical-model 50% is a pairing sanity check, not evidence of unchanged playing strength."
            ),
        })
        for repeat in range(args.repeats):
            # Rotate the order to expose systematic warm-up/thermal order effects.
            offset = repeat % len(configurations)
            order = configurations[offset:] + configurations[:offset]
            baseline_seconds = None
            baseline_result = None
            for config in order:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                diagnostics = {}
                started = perf_counter()
                result = evaluate_models(
                    candidate, reference, **config, diagnostics=diagnostics,
                    **{**settings, "seed": (args.seed + repeat) % (2 ** 64)},
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                if identical_models and result.wins != result.games // 2:
                    raise RuntimeError("Identical models lost paired-color 50% symmetry")
                if tuple(model_digest(model) for model in models) != original_digests:
                    raise RuntimeError("Evaluation modified model weights")
                if config["backend"] == "legacy":
                    baseline_seconds, baseline_result = elapsed, result
                emit({
                    "kind": "completed_batched_evaluation_measurement",
                    "repeat": repeat + 1, "seed": (args.seed + repeat) % (2 ** 64),
                    **config, "workers": args.workers, "effective_workers": min(args.workers, args.games),
                    "simulations_per_move": args.simulations, "elapsed_seconds": elapsed,
                    "games_per_second": result.games / elapsed, "plies_per_second": result.total_plies / elapsed,
                    "wall_time_ratio_to_measured_legacy": (
                        baseline_seconds / elapsed if baseline_seconds is not None else None
                    ),
                    "matches_measured_legacy_result": (
                        result == baseline_result if baseline_result is not None else None
                    ),
                    "result": {**asdict(result), "win_rate": result.win_rate},
                    "diagnostics": diagnostics,
                    "model_weights_unchanged": True,
                    "peak_torch_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                })
        current_files = {str(path): file_digest(path) for path in paths}
        if current_files != original_files:
            raise RuntimeError("An input model file changed during benchmarking")
        emit({"kind": "batched_evaluation_integrity", "input_files_unchanged": True,
              "model_weights_unchanged": True, "model_files_sha256": current_files})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
