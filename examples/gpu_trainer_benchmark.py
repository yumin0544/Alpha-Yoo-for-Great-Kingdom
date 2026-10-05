"""Measure completed GPU Trainer iterations while varying self-play batch width.

Each measurement starts from the same CPU model weights and a fresh replay
buffer and optimizer. Compilation, warm-up and Trainer construction are outside
the iteration timer. Iterations include CPU replay validation, training and
evaluation, and exclude checkpoint/file writes. Batch widths consume random
streams differently, so completed game trajectories need not match.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
from statistics import median

import torch

from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig, load_model
from kingdom_ai.gpu_puct import GpuPUCT, GpuPUCTOptions
from kingdom_ai.gpu_rules import GpuStateBatch
from kingdom_ai.gpu_runtime import nvrtc_version
from kingdom_ai.training import TrainingBatch, train_step


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


def clean_cuda():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def warm_training(source):
    """Warm forward, backward and Adam with an independent throwaway model."""
    model = deepcopy(source).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    features, masks = GpuStateBatch.initial(64).encode()
    policy = masks.to(torch.float32)
    policy /= policy.sum(dim=1, keepdim=True)
    batch = TrainingBatch(features, masks, policy, torch.zeros(64, device="cuda"))
    train_step(model, optimizer, batch)
    torch.cuda.synchronize()


def warm_search(source, width):
    searcher = GpuPUCT(source, GpuPUCTOptions(
        simulations=2, dirichlet_epsilon=0.25, seed=42,
    ))
    searcher.search(GpuStateBatch.initial(width))
    torch.cuda.synchronize()


def memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def verify_completed(trainer, metrics, source, source_digest):
    config = trainer.config
    if (trainer.iteration != 1 or trainer.self_play_games != config.games_per_iteration
            or trainer.training_steps != config.train_steps_per_iteration
            or sum(metrics["self_play_endings"].values()) != config.games_per_iteration
            or metrics["generated_samples"] < 1
            or len(trainer.replay) != min(config.replay_capacity, metrics["generated_samples"])
            or metrics["evaluation"]["games"] != config.evaluation_games):
        raise RuntimeError("Completed Trainer counters or replay contents are inconsistent")
    updated = {name: tensor.detach().cpu() for name, tensor in trainer.model.state_dict().items()}
    if any(not bool(torch.isfinite(tensor).all()) for tensor in updated.values()):
        raise RuntimeError("Training produced non-finite model parameters")
    updated_digest = model_digest(trainer.model)
    if updated_digest == source_digest:
        raise RuntimeError("Trainer did not update the initial model weights")
    optimizer_states = list(trainer.optimizer.state.values())
    if (not optimizer_states or any(
            float(state["step"]) != config.train_steps_per_iteration
            for state in optimizer_states)):
        raise RuntimeError("Adam update counts differ from the requested training steps")
    if model_digest(source) != source_digest:
        raise RuntimeError("Benchmark modified the shared initial model")
    return {
        "completed_counters_verified": True,
        "finite_updated_weights_verified": True,
        "initial_model_unchanged_verified": True,
        "updated_model_weight_sha256": updated_digest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=positive_integer, default=1024)
    parser.add_argument("--batch-sizes", type=positive_integer, nargs="+",
                        default=[128, 256, 512, 1024])
    parser.add_argument("--repeats", type=positive_integer, default=1)
    parser.add_argument("--simulations", type=positive_integer, default=32)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSONL file; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; select a new JSONL path")
    if args.simulations > 2 ** 31 - 2:
        parser.error("CUDA simulations must not exceed 2147483646")
    if not torch.cuda.is_available():
        parser.error("An available CUDA device is required")
    torch.set_num_threads(1)
    if args.initial_model:
        source = load_model(args.initial_model, device="cpu")
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            source = PolicyValueNet(channels=32, residual_blocks=2)
    source_digest = model_digest(source)
    widths = list(dict.fromkeys(args.batch_sizes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        def emit(row):
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            output.write(line + "\n")
            output.flush()

        emit({
            "kind": "gpu_trainer_batch_environment",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "nvrtc": nvrtc_version(), "gpu": torch.cuda.get_device_name(),
            "gpu_compute_capability": torch.cuda.get_device_capability(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "model_config": source.model_config,
            "model_parameter_count": sum(parameter.numel() for parameter in source.parameters()),
            "initial_model": str(args.initial_model.resolve()) if args.initial_model else None,
            "initial_model_weight_sha256": source_digest,
            "games_per_iteration": args.games, "batch_sizes": widths,
            "repeats": args.repeats, "simulations_per_move": args.simulations,
            "seed_per_repeat": [42 + repeat for repeat in range(args.repeats)],
            "cpu_threads": torch.get_num_threads(),
            "train_steps": 8, "training_minibatch_size": 64,
            "replay_capacity_positions": 10000,
            "evaluation_games": 2, "evaluation_simulations_per_move": 32,
            "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "temperature": 1.0,
            "network_precision": "float32", "tree_accumulator_precision": "float64",
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "timing_includes": ["GPU self-play", "GPU history collection", "CPU transfer",
                                "CPU replay validation and storage", "Adam updates", "evaluation"],
            "timing_excludes": ["process and device startup", "first NVRTC compilation",
                                "search and training warm-up", "Trainer construction",
                                "checkpoint and JSONL writes", "post-run verification"],
            "memory_scope": "Torch allocator peaks; CUDA modules/context and other processes excluded",
            "trajectory_comparison": "Batch grouping changes random-stream consumption and trajectories",
            "evaluation_scope": "Two games verify integration; absolute playing strength is not measured",
        })
        warm_training(source)
        clean_cuda()
        measurements = {width: [] for width in widths}
        for repeat in range(args.repeats):
            # Rotate order on repeated runs to reduce systematic order effects.
            offset = repeat % len(widths)
            for width in widths[offset:] + widths[:offset]:
                trainer = None
                phase = "warmup"
                try:
                    warm_search(source, min(width, args.games))
                    clean_cuda()
                    config = TrainingConfig(
                        games_per_iteration=args.games, simulations=args.simulations,
                        self_play_backend="cuda", self_play_batch_size=width,
                        seed=42 + repeat, train_steps_per_iteration=8, batch_size=64,
                        replay_capacity=10000, evaluation_games=2, evaluation_simulations=32,
                        dirichlet_epsilon=0.25, temperature=1.0,
                    )
                    phase = "initialization"
                    trainer = Trainer(config, model=source, device="cuda")
                    torch.cuda.synchronize()
                    before = memory_snapshot()
                    torch.cuda.reset_peak_memory_stats()
                    phase = "iteration"
                    metrics = trainer.run_iteration()
                    torch.cuda.synchronize()
                    after = memory_snapshot()
                    peak_allocated = torch.cuda.max_memory_allocated()
                    peak_reserved = torch.cuda.max_memory_reserved()
                    phase = "verification"
                    verified = verify_completed(trainer, metrics, source, source_digest)
                    row = {
                        "kind": "completed_gpu_trainer_batch_iteration",
                        "repeat": repeat + 1, "seed": config.seed,
                        "self_play_batch_size": width,
                        "effective_self_play_batch_size": min(width, args.games),
                        "games": args.games, "config": asdict(config),
                        "metrics": metrics, "memory_before": before, "memory_after": after,
                        "peak_torch_allocated_bytes": peak_allocated,
                        "peak_torch_reserved_bytes": peak_reserved,
                        **verified,
                    }
                    measurements[width].append(row)
                    emit(row)
                except torch.cuda.OutOfMemoryError as error:
                    emit({
                        "kind": "gpu_trainer_batch_out_of_memory", "repeat": repeat + 1,
                        "seed": 42 + repeat, "self_play_batch_size": width,
                        "effective_self_play_batch_size": min(width, args.games),
                        "phase": phase, "error": str(error),
                    })
                finally:
                    trainer = None
                    clean_cuda()
        if args.repeats > 1:
            emit({
                "kind": "gpu_trainer_batch_median_summary",
                "measurements": [{
                    "self_play_batch_size": width, "completed_repeats": len(rows),
                    "requested_repeats": args.repeats,
                    "median_self_play_games_per_second": median(
                        row["metrics"]["self_play_games_per_second"] for row in rows),
                    "median_iteration_games_per_second": median(
                        row["metrics"]["iteration_games_per_second"] for row in rows),
                    "median_self_play_seconds": median(
                        row["metrics"]["self_play_seconds"] for row in rows),
                    "median_elapsed_seconds": median(
                        row["metrics"]["elapsed_seconds"] for row in rows),
                    "median_mean_self_play_plies": median(
                        row["metrics"]["mean_self_play_plies"] for row in rows),
                    "max_peak_torch_allocated_bytes": max(
                        row["peak_torch_allocated_bytes"] for row in rows),
                    "max_peak_torch_reserved_bytes": max(
                        row["peak_torch_reserved_bytes"] for row in rows),
                } for width, rows in measurements.items() if rows],
                "uncompleted_batch_sizes": [width for width, rows in measurements.items() if not rows],
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
