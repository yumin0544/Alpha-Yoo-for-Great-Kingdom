"""Compare completed neural PUCT games on CPU and CUDA with batched inference."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import torch
import my_board_engine as engine

from kingdom_ai import PolicyValueNet, encode_state, load_model, masked_policy
from kingdom_ai.batching import BatchedEvaluator


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def completed_game(evaluator, simulations, seed):
    game = engine.State()
    searcher = engine.PUCT(engine.PUCTOptions(
        simulations=simulations, seed=seed, dirichlet_epsilon=0.25,
    ))
    plies = searches = evaluations = 0
    while not game.result.finished():
        result = searcher.search(game, evaluator)
        if result.best_move is None or not game.play(result.best_move).accepted():
            raise RuntimeError("PUCT did not produce an accepted move")
        plies += 1
        searches += result.simulations
        evaluations += result.network_evaluations
        if plies > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Game exceeded the finite move bound")
    return (plies, searches, evaluations, game.result.reason.name,
            game.result.winner.name, game.board.to_string())


def self_play(model, device, games, simulations, workers, batch_size, wait_ms, seed):
    workers = min(workers, games)
    with BatchedEvaluator(model, device=device, max_batch_size=batch_size,
                          max_wait_ms=wait_ms if batch_size > 1 else 0.0) as evaluator:
        # Warm both the single-state and concurrent-batch inference paths.
        evaluator(engine.State())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(evaluator, (engine.State() for _ in range(workers * 2))))
        evaluator.reset_stats()
        synchronize(device)
        started = perf_counter()
        play = partial(completed_game, evaluator, simulations)
        seeds = ((seed + index) % 2 ** 64 for index in range(games))
        if workers == 1:
            results = list(map(play, seeds))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(play, seeds))
        synchronize(device)
        elapsed = perf_counter() - started
        stats = evaluator.stats

    evaluations = sum(result[2] for result in results)
    if stats["network_evaluations"] != evaluations:
        raise RuntimeError("Network evaluation accounting differs from completed searches")
    plies = sum(result[0] for result in results)
    total_searches = sum(result[1] for result in results)
    endings = Counter(result[3] for result in results)
    winners = Counter(result[4] for result in results)
    checksum = hashlib.sha256(json.dumps(results, separators=(",", ":")).encode()).hexdigest()
    return {
        "kind": "completed_neural_self_play", "device": device,
        "worker_threads": workers, "cpu_threads": torch.get_num_threads(),
        "max_batch_size": batch_size, "max_wait_ms": wait_ms if batch_size > 1 else 0.0,
        "simulations_per_move": simulations, "games": games, "seconds": elapsed,
        "games_per_second": games / elapsed, "mean_plies": plies / games,
        "simulations": total_searches, "network_evaluations": evaluations,
        "network_evaluations_per_second": evaluations / elapsed,
        "endings": dict(endings), "winners": dict(winners),
        "result_checksum": checksum, **stats,
        "_results": results,
    }


def fixed_inference(model, device, batch_size, iterations):
    """Fixed encoded positions separate inference speed from game trajectories."""
    encoded = encode_state(engine.State())
    features = encoded.features.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    masks = encoded.legal_mask.unsqueeze(0).repeat(batch_size, 1)
    network = deepcopy(model).to(device=device, dtype=torch.float32).eval()

    def evaluate():
        logits, values = network(features.to(device))
        packed = torch.cat((logits, values.unsqueeze(1)), dim=1).cpu()
        return masked_policy(packed[:, :-1], masks), packed[:, -1]

    with torch.inference_mode():
        for _ in range(5):
            evaluate()
        synchronize(device)
        started = perf_counter()
        for _ in range(iterations):
            policies, values = evaluate()
        synchronize(device)
        elapsed = perf_counter() - started
    return {
        "kind": "fixed_position_inference", "device": device,
        "cpu_threads": torch.get_num_threads(), "batch_size": batch_size,
        "iterations": iterations, "seconds": elapsed,
        "evaluations_per_second": batch_size * iterations / elapsed,
        "_policies": policies, "_values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--workers", type=positive_integer, default=12)
    parser.add_argument("--batch-sizes", type=positive_integer, nargs="+", default=[1, 12])
    parser.add_argument("--cpu-threads", type=positive_integer, nargs="+", default=[1])
    parser.add_argument("--games", type=positive_integer, default=24)
    parser.add_argument("--simulations", type=positive_integer, default=64)
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--wait-ms", type=float, default=0.0,
                        help="Batch collection wait; 0 avoids timer delays on Windows")
    parser.add_argument("--inference-iterations", type=positive_integer, default=100)
    parser.add_argument("--channels", type=positive_integer, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64:
        parser.error("seed must be an unsigned 64-bit integer")
    if args.residual_blocks < 0 or not 0 <= args.wait_ms < float("inf"):
        parser.error("residual-blocks and finite wait-ms must be non-negative")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install a compatible CUDA PyTorch build or use --devices cpu")

    torch.set_num_threads(args.cpu_threads[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.checkpoint is not None:
        model = load_model(args.checkpoint, device="cpu")
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            model = PolicyValueNet(args.channels, args.residual_blocks)
    weight_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        weight_hash.update(name.encode())
        weight_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    print(json.dumps({
        "kind": "neural_environment", "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "model_config": model.model_config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_weight_sha256": weight_hash.hexdigest(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_loaded": args.checkpoint is not None,
        "seed": args.seed, "seed_strategy": "per_game", "precision": "float32",
        "tf32": False, "root_noise": 0.25, "move_selection": "best_move",
        "workers": args.workers, "batch_sizes": args.batch_sizes,
        "cpu_threads": args.cpu_threads, "repeats": args.repeats,
    }), flush=True)

    configurations = list(dict.fromkeys(
        (device, threads, batch) for threads in args.cpu_threads
        for device in args.devices for batch in args.batch_sizes
    ))
    measurements = {config: [] for config in configurations}
    reference_results = None
    reference_fixed = {}
    for repeat in range(args.repeats):
        offset = repeat % len(configurations)
        for device, threads, batch in configurations[offset:] + configurations[:offset]:
            torch.set_num_threads(threads)
            config = (device, threads, batch)
            if repeat == 0:
                fixed = fixed_inference(model, device, batch, args.inference_iterations)
                policies, values = fixed.pop("_policies"), fixed.pop("_values")
                key = (threads, batch)
                if key not in reference_fixed:
                    reference_fixed[key] = (policies, values)
                expected_policy, expected_value = reference_fixed[key]
                fixed["max_policy_difference"] = (policies - expected_policy).abs().max().item()
                fixed["max_value_difference"] = (values - expected_value).abs().max().item()
                if not torch.allclose(policies, expected_policy, rtol=1e-4, atol=1e-5) or not torch.allclose(
                        values, expected_value, rtol=1e-4, atol=1e-5):
                    raise RuntimeError("CPU/CUDA fixed-position inference differs beyond tolerance")
                print(json.dumps(fixed), flush=True)
            result = self_play(model, device, args.games, args.simulations,
                               args.workers, batch, args.wait_ms, args.seed)
            outcomes = result.pop("_results")
            if reference_results is None:
                reference_results = outcomes
            # Floating-point differences may change PUCT moves; report them explicitly.
            result["changed_games_vs_first_config"] = sum(
                actual != expected for actual, expected in zip(outcomes, reference_results)
            )
            result["repeat"] = repeat + 1
            measurements[config].append(result)
            print(json.dumps(result), flush=True)

    reference_config = next((config for config in configurations if config[0] == "cpu" and config[2] == 1),
                            configurations[0])
    reference_rate = median(r["games_per_second"] for r in measurements[reference_config])
    cpu_rates = {config: median(r["games_per_second"] for r in rows)
                 for config, rows in measurements.items() if config[0] == "cpu"}
    best_cpu_rate = max(cpu_rates.values()) if cpu_rates else None
    summary = []
    for (device, threads, batch), rows in measurements.items():
        rate = median(r["games_per_second"] for r in rows)
        summary.append({
            "device": device, "cpu_threads": threads, "max_batch_size": batch,
            "median_games_per_second": rate,
            "median_mean_plies": median(r["mean_plies"] for r in rows),
            "median_network_evaluations_per_second": median(r["network_evaluations_per_second"] for r in rows),
            "median_mean_batch_size": median(r["mean_batch_size"] for r in rows),
            "speedup_vs_reference": rate / reference_rate,
            "speedup_vs_matching_cpu": (
                rate / cpu_rates[("cpu", threads, batch)]
                if ("cpu", threads, batch) in cpu_rates else None
            ),
            "speedup_vs_best_cpu": rate / best_cpu_rate if best_cpu_rate is not None else None,
        })
    print(json.dumps({
        "kind": "neural_scaling_summary", "games_per_repeat": args.games,
        "simulations_per_move": args.simulations, "repeats": args.repeats,
        "reference_config": reference_config,
        "best_config": max(summary, key=lambda row: row["median_games_per_second"]),
        "measurements": summary,
    }), flush=True)


if __name__ == "__main__":
    main()
