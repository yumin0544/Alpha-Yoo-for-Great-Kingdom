"""Compare CPU neural PUCT with CUDA rules, CUDA trees and CUDA inference."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import my_board_engine as engine

from kingdom_ai import BatchedEvaluator, PolicyValueNet, load_model
from kingdom_ai.encoding import action_to_move
from kingdom_ai.gpu_puct import GpuPUCT, GpuPUCTOptions
from kingdom_ai.gpu_rules import GpuStateBatch, GPU_REASON
from kingdom_ai.gpu_runtime import nvrtc_version


REASONS = {"None_": 0, "Capture": 1, "Suicide": 2, "TwoPasses": 3}
REASON_NAMES = {0: "None", 1: "Capture", 2: "Suicide", 3: "TwoPasses"}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def cpu_snapshot(state):
    return [int(cell) for cell in state.board.cells] + [int(owner) for owner in state.ownership] + [
        int(state.to_play), state.remaining_stones(engine.Cell.Black),
        state.remaining_stones(engine.Cell.White), state.consecutive_passes,
        REASONS[state.result.reason.name], int(state.result.winner), state.result.captured_stones, 0,
    ]


def cpu_game(evaluator, simulations, root_noise, seed):
    state = engine.State()
    searcher = engine.PUCT(engine.PUCTOptions(
        simulations=simulations, seed=seed, dirichlet_epsilon=root_noise,
    ))
    history = []
    evaluations = searches = 0
    while not state.result.finished():
        result = searcher.search(state, evaluator)
        move = result.best_move
        if move is None or not state.play(move).accepted():
            raise RuntimeError("CPU PUCT returned an illegal move")
        history.append(81 if move.is_pass() else move.point.row * 9 + move.point.col)
        evaluations += result.network_evaluations
        searches += result.simulations
        if len(history) > 164:
            raise RuntimeError("CPU game exceeded the finite move bound")
    return {"state": cpu_snapshot(state), "history": history,
            "simulations": searches, "evaluations": evaluations}


def cpu_self_play(model, args):
    workers = min(args.workers, args.games)
    with BatchedEvaluator(model, device="cpu", max_batch_size=workers) as evaluator:
        evaluator(engine.State())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(evaluator, [engine.State() for _ in range(workers * 2)]))
        evaluator.reset_stats()
        play = partial(cpu_game, evaluator, args.simulations, args.root_noise)
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(play, ((args.seed + i) % 2 ** 64 for i in range(args.games))))
        seconds = perf_counter() - started
        stats = evaluator.stats
    if sum(row["evaluations"] for row in records) != stats["network_evaluations"]:
        raise RuntimeError("CPU network evaluation accounting differs")
    return records, seconds, {"backend": "cpu_puct", "workers": workers,
                              "max_batch_size": workers, **stats}


def gpu_self_play(model, args, batch_size):
    width = min(batch_size, args.games)
    searcher = GpuPUCT(model, GpuPUCTOptions(
        simulations=args.simulations, dirichlet_epsilon=args.root_noise, seed=args.seed,
    ))
    # Compilation, model transfer and one complete search are outside the timer.
    searcher.search(GpuStateBatch.initial(width))
    searcher.reset_seed()
    torch.cuda.synchronize()
    started = perf_counter()
    records = []
    for start in range(0, args.games, width):
        size = min(width, args.games - start)
        state = GpuStateBatch.initial(size)
        searches = torch.zeros(size, device="cuda", dtype=torch.int64)
        evaluations = torch.zeros_like(searches)
        histories = []
        for _ in range(164):
            active = state.states[:, GPU_REASON] == 0
            if not bool(active.any()):
                break
            result = searcher.search(state)
            accepted = state.play(result.actions)
            if bool((active & ~accepted).any()):
                raise RuntimeError("GPU PUCT returned an illegal move")
            searches += result.simulations
            evaluations += result.network_evaluations
            histories.append(result.actions)
        else:
            if bool((state.states[:, GPU_REASON] == 0).any()):
                raise RuntimeError("GPU game exceeded the finite move bound")
        snapshots = state.states.cpu().tolist()
        turns = torch.stack(histories).cpu().tolist()
        counts = torch.stack((searches, evaluations), dim=1).cpu().tolist()
        for lane, (snapshot, count) in enumerate(zip(snapshots, counts)):
            records.append({"state": snapshot, "history": [turn[lane] for turn in turns if turn[lane] >= 0],
                            "simulations": count[0], "evaluations": count[1]})
    torch.cuda.synchronize()
    seconds = perf_counter() - started
    return records, seconds, {"backend": "cuda_puct", "max_batch_size": width,
                              "rules_device": "cuda", "tree_device": "cuda",
                              "inference_device": "cuda"}


def check_gpu_records(records):
    """Replay GPU's completed moves in the CPU oracle outside the timing."""
    started = perf_counter()
    for row in records:
        oracle = engine.State()
        for action in row["history"]:
            if not oracle.play(action_to_move(action)).accepted():
                raise RuntimeError("GPU recorded a move rejected by the CPU oracle")
        if not oracle.result.finished() or cpu_snapshot(oracle) != row["state"]:
            raise RuntimeError("GPU completed state differs from the CPU rule oracle")
    return perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--games", type=positive, default=24)
    parser.add_argument("--workers", type=positive, default=12)
    parser.add_argument("--gpu-batch-sizes", nargs="+", type=positive, default=[12])
    parser.add_argument("--simulations", type=positive, default=32)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--cpu-threads", type=positive, default=1)
    parser.add_argument("--root-noise", type=float, default=0.0,
                        help="0 enables comparison without differing CPU/CUDA random streams")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64 or not 0 <= args.root_noise <= 1:
        parser.error("seed must be uint64 and root-noise must be finite in [0,1]")
    if "cuda" in args.backends and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.checkpoint:
        model = load_model(args.checkpoint, device="cpu")
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            model = PolicyValueNet()
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    print(json.dumps({"kind": "gpu_puct_environment", "torch": torch.__version__,
                      "cuda_runtime": torch.version.cuda, "nvrtc": nvrtc_version() if "cuda" in args.backends else None,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "model_config": model.model_config, "model_weight_sha256": digest.hexdigest(),
                      "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                      "games": args.games, "simulations_per_move": args.simulations,
                      "cpu_threads": args.cpu_threads, "cpu_workers": args.workers,
                      "root_noise": args.root_noise, "seed": args.seed, "precision": "float32",
                      "root_noise_rng": {"cpu": "mt19937_64_per_game_seed_plus_index",
                                         "cuda": "torch_cuda_generator_batched_stream"},
                      "tree_accumulator_precision": "float64", "tf32": False,
                      "repeats": args.repeats}), flush=True)
    configs = []
    if "cpu" in args.backends:
        configs.append(("cpu", min(args.workers, args.games)))
    if "cuda" in args.backends:
        configs += [("cuda", size) for size in dict.fromkeys(min(size, args.games) for size in args.gpu_batch_sizes)]
    measurements = {config: [] for config in configs}
    reference = None
    for repeat in range(args.repeats):
        offset = repeat % len(configs)
        for backend, width in configs[offset:] + configs[:offset]:
            if backend == "cpu":
                records, seconds, details = cpu_self_play(model, args)
            else:
                records, seconds, details = gpu_self_play(model, args, width)
            verified_seconds = check_gpu_records(records) if backend == "cuda" else None
            trajectories = [(row["history"], row["state"]) for row in records]
            if reference is None:
                reference = trajectories
            evaluations = sum(row["evaluations"] for row in records)
            result = {"kind": "completed_puct_self_play", "repeat": repeat + 1,
                      "games": len(records), "seconds": seconds, "games_per_second": len(records) / seconds,
                      "mean_plies": sum(len(row["history"]) for row in records) / len(records),
                      "simulations": sum(row["simulations"] for row in records), "network_evaluations": evaluations,
                      "network_evaluations_per_second": evaluations / seconds,
                      "endings": dict(Counter(REASON_NAMES[row["state"][166]] for row in records)),
                      "winners": dict(Counter(row["state"][167] for row in records)),
                      "changed_games_vs_first_config": sum(left != right for left, right in zip(trajectories, reference)),
                      "result_checksum": hashlib.sha256(json.dumps(trajectories, separators=(",", ":")).encode()).hexdigest(),
                      "cpu_oracle_verified": backend == "cuda", "oracle_verification_seconds": verified_seconds,
                      **details}
            measurements[(backend, width)].append(result)
            print(json.dumps(result), flush=True)
    print(json.dumps({"kind": "gpu_puct_scaling_summary", "measurements": [
        {"backend": backend, "max_batch_size": width,
         "median_games_per_second": median(row["games_per_second"] for row in rows),
         "median_seconds": median(row["seconds"] for row in rows),
         "median_mean_plies": median(row["mean_plies"] for row in rows),
         "median_network_evaluations_per_second": median(row["network_evaluations_per_second"] for row in rows)}
        for (backend, width), rows in measurements.items()
    ]}), flush=True)


if __name__ == "__main__":
    main()
