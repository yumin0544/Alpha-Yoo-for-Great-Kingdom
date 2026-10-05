"""Compare CUDA training batches with equal sample draws from a saved replay."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from kingdom_ai import Trainer
from kingdom_ai.training import train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--sample-draws", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.sample_draws < 1 or args.repeats < 1 or
            any(size < 1 or args.sample_draws % size for size in args.batch_sizes)):
        parser.error("Positive batch sizes must divide sample-draws; repeats must be positive")
    if args.output.exists():
        parser.error("Use a new output file")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.set_num_threads(1)
    source = Trainer.load_checkpoint(args.checkpoint, device="cuda")
    original = {name: tensor.clone() for name, tensor in source.model.state_dict().items()}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        metadata = {"type": "environment", "checkpoint": str(args.checkpoint.resolve()),
                    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                    "replay_size": len(source.replay), "sample_draws": args.sample_draws,
                    "timing": "replay sampling, device transfer and training; excludes warmup and saving"}
        output.write(json.dumps(metadata) + "\n")
        for repeat in range(args.repeats):
            # Rotate the order to reduce a fixed first/last-run bias.
            shift = repeat % len(args.batch_sizes)
            sizes = args.batch_sizes[shift:] + args.batch_sizes[:shift]
            for size in sizes:
                warm_model = deepcopy(source.model)
                warm_optimizer = torch.optim.Adam(warm_model.parameters())
                generator = torch.Generator().manual_seed(42)
                for _ in range(3):
                    train_step(warm_model, warm_optimizer, source.replay.sample(
                        size, generator=generator, device="cuda"))
                del warm_model, warm_optimizer
                model = deepcopy(source.model)
                optimizer = torch.optim.Adam(model.parameters(), lr=source.config.learning_rate,
                                             weight_decay=source.config.weight_decay)
                optimizer.load_state_dict(deepcopy(source.optimizer.state_dict()))
                generator.manual_seed(42 + repeat)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = perf_counter()
                steps = args.sample_draws // size
                for _ in range(steps):
                    loss = train_step(model, optimizer, source.replay.sample(
                        size, generator=generator, device="cuda"))
                torch.cuda.synchronize()
                seconds = perf_counter() - started
                if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                    raise RuntimeError("Non-finite trained parameters")
                if not all(int(state["step"]) == source.training_steps + steps
                           for state in optimizer.state.values()):
                    raise RuntimeError("Incorrect optimizer step count")
                row = {"type": "measurement", "repeat": repeat, "batch_size": size,
                       "steps": steps, "sample_draws": args.sample_draws, "seconds": seconds,
                       "samples_per_second": args.sample_draws / seconds, "final_loss": loss,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                del model, optimizer
        summary = {"type": "summary", "median_seconds": {
            str(size): median(row["seconds"] for row in rows if row["batch_size"] == size)
            for size in args.batch_sizes}}
        output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    if not all(torch.equal(value, original[name]) for name, value in source.model.state_dict().items()):
        raise RuntimeError("The source model changed")
    # These timed clones are disposable; the input checkpoint is never overwritten.


if __name__ == "__main__":
    main()
