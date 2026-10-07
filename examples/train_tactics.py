"""Bounded proof-labelled tactical fine-tuning without champion promotion.

Run with the freshly built/installed C++ solver. Standard Go games are not
training labels: capture, suicide, permanent houses and the score rule are
always resolved by this project's verified Kingdom engine.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
import my_board_engine as engine

from kingdom_ai.checkpoint import load_model, save_model
from kingdom_ai.loop import Trainer
from kingdom_ai.tactical_training import (
    collect_certified_samples, fine_tune_tactics, model_digest, split_tactical_samples,
)


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("A positive integer is required")
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description="Kingdom proof-labelled tactical curriculum")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--resume-training", type=Path,
                        help="Full latest.pt; preserve optimizer, progress, replay and champion")
    source.add_argument("--initial-model", type=Path,
                        help="Inference model; train an independent candidate without original replay")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (never overwrite)")
    parser.add_argument("--casebook", type=Path, help="Extra JSON cases, appended to built-in curriculum")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--steps", type=positive_integer, default=256)
    parser.add_argument("--batch-size", type=positive_integer, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--tactical-fraction", type=float, default=0.25,
                        help="With replay, certified tactics fraction per minibatch (default 0.25)")
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42, help="Original-family split and standalone sampling seed")
    parser.add_argument("--max-depth", type=positive_integer, default=12)
    parser.add_argument("--max-nodes", type=positive_integer, default=100000)
    parser.add_argument("--time-limit-ms", type=positive_integer, default=1000)
    parser.add_argument("--max-cases", type=positive_integer, default=64)
    parser.add_argument("--generation-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate/audit proofs and split only; do not load or train a model")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output to preserve earlier files")
    if not 0 < args.tactical_fraction <= 1 or not 0 < args.heldout_fraction < 1:
        parser.error("tactical-fraction must be in (0,1], heldout-fraction in (0,1)")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.generation_seconds) or args.generation_seconds <= 0
            or not 0 <= args.seed < 2 ** 64):
        parser.error("Invalid learning rate, generation budget or seed")
    source = args.resume_training or args.initial_model
    if not source.is_file():
        parser.error("Source checkpoint/model does not exist")
    if not hasattr(engine, "solve_tactics") or not hasattr(engine, "TacticalSolverOptions"):
        parser.error("Build/install the tactical solver binding before running this command")
    from kingdom_ai.tactical_positions import (
        curriculum_positions, load_casebook, load_position, user_game_two_positions,
    )
    torch.set_num_threads(args.threads)
    options = engine.TacticalSolverOptions(
        max_depth=args.max_depth, max_nodes=args.max_nodes, time_limit_ms=args.time_limit_ms)
    cases = curriculum_positions() + user_game_two_positions()
    if args.casebook:
        cases += load_casebook(args.casebook)
    total_started = perf_counter()
    source_hash = file_digest(source)
    samples, generation = collect_certified_samples(
        cases, options, load_position=load_position,
        max_cases=args.max_cases, generation_seconds=args.generation_seconds)
    train, heldout = split_tactical_samples(samples, heldout_fraction=args.heldout_fraction, seed=args.seed)
    report = {
        "format_version": 1, "source_path": str(source.resolve()),
        "source_sha256_before": source_hash, "source_kind": "training" if args.resume_training else "inference",
        "solver_options": {"max_depth": args.max_depth, "max_nodes": args.max_nodes,
                           "time_limit_ms": args.time_limit_ms},
        "generation": generation, "dry_run": args.dry_run,
        "split": {"train_case_ids": [row.case_id for row in train],
                  "heldout_case_ids": [row.case_id for row in heldout],
                  "train_families": sorted({row.family_id for row in train}),
                  "heldout_families": sorted({row.family_id for row in heldout})},
        "label_policy": "Only certified nonterminal WIN; UNKNOWN and LOSS excluded from imitation",
        "heldout_warning": (None if heldout else "Insufficient independent proof families; no heldout claim"),
        "promoted": False,
    }
    if args.dry_run:
        report["source_sha256_after"] = file_digest(source)
        report["total_seconds"] = perf_counter() - total_started
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "tactical_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"dry_run": True, "certified_samples": len(samples),
                          "train_cases": len(train), "heldout_cases": len(heldout),
                          "report": str(args.output / "tactical_report.json")}, ensure_ascii=False))
        return report
    if not train:
        parser.error("No certified WIN samples within solver budgets; no training or model files were written")
    if args.resume_training:
        trainer = Trainer.load_checkpoint(source, device=args.device)
        if not hasattr(trainer, "train_tactical_batch"):
            parser.error("Install checkpoint-v6 tactical Trainer integration first")
        before_progress = {name: getattr(trainer, name) for name in
                           ("iteration", "self_play_games", "training_steps", "tactical_training_steps", "champion_version")}
        champion_hash = model_digest(trainer.champion)
        trainer.reconfigure(learning_rate=args.learning_rate)
        candidate, training = fine_tune_tactics(
            trainer.model, train, heldout, steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, tactical_fraction=args.tactical_fraction,
            replay=trainer.replay, generator=trainer.generator, seed=args.seed,
            copy_model=False, update_callback=trainer.train_tactical_batch)
        report["progress_before"] = before_progress
        report["progress_after"] = {name: getattr(trainer, name) for name in before_progress}
        report["training_config"] = asdict(trainer.config)
        report["champion_sha256_before"] = champion_hash
        report["champion_sha256_after"] = model_digest(trainer.champion)
        if report["champion_sha256_after"] != champion_hash:
            raise RuntimeError("Tactical fine-tuning unexpectedly changed the champion")
    else:
        trainer = None
        model = load_model(source, device=args.device)
        candidate, training = fine_tune_tactics(
            model, train, heldout, steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, tactical_fraction=args.tactical_fraction, seed=args.seed)
        report["source_model_sha256_unchanged"] = model_digest(model) == training["model_sha256_before"]
    report["training"] = training
    report["source_sha256_after"] = file_digest(source)
    if report["source_sha256_after"] != source_hash:
        raise RuntimeError("Source checkpoint was changed during training")
    # Create new output only after completed updates and all integrity checks.
    args.output.mkdir(parents=True, exist_ok=False)
    save_model(candidate, args.output / "candidate.pt")
    if trainer is not None:
        trainer.save_checkpoint(args.output / "latest.pt")
        report["checkpoint_version"] = torch.load(
            args.output / "latest.pt", map_location="cpu", weights_only=True, mmap=True)["checkpoint_version"]
    report["total_seconds"] = perf_counter() - total_started
    (args.output / "tactical_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"steps": training["steps"], "train_cases": len(train),
                      "heldout_cases": len(heldout), "before": training["before"],
                      "after": training["after"], "training_seconds": training["training_seconds"],
                      "promoted": False, "candidate": str(args.output / "candidate.pt")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
