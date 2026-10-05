"""Completed CUDA PUCT self-play games in the CPU replay-buffer schema.

Rules, search, visit sampling and position histories stay on CUDA until an
entire chunk finishes. Only completion/error scalars synchronize inside the
game loop; position tensors are copied to the CPU in bulk afterwards.
"""

from __future__ import annotations

import math

import torch
import my_board_engine as engine

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS
from .gpu_puct import GpuPUCT, GpuPUCTOptions
from .gpu_rules import (
    GPU_ACTOR, GPU_CAPTURE, GPU_REASON, GPU_SUICIDE, GPU_TWO_PASSES,
    GPU_WINNER, GpuStateBatch,
)
from .training import GameData, TrainingSample


_MAX_PLIES = 2 * engine.CELL_COUNT + 2
_SAMPLING_SEED_SALT = 0x9E3779B97F4A7C15


def _visit_sampling_weights(visits, temperature):
    """Use relative log counts to avoid overflow at tiny temperatures."""
    positive = visits > 0
    logs = visits.to(torch.float64).clamp_min(1).log()
    logs -= logs.amax(dim=1, keepdim=True)
    # With int32 counts, every smaller count already underflows to zero below
    # the normal FP64 range. Clamping subnormal temperatures also prevents a
    # device implementation from turning 0 / temperature into 0 * infinity.
    temperature = max(temperature, torch.finfo(torch.float64).tiny)
    return (logs / temperature).exp().masked_fill(~positive, 0.0)


def _collect_chunk(searcher, state, temperature, generator):
    batch = len(state)
    device = state.device
    features_history = torch.empty(
        (_MAX_PLIES, batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=torch.float32, device=device,
    )
    masks_history = torch.empty(
        (_MAX_PLIES, batch, ACTION_SIZE), dtype=torch.bool, device=device,
    )
    policy_history = torch.empty_like(masks_history, dtype=torch.float32)
    actors_history = torch.empty((_MAX_PLIES, batch), dtype=torch.int32, device=device)
    lengths = torch.zeros(batch, dtype=torch.int32, device=device)
    invalid_move = torch.zeros((), dtype=torch.bool, device=device)
    plies = 0

    for step in range(_MAX_PLIES):
        alive = state.states[:, GPU_REASON] == 0
        if not bool(alive.any().item()):
            break
        features, masks = state.encode()
        # Store the actor before play(), including on a terminal capture/suicide.
        features_history[step].copy_(features)
        masks_history[step].copy_(masks)
        actors_history[step].copy_(state.states[:, GPU_ACTOR])
        result = searcher.search(state)
        # Policy targets are raw visit proportions, independent of move temperature.
        policy_history[step].copy_(result.policy)
        if temperature == 0:
            actions = result.actions
        else:
            weights = _visit_sampling_weights(result.visits, temperature)
            # Terminal lanes have no visits. A dummy pass lets multinomial run
            # with fixed-size batches; their eventual action remains -1.
            weights[:, -1] += (~alive).to(weights.dtype)
            sampled = torch.multinomial(weights, 1, generator=generator).flatten()
            actions = torch.where(alive, sampled, result.actions)
        accepted = state.play(actions)
        invalid_move |= (alive & ~accepted).any()
        lengths += alive.to(lengths.dtype)
        plies = step + 1

    if bool(invalid_move.item()):
        raise RuntimeError("GPU PUCT failed to produce an accepted game move")
    if bool((state.states[:, GPU_REASON] == 0).any().item()):
        raise RuntimeError("Game exceeded its finite move bound")

    # One transfer per dense history tensor, never a per-position board snapshot.
    features_cpu = features_history[:plies].cpu()
    masks_cpu = masks_history[:plies].cpu()
    policy_cpu = policy_history[:plies].cpu()
    actors_cpu = actors_history[:plies].cpu()
    metadata = torch.stack((lengths, state.states[:, GPU_WINNER],
                            state.states[:, GPU_REASON]), dim=1).cpu().tolist()
    players = {1: engine.Cell.Black, 2: engine.Cell.White}
    reasons = {
        GPU_CAPTURE: engine.EndReason.Capture,
        GPU_SUICIDE: engine.EndReason.Suicide,
        GPU_TWO_PASSES: engine.EndReason.TwoPasses,
    }
    games = []
    for lane, (length, winner_code, reason_code) in enumerate(metadata):
        winner = players[winner_code]
        samples = []
        for step in range(length):
            actor = players[int(actors_cpu[step, lane])]
            samples.append(TrainingSample(
                features_cpu[step, lane], masks_cpu[step, lane], policy_cpu[step, lane],
                1.0 if actor == winner else -1.0, actor,
            ))
        games.append(GameData(samples, winner, reasons[reason_code]))
    return games


def collect_gpu_puct_games(model, games, *, options=None, temperature=1.0,
                           batch_size=128, seed=42, device="cuda") -> list[GameData]:
    """Generate ordered complete games with CUDA PUCT and CPU training samples.

    ``options.seed`` controls root noise, and ``seed`` controls visit sampling;
    supply the same seed to both to select a complete reproducible call. The
    default options do this automatically and enable root noise with epsilon
    0.25. Each call owns new private generators and preserves global RNG state
    and the source model. Games run in fixed chunks without filling finished
    lanes. Changing the chunk width changes the random stream's lane grouping.
    """
    if type(games) is not int or games < 0:
        raise ValueError("games must be a non-negative integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("Temperature must be a real number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative")
    if options is None:
        options = GpuPUCTOptions(seed=seed)
    if not isinstance(options, GpuPUCTOptions):
        raise TypeError("options must be GpuPUCTOptions")
    if games == 0:
        return []

    searcher = GpuPUCT(model, options=options, device=device)
    # Give root noise and move sampling separate reproducible random streams.
    generator = torch.Generator(device=searcher.device).manual_seed(seed ^ _SAMPLING_SEED_SALT)
    completed = []
    # no_grad keeps returned CPU samples ordinary mutable tensors. Search owns
    # its stricter inference_mode context internally.
    with torch.cuda.device(searcher.device), torch.no_grad():
        for start in range(0, games, batch_size):
            state = GpuStateBatch.initial(min(batch_size, games - start), device=searcher.device)
            completed.extend(_collect_chunk(searcher, state, temperature, generator))
    return completed
