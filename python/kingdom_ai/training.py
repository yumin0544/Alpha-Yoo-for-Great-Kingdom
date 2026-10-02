"""Small MCTS-teacher bootstrap; neural PUCT/self-play training comes later."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
import my_board_engine as engine

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, encode_state, visit_policy


@dataclass(frozen=True)
class TrainingSample:
    features: torch.Tensor
    legal_mask: torch.Tensor
    policy: torch.Tensor
    value: float
    to_play: engine.Cell


@dataclass(frozen=True)
class GameData:
    samples: list[TrainingSample]
    winner: engine.Cell
    reason: engine.EndReason


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    legal_mask: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class Losses:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


def collect_mcts_game(simulations=64, seed=42) -> GameData:
    game = engine.State()
    searcher = engine.MCTS(engine.MCTSOptions(simulations=simulations, seed=seed))
    positions = []
    while not game.result.finished():
        encoded = encode_state(game)
        search = searcher.search(game)
        policy = visit_policy(search, game)
        # Keep the PRE-MOVE perspective. The engine changes turns on terminal moves.
        positions.append((encoded, policy))
        if search.best_move is None or not game.play(search.best_move).accepted():
            raise RuntimeError("MCTS failed to produce an accepted game move")
        if len(positions) > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Game exceeded its finite move bound")
    winner = game.result.winner
    samples = [TrainingSample(
        encoded.features, encoded.legal_mask, policy,
        1.0 if encoded.to_play == winner else -1.0, encoded.to_play,
    ) for encoded, policy in positions]
    return GameData(samples, winner, game.result.reason)


def make_batch(samples: Sequence[TrainingSample], device="cpu") -> TrainingBatch:
    if not samples:
        raise ValueError("A training batch must contain at least one sample")
    for sample in samples:
        if sample.features.shape != (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError("Unexpected feature shape")
        if sample.legal_mask.shape != (ACTION_SIZE,) or sample.policy.shape != (ACTION_SIZE,):
            raise ValueError("Unexpected action shape")
        if sample.legal_mask.dtype != torch.bool:
            raise ValueError("Legal action masks must be bool tensors")
    return TrainingBatch(
        torch.stack([sample.features for sample in samples]).to(device=device, dtype=torch.float32),
        torch.stack([sample.legal_mask for sample in samples]).to(device=device),
        torch.stack([sample.policy for sample in samples]).to(device=device, dtype=torch.float32),
        torch.tensor([sample.value for sample in samples], device=device, dtype=torch.float32),
    )


def policy_value_loss(logits, values, target_policy, target_value, legal_mask) -> Losses:
    if logits.ndim != 2 or logits.shape[1] != ACTION_SIZE or logits.shape[0] < 1:
        raise ValueError("Policy logits must have shape [N, 82] with N > 0")
    if (target_policy.shape != logits.shape or legal_mask.shape != logits.shape
            or legal_mask.dtype != torch.bool):
        raise ValueError("Policy targets and bool masks must match logits")
    if values.shape != (logits.shape[0],) or target_value.shape != values.shape:
        raise ValueError("Values and value targets must have shape [N]")
    tensors = (values, target_policy, target_value, legal_mask)
    if any(tensor.device != logits.device for tensor in tensors):
        raise ValueError("Loss tensors must be on the same device")
    if not logits.is_floating_point() or any(
            not tensor.is_floating_point() for tensor in tensors[:-1]):
        raise ValueError("Logits, values and targets must be floating-point tensors")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (logits, *tensors[:-1])):
        raise ValueError("Loss inputs must be finite")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("Terminal/all-illegal rows cannot be policy training samples")
    if bool((target_policy < 0).any()) or bool((target_policy[~legal_mask] != 0).any()):
        raise ValueError("Policy targets must be non-negative and assign zero to illegal moves")
    if not torch.allclose(target_policy.sum(dim=1), torch.ones_like(target_value), atol=1e-6, rtol=0):
        raise ValueError("Each policy target must sum to one")
    if bool((target_value.abs() > 1).any()):
        raise ValueError("Value targets must be between -1 and 1")
    masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
    log_probs = F.log_softmax(masked_logits, dim=1)
    # Avoid multiplying zero targets by -inf at forbidden actions.
    log_probs = torch.where(legal_mask, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_value)
    if not bool(torch.isfinite(policy_loss + value_loss)):
        raise ValueError("Policy/value loss overflowed; check model outputs and targets")
    return Losses(policy_loss + value_loss, policy_loss, value_loss)


def train_step(model, optimizer, batch: TrainingBatch) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(batch.features)
    losses = policy_value_loss(logits, values, batch.policy, batch.value, batch.legal_mask)
    losses.total.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
           for parameter in model.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Non-finite model gradients")
    optimizer.step()
    return {
        "loss": losses.total.detach().item(),
        "policy_loss": losses.policy.detach().item(),
        "value_loss": losses.value.detach().item(),
    }
