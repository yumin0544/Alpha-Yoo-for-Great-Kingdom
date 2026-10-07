"""Completed MCTS/PUCT games and policy/value training batches."""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
import my_board_engine as engine

from .encoding import (
    ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, encode_state, move_to_action, visit_policy,
)


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
    action_history: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    legal_mask: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class ProofTrainingBatch(TrainingBatch):
    """Proof rows whose policy is enabled only for certified winning moves.

    LOSS rows retain a normalized legal placeholder policy for storage/schema
    validation. That placeholder is never an imitation target.
    """
    policy_enabled: torch.Tensor


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


def _validate_temperature_schedule(temperature_moves, final_temperature):
    if temperature_moves is not None and (type(temperature_moves) is not int
                                          or temperature_moves < 0):
        raise ValueError("temperature_moves must be None or a non-negative integer")
    if not isinstance(final_temperature, (int, float)) or isinstance(final_temperature, bool):
        raise TypeError("Final temperature must be a real number")
    if not math.isfinite(final_temperature) or final_temperature < 0:
        raise ValueError("Final temperature must be finite and non-negative")


def collect_puct_game(model, options=None, temperature=1.0, seed=42, *,
                      temperature_moves=None, final_temperature=0.0,
                      record_history=False) -> GameData:
    """Play both sides with neural PUCT and label PRE-MOVE player outcomes.

    With ``temperature_moves=None`` the original temperature applies throughout
    the game. Otherwise only the first ``temperature_moves`` plies use it, and
    subsequent moves use ``final_temperature``. Policy targets remain the raw
    root visit proportions at either temperature.

    ``record_history=True`` retains the actual selected actions, not policy
    argmaxes, so that the complete game can be replayed from the default initial
    state without inferring permanent house ownership from a board snapshot.
    """
    from .puct import PUCT, PUCTOptions, sample_visits
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("Temperature must be a real number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative")
    _validate_temperature_schedule(temperature_moves, final_temperature)
    if type(record_history) is not bool:
        raise TypeError("record_history must be bool")
    if options is None:
        options = PUCTOptions(seed=seed, dirichlet_epsilon=0.25)
    searcher = PUCT(model, options)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    game = engine.State()
    positions = []
    actions = [] if record_history else None
    while not game.result.finished():
        encoded = encode_state(game)
        search = searcher.search(game)
        policy = visit_policy(search, game)
        move_temperature = (temperature if temperature_moves is None
                            or len(positions) < temperature_moves else final_temperature)
        move = sample_visits(search, temperature=move_temperature, generator=generator)
        positions.append((encoded, policy))
        if not game.play(move).accepted():
            raise RuntimeError("PUCT failed to produce an accepted game move")
        if actions is not None:
            actions.append(move_to_action(move))
        if len(positions) > 2 * engine.CELL_COUNT + 2:
            raise RuntimeError("Game exceeded its finite move bound")
    winner = game.result.winner
    samples = [TrainingSample(
        encoded.features, encoded.legal_mask, policy,
        1.0 if encoded.to_play == winner else -1.0, encoded.to_play,
    ) for encoded, policy in positions]
    return GameData(samples, winner, game.result.reason,
                    None if actions is None else tuple(actions))


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


def _policy_value_rows(logits, values, target_policy, target_value, legal_mask):
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
    policy_rows = -(target_policy * log_probs).sum(dim=1)
    value_rows = (values - target_value).square()
    if not bool(torch.isfinite(policy_rows + value_rows).all()):
        raise ValueError("Policy/value loss overflowed; check model outputs and targets")
    return policy_rows, value_rows


def policy_value_loss(logits, values, target_policy, target_value, legal_mask) -> Losses:
    policy_rows, value_rows = _policy_value_rows(
        logits, values, target_policy, target_value, legal_mask)
    policy_loss, value_loss = policy_rows.mean(), value_rows.mean()
    return Losses(policy_loss + value_loss, policy_loss, value_loss)


def train_step(model, optimizer, batch: TrainingBatch) -> dict[str, float]:
    if isinstance(batch, ProofTrainingBatch):
        raise ValueError("Proof batches require train_mixed_step so LOSS policy placeholders are masked")
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


def augment_proof_batch(batch: ProofTrainingBatch, *, generator) -> ProofTrainingBatch:
    """Apply D4 to observations/actions, keeping the row-level policy mask."""
    from .augmentation import augment_batch
    if not isinstance(batch, ProofTrainingBatch):
        raise TypeError("Expected a ProofTrainingBatch")
    augmented = augment_batch(batch, generator=generator)
    return ProofTrainingBatch(augmented.features, augmented.legal_mask,
                              augmented.policy, augmented.value,
                              batch.policy_enabled.clone())


def mixed_policy_value_loss(logits, values, target_policy, target_value, legal_mask,
                            *, normal_rows, policy_enabled):
    """Separated losses without changing the total teacher row budget.

    Every row contributes value loss; only ordinary or WIN rows contribute
    policy loss. Both overall terms use ALL rows as their denominator. In
    particular, averaging policy over WIN rows alone would accidentally amplify
    their weight as LOSS rows are added to the fixed teacher fraction.
    Category metrics are unweighted means, except teacher_policy_loss, whose
    denominator is all teacher rows rather than just policy-enabled rows.
    """
    count = logits.shape[0] if logits.ndim else 0
    if type(normal_rows) is not int or not 0 <= normal_rows < count:
        raise ValueError("Mixed loss requires at least one teacher row")
    if (not isinstance(policy_enabled, torch.Tensor)
            or policy_enabled.dtype != torch.bool or policy_enabled.shape != (count,)
            or policy_enabled.device != logits.device):
        raise ValueError("Policy-enabled mask must be bool [N] on the logits device")
    if not bool(policy_enabled[:normal_rows].all()):
        raise ValueError("All ordinary policy targets must be enabled")
    teacher_enabled = policy_enabled[normal_rows:]
    teacher_values = target_value[normal_rows:]
    if (teacher_values.shape != teacher_enabled.shape
            or bool((teacher_values != torch.where(teacher_enabled, 1.0, -1.0)).any())):
        raise ValueError("Teacher WIN/+1 policies must be enabled; LOSS/-1 policies disabled")
    policy_rows, value_rows = _policy_value_rows(
        logits, values, target_policy, target_value, legal_mask)
    enabled_policy = policy_rows * policy_enabled.to(policy_rows.dtype)
    policy_loss, value_loss = enabled_policy.mean(), value_rows.mean()
    zero = values.sum() * 0.0
    def mean_or_zero(rows):
        return rows.mean() if rows.numel() else zero
    teacher_value_rows = value_rows[normal_rows:]
    return {
        "loss": policy_loss + value_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "normal_policy_loss": mean_or_zero(policy_rows[:normal_rows]),
        "normal_value_loss": mean_or_zero(value_rows[:normal_rows]),
        "teacher_policy_loss": enabled_policy[normal_rows:].mean(),
        "teacher_value_loss": teacher_value_rows.mean(),
        "teacher_win_value_loss": mean_or_zero(teacher_value_rows[teacher_enabled]),
        "teacher_loss_value_loss": mean_or_zero(teacher_value_rows[~teacher_enabled]),
    }


def train_mixed_step(model, optimizer, normal_batch: TrainingBatch | None,
                     proof_batch: ProofTrainingBatch) -> dict[str, float]:
    """One Adam update for normal replay plus masked certified WIN/LOSS rows."""
    if not isinstance(proof_batch, ProofTrainingBatch):
        raise TypeError("Teacher data requires a ProofTrainingBatch")
    if (normal_batch is not None and (not isinstance(normal_batch, TrainingBatch)
                                     or isinstance(normal_batch, ProofTrainingBatch))):
        raise TypeError("Ordinary data requires a TrainingBatch")
    normal_rows = 0 if normal_batch is None else normal_batch.features.shape[0]
    if normal_batch is None:
        batch = proof_batch
        enabled = proof_batch.policy_enabled
    else:
        batch = TrainingBatch(*(torch.cat((getattr(normal_batch, field),
                                           getattr(proof_batch, field)), dim=0)
                                for field in ("features", "legal_mask", "policy", "value")))
        enabled = torch.cat((torch.ones(normal_rows, dtype=torch.bool,
                                        device=batch.features.device),
                             proof_batch.policy_enabled))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(batch.features)
    losses = mixed_policy_value_loss(logits, values, batch.policy, batch.value,
                                    batch.legal_mask, normal_rows=normal_rows,
                                    policy_enabled=enabled)
    losses["loss"].backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
           for parameter in model.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Non-finite model gradients")
    optimizer.step()
    teacher_rows = batch.features.shape[0] - normal_rows
    teacher_win_rows = int(proof_batch.policy_enabled.sum().item())
    metrics = {key: value.detach().item() for key, value in losses.items()}
    metrics.update({"normal_rows": normal_rows, "teacher_rows": teacher_rows,
                    "teacher_policy_rows": teacher_win_rows,
                    "teacher_win_rows": teacher_win_rows,
                    "teacher_loss_rows": teacher_rows - teacher_win_rows,
                    "actual_tactical_fraction": teacher_rows / batch.features.shape[0]})
    # Colour-specific value errors expose a one-sided regression without
    # changing loss weights. Actor colour is encoded as a constant plane.
    black = batch.features[:, 5, 0, 0] == 1
    squared_error = (values.detach() - batch.value).square()
    for prefix, subset in (("normal", slice(0, normal_rows)),
                           ("teacher", slice(normal_rows, None))):
        for color, selected in (("black", black[subset]), ("white", ~black[subset])):
            errors = squared_error[subset][selected]
            metrics[f"{prefix}_{color}_rows"] = errors.numel()
            metrics[f"{prefix}_{color}_value_loss"] = errors.mean().item() if errors.numel() else 0.0
    return metrics
