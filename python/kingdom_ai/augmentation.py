"""Independent square-board symmetries for sampled policy/value batches."""

from __future__ import annotations

import torch

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, PASS_ACTION
from .training import TrainingBatch


# Each row maps an output square to its source square. The last four rows
# reflect columns before rotating. The same permutation must transform every
# observation plane, placement mask and placement probability together.
_SQUARES = torch.arange(PASS_ACTION, dtype=torch.int64, device="cpu").reshape(BOARD_SIZE, BOARD_SIZE)
_PERMUTATIONS = torch.stack([
    torch.rot90(_SQUARES if symmetry < 4 else _SQUARES.flip(-1), symmetry % 4,
                dims=(-2, -1)).reshape(-1)
    for symmetry in range(8)
])


def augment_batch(batch: TrainingBatch, *, generator: torch.Generator) -> TrainingBatch:
    """Draw one of the eight D4 symmetries independently for each position.

    Only the caller's CPU generator is consumed, so Trainer checkpoints can
    reproduce the next augmentation exactly. Tensor transformations run on the
    batch device without multiplying replay storage or mini-batch size. Pass,
    current-player value and scalar observation planes retain their meaning.
    The returned batch owns storage independent of the input, even for identity
    transformations.
    """
    if not isinstance(batch, TrainingBatch):
        raise TypeError("batch must be a TrainingBatch")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Augmentation requires a caller-owned CPU torch.Generator")
    if (batch.features.ndim != 4 or tuple(batch.features.shape[1:])
            != (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE) or batch.features.shape[0] < 1):
        raise ValueError("Batch features must have shape [N, 10, 9, 9] with N > 0")
    size = batch.features.shape[0]
    if (batch.legal_mask.shape != (size, ACTION_SIZE)
            or batch.policy.shape != (size, ACTION_SIZE) or batch.value.shape != (size,)):
        raise ValueError("Batch policy, mask and values must match the observation count")
    if batch.legal_mask.dtype != torch.bool:
        raise ValueError("Batch legal action masks must have bool dtype")
    if any(tensor.device != batch.features.device
           for tensor in (batch.legal_mask, batch.policy, batch.value)):
        raise ValueError("Batch tensors must share a device")
    symmetries = torch.randint(8, (size,), generator=generator, device="cpu")
    indices = _PERMUTATIONS[symmetries].to(device=batch.features.device)
    features = batch.features.flatten(2).gather(
        2, indices.unsqueeze(1).expand(-1, INPUT_CHANNELS, -1),
    ).reshape_as(batch.features)
    mask = torch.cat((batch.legal_mask[:, :PASS_ACTION].gather(1, indices),
                      batch.legal_mask[:, PASS_ACTION:]), dim=1)
    policy = torch.cat((batch.policy[:, :PASS_ACTION].gather(1, indices),
                        batch.policy[:, PASS_ACTION:]), dim=1)
    return TrainingBatch(features, mask, policy, batch.value.clone())
