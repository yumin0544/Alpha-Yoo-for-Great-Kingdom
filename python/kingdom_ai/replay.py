"""Bounded CPU replay storage with reproducible minibatch sampling."""

from collections.abc import Iterable, Mapping
from numbers import Real

import torch
import my_board_engine as engine

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS, PASS_ACTION
from .training import TrainingBatch, TrainingSample


_STATE_VERSION = 1
_BULK_ROWS = 4096
_STATE_KEYS = {
    "version", "capacity", "size", "next_index", "features", "legal_mask",
    "policy", "value", "to_play",
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cpu_tensor(value, shape, dtype, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    return value.detach().to(device="cpu")


def _validate_rows(features, legal_mask, policy, value, to_play):
    """Validate observation schema 1 and completed-game targets on CPU."""
    if features.shape[0] == 0:
        return
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(policy).all()):
        raise ValueError("Replay features and policies must be finite")
    if bool(((features < 0) | (features > 1)).any()):
        raise ValueError("Replay features must lie between zero and one")
    for plane in (0, 1, 2, 3, 4, 5, 9):
        channel = features[:, plane]
        if bool(((channel != 0) & (channel != 1)).any()):
            raise ValueError("Binary observation planes must contain only zero or one")
    for plane in (5, 6, 7, 8):
        channel = features[:, plane].flatten(start_dim=1)
        if bool((channel != channel[:, :1]).any()):
            raise ValueError("Scalar observation planes must be constant")
    if bool(((to_play != 1) & (to_play != 2)).any()):
        raise ValueError("Replay actors must be Black or White")
    if bool((features[:, 5, 0, 0] != (to_play == 1).to(torch.float32)).any()):
        raise ValueError("Observation player perspective does not match its actor")
    passes = features[:, 8, 0, 0]
    if bool(((passes != 0) & (passes != 0.5)).any()):
        raise ValueError("Replay observations must be taken before game termination")
    if not bool(legal_mask[:, PASS_ACTION].all()):
        raise ValueError("Nonterminal replay observations must allow passing")
    if not torch.equal(features[:, 9].flatten(start_dim=1).bool(),
                       legal_mask[:, :PASS_ACTION]):
        raise ValueError("Observation legal placements do not match the action mask")
    if bool((features[:, :3].sum(dim=1) > 1).any()):
        raise ValueError("Stone observation planes cannot overlap")
    if bool((features[:, 2].sum(dim=(1, 2)) > 1).any()):
        raise ValueError("An observation can contain at most one neutral stone")
    if bool((features[:, 3] + features[:, 4] > 1).any()):
        raise ValueError("Opposing territory observation planes cannot overlap")
    if bool((policy < 0).any()) or bool((policy[~legal_mask] != 0).any()):
        raise ValueError("Replay policies must assign non-negative mass to legal actions only")
    if not torch.allclose(policy.sum(dim=1), torch.ones_like(value), rtol=0, atol=1e-6):
        raise ValueError("Each replay policy must sum to one")
    if not bool(torch.isfinite(value).all()) or bool(((value != -1) & (value != 1)).any()):
        raise ValueError("Completed-game replay values must be -1 or +1")


def _prepare_sample_components(sample):
    if not isinstance(sample, TrainingSample):
        raise TypeError("Replay entries must be TrainingSample objects")
    features = _cpu_tensor(sample.features, (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                           torch.float32, "features")
    legal_mask = _cpu_tensor(sample.legal_mask, (ACTION_SIZE,), torch.bool, "legal_mask")
    policy = _cpu_tensor(sample.policy, (ACTION_SIZE,), torch.float32, "policy")
    if isinstance(sample.value, bool) or not isinstance(sample.value, Real):
        raise ValueError("Sample value must be -1 or +1")
    if sample.value not in (-1, 1):
        raise ValueError("Sample value must be -1 or +1")
    if (not isinstance(sample.to_play, engine.Cell)
            or sample.to_play not in (engine.Cell.Black, engine.Cell.White)):
        raise ValueError("Sample actor must be Black or White")
    actor = 1 if sample.to_play == engine.Cell.Black else 2
    return features, legal_mask, policy, float(sample.value), actor


def _stack_components(rows):
    """Materialize a bounded group of sample views as contiguous CPU rows."""
    return (
        torch.stack([row[0] for row in rows]),
        torch.stack([row[1] for row in rows]),
        torch.stack([row[2] for row in rows]),
        torch.tensor([row[3] for row in rows], dtype=torch.float32, device="cpu"),
        torch.tensor([row[4] for row in rows], dtype=torch.int8, device="cpu"),
    )


class ReplayBuffer:
    """FIFO ring measured in positions, sampled uniformly with replacement.

    Storage occupies ``capacity * 3655`` tensor bytes for observation schema 1.
    Samples, batches, and serialized state never share writable tensor storage.
    A caller-owned CPU generator controls sampling without using global RNG state.
    """

    def __init__(self, capacity: int):
        self._capacity = _positive_integer(capacity, "Replay capacity")
        self._size = 0
        self._next_index = 0
        self._features = torch.empty((capacity, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                                     dtype=torch.float32, device="cpu")
        self._legal_mask = torch.empty((capacity, ACTION_SIZE), dtype=torch.bool, device="cpu")
        self._policy = torch.empty((capacity, ACTION_SIZE), dtype=torch.float32, device="cpu")
        self._value = torch.empty(capacity, dtype=torch.float32, device="cpu")
        # 1 = Black, 2 = White. Keep actor metadata in checkpoints, not model batches.
        self._to_play = torch.empty(capacity, dtype=torch.int8, device="cpu")

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        """Validate the complete input before replacing any existing positions."""
        incoming = list(samples)
        if not incoming:
            return
        # Keep validation atomic, but validate tensor contents in large vectorized
        # groups instead of launching dozens of tiny operations per position.
        prepared = [_prepare_sample_components(sample) for sample in incoming]
        for start in range(0, len(prepared), _BULK_ROWS):
            _validate_rows(*_stack_components(prepared[start:start + _BULK_ROWS]))

        # Skip writes that would be immediately evicted, but preserve ring slots.
        skipped = max(0, len(incoming) - self._capacity)
        index = (self._next_index + skipped) % self._capacity
        offset = skipped
        while offset < len(prepared):
            count = min(_BULK_ROWS, len(prepared) - offset, self._capacity - index)
            features, legal_mask, policy, value, actor = _stack_components(
                prepared[offset:offset + count])
            target = slice(index, index + count)
            self._features[target].copy_(features)
            self._legal_mask[target].copy_(legal_mask)
            self._policy[target].copy_(policy)
            self._value[target].copy_(value)
            self._to_play[target].copy_(actor)
            offset += count
            index = (index + count) % self._capacity
        self._next_index = index
        self._size = min(self._capacity, self._size + len(incoming))

    def sample(self, batch_size: int, *, generator: torch.Generator,
               device="cpu") -> TrainingBatch:
        """Return an independent random batch; batch_size may exceed buffer size."""
        _positive_integer(batch_size, "Batch size")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Replay sampling requires a caller-owned CPU torch.Generator")
        if self._size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        indices = torch.randint(self._size, (batch_size,), generator=generator, device="cpu")
        return TrainingBatch(
            self._features[indices].to(device=device),
            self._legal_mask[indices].to(device=device),
            self._policy[indices].to(device=device),
            self._value[indices].to(device=device),
        )

    def state_dict(self) -> dict:
        """Copy only occupied physical slots, preserving exact seeded sampling."""
        return {
            "version": _STATE_VERSION,
            "capacity": self._capacity,
            "size": self._size,
            "next_index": self._next_index,
            "features": self._features[:self._size].clone(),
            "legal_mask": self._legal_mask[:self._size].clone(),
            "policy": self._policy[:self._size].clone(),
            "value": self._value[:self._size].clone(),
            "to_play": self._to_play[:self._size].clone(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping) -> "ReplayBuffer":
        """Restore a validated tensor-only checkpoint with no shared storage."""
        if not isinstance(payload, Mapping) or set(payload) != _STATE_KEYS:
            raise ValueError("Unexpected replay checkpoint fields")
        version = payload["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != _STATE_VERSION:
            raise ValueError("Unsupported replay checkpoint version")
        capacity = _positive_integer(payload["capacity"], "Replay capacity")
        size, next_index = payload["size"], payload["next_index"]
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= capacity:
            raise ValueError("Replay checkpoint size is outside capacity")
        if (isinstance(next_index, bool) or not isinstance(next_index, int)
                or not 0 <= next_index < capacity
                or (size < capacity and next_index != size)):
            raise ValueError("Replay checkpoint ring index is inconsistent")
        shapes = {
            "features": ((size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), torch.float32),
            "legal_mask": ((size, ACTION_SIZE), torch.bool),
            "policy": ((size, ACTION_SIZE), torch.float32),
            "value": ((size,), torch.float32),
            "to_play": ((size,), torch.int8),
        }
        tensors = {name: _cpu_tensor(payload[name], shape, dtype, name)
                   for name, (shape, dtype) in shapes.items()}
        # Bound validation temporaries even when a checkpoint holds many positions.
        for start in range(0, size, 1024):
            end = min(size, start + 1024)
            _validate_rows(*(tensors[name][start:end] for name in shapes))
        restored = cls(capacity)
        for name, tensor in tensors.items():
            getattr(restored, "_" + name)[:size].copy_(tensor)
        restored._size = size
        restored._next_index = next_index
        return restored
