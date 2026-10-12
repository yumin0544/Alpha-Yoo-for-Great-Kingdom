"""Bounded certified action-loss replay, never an UNKNOWN value/safety target.

Certificates are solver audit records, not a replay of the full minimax tree.
The learned action loss concerns the actual Kingdom winner, including scoring
and suicide as well as capture; a Capture PV alone is not a capture-only proof.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json

import torch

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS
from .training import CaptureDefenseBatch, TrainingBatch


@dataclass(frozen=True)
class CertifiedDefenseSample:
    case_id: str
    family_id: str
    features: torch.Tensor
    legal_mask: torch.Tensor
    losing_mask: torch.Tensor
    proof: dict


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_capture_defense_batch(batch):
    """Validate tensor schema and nontrivial legal counterexample masks."""
    if not isinstance(batch, CaptureDefenseBatch):
        raise TypeError("Expected CaptureDefenseBatch")
    features = batch.features
    if (not isinstance(features, torch.Tensor) or features.layout != torch.strided
            or features.ndim != 4 or features.shape[0] < 1
            or tuple(features.shape[1:]) != (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            or features.dtype != torch.float32):
        raise ValueError("Defense features must be float32 [N, 10, 9, 9] with N > 0")
    for name in ("legal_mask", "losing_mask"):
        mask = getattr(batch, name)
        if (not isinstance(mask, torch.Tensor) or mask.layout != torch.strided
                or mask.shape != (features.shape[0], ACTION_SIZE) or mask.dtype != torch.bool
                or mask.device != features.device):
            raise ValueError(f"Defense {name} must be bool [N, 82] on the features device")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Defense features must be finite")
    if bool((batch.losing_mask & ~batch.legal_mask).any()):
        raise ValueError("Certified losing actions must be legal")
    if not bool(batch.losing_mask.any(dim=1).all()):
        raise ValueError("Each defense row must contain a certified losing action")
    if not bool((batch.legal_mask & ~batch.losing_mask).any(dim=1).all()):
        raise ValueError("All-legal-actions-losing positions cannot train avoidance")


def _validate_observations(features, legal_mask):
    # Reuse schema validation only; these dummy targets are never stored, used
    # for training, or asserted to be a value/policy certificate for the parent.
    from .replay import _validate_rows
    count = features.shape[0]
    placeholder = legal_mask.to(torch.float32)
    placeholder /= placeholder.sum(dim=1, keepdim=True)
    actor = torch.where(features[:, 5, 0, 0] == 1, 1, 2).to(torch.int8)
    _validate_rows(features, legal_mask, placeholder, torch.ones(count), actor)


def _metadata(row):
    if not isinstance(row, CertifiedDefenseSample):
        raise TypeError("Defense replay requires CertifiedDefenseSample entries")
    for name in ("case_id", "family_id"):
        if type(getattr(row, name)) is not str or not getattr(row, name):
            raise ValueError("Defense identifiers must be nonempty strings")
    proof = row.proof
    if not isinstance(proof, dict) or proof.get("outcome") != "ACTION_LOSS":
        raise ValueError("Defense data requires an ACTION_LOSS certificate, not UNKNOWN")
    _positive_integer(proof.get("proof_depth"), "Defense proof depth")
    actions = proof.get("losing_actions")
    if (not isinstance(actions, list) or not actions
            or any(type(action) is not int or not 0 <= action < ACTION_SIZE for action in actions)
            or len(actions) != len(set(actions))
            or set(actions) != set(torch.nonzero(row.losing_mask).flatten().tolist())):
        raise ValueError("Defense certificate actions must match losing_mask support")
    if proof.get("successor_outcome") != "WIN":
        raise ValueError("Only a proved opponent WIN certifies a successor counterexample")
    if "target_kind" in proof and proof["target_kind"] != "negative_policy_only":
        raise ValueError("Defense certificates must be negative-policy-only")
    try:
        json.dumps(proof, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Defense certificates must contain finite JSON data") from error
    return {"case_id": row.case_id, "family_id": row.family_id, "proof": deepcopy(proof)}


def augment_capture_defense_batch(batch, *, generator):
    """Apply exactly one shared D4 permutation to features and both action masks."""
    from .augmentation import augment_batch
    validate_capture_defense_batch(batch)
    # augment_batch performs permutations only, so a bool counterexample mask
    # can use its policy slot without pretending to be an imitation target.
    transformed = augment_batch(TrainingBatch(batch.features, batch.legal_mask,
        batch.losing_mask.to(torch.float32), torch.zeros(batch.features.shape[0],
                                                        device=batch.features.device)),
        generator=generator)
    return CaptureDefenseBatch(transformed.features, transformed.legal_mask,
                               transformed.policy.bool())


class CaptureDefenseReplay:
    """CPU tensor FIFO ring with physical-slot metadata and private sampling RNG."""

    def __init__(self, capacity):
        self._capacity = _positive_integer(capacity, "Defense replay capacity")
        self._size = 0
        self._next_index = 0
        self._features = torch.empty((capacity, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                                     dtype=torch.float32, device="cpu")
        self._legal_mask = torch.empty((capacity, ACTION_SIZE), dtype=torch.bool, device="cpu")
        self._losing_mask = torch.empty_like(self._legal_mask)
        self._certificates = [None] * capacity

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return self._size

    def extend(self, rows):
        """Validate every row atomically before replacing any existing ring slot."""
        incoming = list(rows)
        if not incoming:
            return
        metadata = []
        for row in incoming:
            if not isinstance(row, CertifiedDefenseSample):
                raise TypeError("Defense replay requires CertifiedDefenseSample entries")
            for name, shape, dtype in (("features", (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), torch.float32),
                                       ("legal_mask", (ACTION_SIZE,), torch.bool),
                                       ("losing_mask", (ACTION_SIZE,), torch.bool)):
                tensor = getattr(row, name)
                if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                        or tensor.shape != shape or tensor.dtype != dtype or tensor.device.type != "cpu"):
                    raise ValueError(f"Defense sample {name} must have CPU shape {shape} and dtype {dtype}")
            metadata.append(_metadata(row))
        for offset in range(0, len(incoming), 1024):
            chunk = incoming[offset:offset + 1024]
            batch = CaptureDefenseBatch(*(torch.stack([getattr(row, name) for row in chunk])
                                          for name in ("features", "legal_mask", "losing_mask")))
            validate_capture_defense_batch(batch)
            _validate_observations(batch.features, batch.legal_mask)
        skipped = max(0, len(incoming) - self.capacity)
        for index in range(skipped, len(incoming)):
            slot = (self._next_index + index) % self.capacity
            for name in ("features", "legal_mask", "losing_mask"):
                getattr(self, "_" + name)[slot].copy_(getattr(incoming[index], name).detach())
            self._certificates[slot] = metadata[index]
        self._next_index = (self._next_index + len(incoming)) % self.capacity
        self._size = min(self.capacity, self._size + len(incoming))

    def sample(self, batch_size, *, generator, device="cpu"):
        _positive_integer(batch_size, "Defense batch size")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Defense sampling requires a caller-owned CPU torch.Generator")
        if not self._size:
            raise ValueError("Cannot sample an empty defense replay")
        indices = torch.randint(self._size, (batch_size,), generator=generator, device="cpu")
        return CaptureDefenseBatch(*(getattr(self, "_" + name)[indices].to(device=device)
                                     for name in ("features", "legal_mask", "losing_mask")))

    def state_dict(self):
        return {"version": 1, "capacity": self.capacity, "size": self._size,
                "next_index": self._next_index,
                **{name: getattr(self, "_" + name)[:self._size].clone()
                   for name in ("features", "legal_mask", "losing_mask")},
                "certificates": deepcopy(self._certificates[:self._size])}

    @classmethod
    def from_state_dict(cls, payload):
        expected = {"version", "capacity", "size", "next_index", "features", "legal_mask",
                    "losing_mask", "certificates"}
        if (not isinstance(payload, Mapping) or set(payload) != expected
                or type(payload["version"]) is not int or payload["version"] != 1):
            raise ValueError("Invalid defense replay checkpoint fields/version")
        capacity = _positive_integer(payload["capacity"], "Defense replay capacity")
        size, next_index = payload["size"], payload["next_index"]
        if type(size) is not int or not 0 <= size <= capacity:
            raise ValueError("Defense replay size is outside capacity")
        if (type(next_index) is not int or not 0 <= next_index < capacity
                or (size < capacity and next_index != size)):
            raise ValueError("Defense replay ring index is inconsistent")
        tensors = {}
        for name, shape, dtype in (("features", (size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), torch.float32),
                                   ("legal_mask", (size, ACTION_SIZE), torch.bool),
                                   ("losing_mask", (size, ACTION_SIZE), torch.bool)):
            tensor = payload[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.shape != shape or tensor.dtype != dtype or tensor.device.type != "cpu"):
                raise ValueError(f"Invalid defense checkpoint {name} tensor")
            tensors[name] = tensor.detach()
        certificates = payload["certificates"]
        if not isinstance(certificates, list) or len(certificates) != size:
            raise ValueError("Defense certificates must match occupied ring slots")
        rows = []
        for index, certificate in enumerate(certificates):
            if not isinstance(certificate, dict) or set(certificate) != {"case_id", "family_id", "proof"}:
                raise ValueError("Invalid defense certificate fields")
            rows.append(CertifiedDefenseSample(**certificate,
                **{name: tensor[index] for name, tensor in tensors.items()}))
        restored = cls(capacity)
        restored.extend(rows)
        restored._next_index = next_index
        return restored
