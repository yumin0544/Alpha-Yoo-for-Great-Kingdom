"""Bounded, resumable WIN-policy/LOSS-value teacher replay and certificates."""

from copy import deepcopy
import json

import my_board_engine as engine
import torch

from .replay import ReplayBuffer
from .tactical_training import CertifiedTacticalSample, _validated_samples
from .training import ProofTrainingBatch, TrainingSample


class CertifiedTacticalReplay:
    """Keep physical ring order and certificates so seeded resume is exact.

    A checkpoint preserves the solver's audit records, not a re-verification of
    the entire minimax tree. Loading validates schema, WIN/LOSS value/policy
    support and matching metadata; it does not invent proofs for observations.
    """

    def __init__(self, capacity):
        self._buffer = ReplayBuffer(capacity)
        self._certificates = []

    @property
    def capacity(self):
        return self._buffer.capacity

    def __len__(self):
        return len(self._buffer)

    @staticmethod
    def _metadata(row):
        for value in (row.case_id, row.family_id, row.motif):
            if type(value) is not str or not value:
                raise ValueError("Teacher identifiers must be nonempty strings")
        depth = row.proof.get("proof_depth")
        if type(depth) is not int or depth < 1:
            raise ValueError("A teacher certificate needs a positive proof depth")
        try:
            json.dumps(row.proof, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Teacher certificate must contain finite JSON data") from error
        return {"case_id": row.case_id, "family_id": row.family_id,
                "motif": row.motif, "proof": deepcopy(row.proof)}

    def extend(self, rows):
        rows = _validated_samples(rows, allow_loss=True)
        metadata = [self._metadata(row) for row in rows]
        if not rows:
            return
        next_index = self._buffer.state_dict()["next_index"]
        self._buffer.extend([row.sample for row in rows])
        for index, certificate in enumerate(metadata):
            slot = (next_index + index) % self.capacity
            if slot == len(self._certificates):
                self._certificates.append(certificate)
            else:
                self._certificates[slot] = certificate

    def sample(self, batch_size, *, generator, device="cpu"):
        batch = self._buffer.sample(batch_size, generator=generator, device=device)
        # Validation requires WIN/+1 with policy and LOSS/-1 without policy.
        # Deriving the sampled mask from that invariant avoids a second RNG
        # draw and keeps masks aligned with physical ring slots on resume.
        return ProofTrainingBatch(batch.features, batch.legal_mask, batch.policy,
                                  batch.value, batch.value == 1)

    def state_dict(self):
        return {"version": 2, "replay": self._buffer.state_dict(),
                "certificates": deepcopy(self._certificates),
                "policy_enabled": torch.tensor([
                    row["proof"]["outcome"] == "WIN" for row in self._certificates],
                    dtype=torch.bool, device="cpu")}

    @classmethod
    def from_state_dict(cls, payload):
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int
                or payload["version"] not in (1, 2)):
            raise ValueError("Invalid teacher replay checkpoint")
        expected = {"version", "replay", "certificates"}
        if payload["version"] == 2:
            expected.add("policy_enabled")
        if set(payload) != expected:
            raise ValueError("Invalid teacher replay checkpoint fields")
        replay = ReplayBuffer.from_state_dict(payload["replay"])
        certificates = payload["certificates"]
        if not isinstance(certificates, list) or len(certificates) != len(replay):
            raise ValueError("Teacher certificates must match occupied replay slots")
        policy_enabled = (torch.ones(len(replay), dtype=torch.bool)
                          if payload["version"] == 1 else payload["policy_enabled"])
        if (not isinstance(policy_enabled, torch.Tensor) or policy_enabled.layout != torch.strided
                or policy_enabled.dtype != torch.bool or policy_enabled.shape != (len(replay),)):
            raise ValueError("Teacher policy mask must be a bool tensor matching occupied slots")
        policy_enabled = policy_enabled.detach().to(device="cpu")
        data = replay.state_dict()
        rows = []
        for index, certificate in enumerate(certificates):
            if (not isinstance(certificate, dict)
                    or set(certificate) != {"case_id", "family_id", "motif", "proof"}
                    or not isinstance(certificate["proof"], dict)):
                raise ValueError("Invalid teacher certificate fields")
            actor = engine.Cell.Black if int(data["to_play"][index]) == 1 else engine.Cell.White
            sample = TrainingSample(data["features"][index], data["legal_mask"][index],
                                    data["policy"][index], float(data["value"][index]), actor)
            row = CertifiedTacticalSample(sample=sample, **certificate,
                                           policy_enabled=bool(policy_enabled[index]))
            cls._metadata(row)
            rows.append(row)
        _validated_samples(rows, allow_loss=payload["version"] == 2)
        restored = cls(replay.capacity)
        restored._buffer = replay
        restored._certificates = deepcopy(certificates)
        return restored
