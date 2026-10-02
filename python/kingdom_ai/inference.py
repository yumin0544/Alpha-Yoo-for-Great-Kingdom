"""Connect policy/value inference to snapshots of the C++ rules engine."""

from __future__ import annotations

from dataclasses import dataclass

import my_board_engine as engine
import torch

from .encoding import ACTION_SIZE, action_to_move, encode_state, terminal_value
from .model import PolicyValueNet, masked_policy


@dataclass(frozen=True)
class Prediction:
    policy: torch.Tensor
    value: float
    best_action: int | None
    best_move: engine.Move | None


class NeuralAgent:
    """Choose the highest policy probability among the engine's legal moves.

    This adapter does one network evaluation. It does not perform tree search,
    and a newly initialized model has not yet learned to play well.
    """

    def __init__(
        self, model: PolicyValueNet, device: str | torch.device | None = None
    ) -> None:
        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet")
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

    def predict(self, state: engine.State) -> Prediction:
        if not isinstance(state, engine.State):
            raise TypeError("state must be a my_board_engine.State")
        snapshot = state.copy()
        encoded = encode_state(snapshot, device=self.device)
        if snapshot.result.finished():
            return Prediction(
                policy=torch.zeros(ACTION_SIZE, dtype=torch.float32),
                value=terminal_value(snapshot),
                best_action=None,
                best_move=None,
            )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                logits, values = self.model(encoded.features.unsqueeze(0))
                policy = masked_policy(logits, encoded.legal_mask.unsqueeze(0))[0]
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("model returned a non-finite value")
                value = float(values[0].item())
                best_action = int(policy.argmax().item())
                policy_cpu = policy.to(device="cpu", dtype=torch.float32)
        finally:
            self.model.train(was_training)
        # Clone outside inference_mode so callers can safely reuse this tensor
        # as an ordinary detached target in a later training computation.
        return Prediction(
            policy=policy_cpu.clone(),
            value=value,
            best_action=best_action,
            best_move=action_to_move(best_action),
        )
