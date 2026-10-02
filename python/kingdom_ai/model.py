"""Small policy/value network for the versioned Great Kingdom input schema."""

from __future__ import annotations

from math import gcd

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = gcd(8, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


class PolicyValueNet(nn.Module):
    """Return policy logits [N, 82] and current-player values [N].

    Policy logits deliberately include every action. Apply ``masked_policy``
    with the engine's legal mask when choosing a move. Values use tanh, so -1
    denotes a loss and +1 a win for the player encoded as ``to_play``.
    """

    def __init__(self, channels: int = 32, residual_blocks: int = 2) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(residual_blocks) is not int or residual_blocks < 0:
            raise ValueError("residual_blocks must be a non-negative integer")
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.trunk = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.GroupNorm(gcd(8, channels), channels),
            nn.ReLU(),
            *[_ResidualBlock(channels) for _ in range(residual_blocks)],
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.GroupNorm(1, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.GroupNorm(1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    @property
    def model_config(self) -> dict[str, int]:
        return {"channels": self.channels, "residual_blocks": self.residual_blocks}

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or tuple(features.shape[1:]) != (
            INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
        ):
            raise ValueError(
                f"features must have shape [N, {INPUT_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}]"
            )
        if not features.is_floating_point():
            raise TypeError("features must be a floating-point tensor")
        shared = self.trunk(features)
        return self.policy_head(shared), self.value_head(shared).squeeze(-1)


def masked_policy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over legal actions, with exact zeros for every forbidden action.

    All-false rows, such as terminal states, return all zeros. Replacing those
    rows before softmax prevents the NaNs produced by softmax of only -inf.
    """
    if logits.ndim != 2 or logits.shape[1] != ACTION_SIZE:
        raise ValueError(f"logits must have shape [N, {ACTION_SIZE}]")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if legal_mask.dtype != torch.bool:
        raise TypeError("legal_mask must have dtype bool")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask and logits must have identical shapes")
    if legal_mask.device != logits.device:
        raise ValueError("legal_mask and logits must be on the same device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")
    any_legal = legal_mask.any(dim=1, keepdim=True)
    masked = logits.masked_fill(~legal_mask, -torch.inf)
    safe_logits = torch.where(any_legal, masked, torch.zeros_like(logits))
    return torch.softmax(safe_logits, dim=1).masked_fill(~legal_mask, 0.0)
