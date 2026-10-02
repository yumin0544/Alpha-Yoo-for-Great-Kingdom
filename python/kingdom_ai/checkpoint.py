"""Portable state_dict checkpoints with strict input/output schema metadata."""

from __future__ import annotations

from pathlib import Path

import torch

from .encoding import ACTION_SIZE, BOARD_SIZE, FEATURE_NAMES, FORMAT_VERSION
from .model import PolicyValueNet


def save_model(model: PolicyValueNet, path: str | Path) -> None:
    if not isinstance(model, PolicyValueNet):
        raise TypeError("model must be a PolicyValueNet")
    state_dict = {
        name: value.detach().to("cpu").clone()
        for name, value in model.state_dict().items()
    }
    if any(value.dtype != torch.float32 for value in state_dict.values()):
        raise ValueError("schema 1 checkpoints require float32 model parameters")
    if any(not bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise ValueError("cannot save model parameters containing NaN or infinity")
    payload = {
        "format_version": FORMAT_VERSION,
        "board_size": BOARD_SIZE,
        "action_size": ACTION_SIZE,
        "feature_names": list(FEATURE_NAMES),
        "perspective": "to_play",
        "model_config": model.model_config,
        "state_dict": state_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> PolicyValueNet:
    # CPU loading also accepts a checkpoint produced on another accelerator.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    expected_fields = {
        "format_version", "board_size", "action_size", "feature_names",
        "perspective", "model_config", "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("checkpoint is missing or has unexpected schema fields")
    for key, expected in (
        ("format_version", FORMAT_VERSION), ("board_size", BOARD_SIZE),
        ("action_size", ACTION_SIZE),
    ):
        if type(payload[key]) is not int or payload[key] != expected:
            raise ValueError(f"checkpoint {key} does not match the supported schema")
    if (
        type(payload["feature_names"]) is not list
        or any(type(name) is not str for name in payload["feature_names"])
        or payload["feature_names"] != list(FEATURE_NAMES)
    ):
        raise ValueError("checkpoint feature_names do not match the input schema")
    if type(payload["perspective"]) is not str or payload["perspective"] != "to_play":
        raise ValueError("checkpoint value perspective must be to_play")
    config = payload["model_config"]
    if not isinstance(config, dict) or set(config) != {"channels", "residual_blocks"}:
        raise ValueError("checkpoint model_config is invalid")
    model = PolicyValueNet(**config)
    state_dict = payload["state_dict"]
    expected_parameters = model.state_dict()
    if not isinstance(state_dict, dict) or set(state_dict) != set(expected_parameters):
        raise ValueError("checkpoint parameters do not match the model configuration")
    for name, expected in expected_parameters.items():
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint parameter {name} is not a tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"checkpoint parameter {name} has the wrong shape or dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint parameter {name} contains NaN or infinity")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
