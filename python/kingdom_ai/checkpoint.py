"""Portable state_dict checkpoints with strict input/output schema metadata."""

from __future__ import annotations

import hashlib
import json
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


def _load_model_payload(payload, device: str | torch.device = "cpu") -> PolicyValueNet:
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
    # Validation/export must not consume the caller's random training state.
    with torch.random.fork_rng(devices=[]):
        model = PolicyValueNet(**config)
    state_dict = payload["state_dict"]
    expected_parameters = model.state_dict()
    if not isinstance(state_dict, dict) or set(state_dict) != set(expected_parameters):
        raise ValueError("checkpoint parameters do not match the model configuration")
    for name, expected in expected_parameters.items():
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint parameter {name} is not a tensor")
        if (value.layout != torch.strided or value.shape != expected.shape
                or value.dtype != expected.dtype):
            raise ValueError(f"checkpoint parameter {name} has the wrong shape or dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint parameter {name} contains NaN or infinity")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_model(path: str | Path, device: str | torch.device = "cpu") -> PolicyValueNet:
    # CPU loading also accepts a checkpoint produced on another accelerator.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _load_model_payload(payload, device)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_digest(model: PolicyValueNet) -> str:
    # Match the fingerprints used by the tactical and two-model match tools.
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _same_primitive(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_primitive(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_primitive(left, right) for left, right in zip(actual, expected))
    return actual == expected


def export_training_model(
    source: str | Path,
    output: str | Path,
    *,
    role: str = "learner",
    manifest_path: str | Path | None = None,
) -> dict:
    """Extract CPU inference weights without resuming training or loading CUDA.

    Only the game/input schema, selected model and provenance counters are
    validated. Optimizer/replay state is deliberately not restored or audited.
    Continue training from the full source checkpoint, not the exported model.
    Existing destinations and ``best.pt`` are never overwritten. The separate
    JSON manifest keeps the portable model's strict schema unchanged.
    """
    if type(role) is not str or role not in ("learner", "champion"):
        raise ValueError("role must be 'learner' or 'champion'")
    # Non-strict resolution tolerates Windows environments that permit file
    # reads but deny GetFinalPathNameByHandle. Opening below still requires an
    # existing readable source; destination creation remains exclusive.
    source_path = Path(source).resolve()
    target = Path(output).absolute()
    manifest = (Path(manifest_path).absolute() if manifest_path is not None
                else target.with_suffix(".manifest.json"))
    if target.name.lower() == "best.pt":
        raise ValueError("Export to a separate candidate path, not best.pt")
    resolved = (target.resolve(), manifest.resolve())
    if source_path in resolved or resolved[0] == resolved[1]:
        raise ValueError("Source, output and manifest must be different paths")
    for path in (target, manifest):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")

    source_sha256 = _file_digest(source_path)
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a full training checkpoint")
    version = payload.get("checkpoint_version")
    if type(version) is not int or version not in range(1, 9):
        raise ValueError("Unsupported full training checkpoint version")
    expected_schema = {
        "format_version": FORMAT_VERSION, "board_size": BOARD_SIZE,
        "action_size": ACTION_SIZE, "feature_names": list(FEATURE_NAMES),
        "perspective": "to_play",
        "rules": {"suicide": "Loses", "own_territory_moves": False,
                  "single_edge_territory": True, "stones_per_player": 41},
    }
    if not _same_primitive(payload.get("schema"), expected_schema):
        raise ValueError("Training checkpoint game/input schema does not match")
    progress = payload.get("progress")
    if (not isinstance(progress, dict)
            or any(type(progress.get(key)) is not int or progress[key] < 0
                   for key in ("iteration", "training_steps", "champion_version"))
            or progress["champion_version"] > progress["iteration"]):
        raise ValueError("Invalid training checkpoint provenance counters")
    state_key = "model" if role == "learner" else "champion"
    portable = {key: value for key, value in expected_schema.items() if key != "rules"}
    portable.update(model_config=payload.get("model_config"), state_dict=payload.get(state_key))
    model = _load_model_payload(portable)
    if _file_digest(source_path) != source_sha256:
        raise RuntimeError("Source checkpoint changed during extraction; retry at a saved boundary")
    provenance = {
        "manifest_version": 1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_version": version,
        "role": role,
        "iteration": progress["iteration"],
        "training_steps": progress["training_steps"],
        "champion_version": progress["champion_version"],
        "model_config": model.model_config,
        "output": str(target.resolve()),
        "weights_sha256": _weights_digest(model),
    }
    # Exclusive creation also protects against a destination appearing between
    # validation and writing. On failure, remove only files created by this call.
    created = []
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination:
            created.append(target)
            torch.save(portable, destination)
        provenance["output_checkpoint_sha256"] = _file_digest(target)
        with manifest.open("x", encoding="utf-8") as destination:
            created.append(manifest)
            json.dump(provenance, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write("\n")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return provenance
