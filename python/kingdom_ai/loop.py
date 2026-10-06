"""CPU or batched CUDA self-play, replay learning, evaluation and full resume."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter

import torch

from .checkpoint import save_model
from .augmentation import augment_batch
from .encoding import ACTION_SIZE, BOARD_SIZE, FEATURE_NAMES, FORMAT_VERSION
from .evaluation import evaluate_models
from .gpu_puct import GpuPUCTOptions
from .gpu_training import collect_gpu_puct_games
from .model import PolicyValueNet
from .puct import PUCTOptions
from .replay import ReplayBuffer
from .training import collect_puct_game, train_step


@dataclass(frozen=True)
class TrainingConfig:
    games_per_iteration: int = 4
    simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    replay_capacity: int = 10_000
    batch_size: int = 64
    train_steps_per_iteration: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    evaluation_games: int = 20
    evaluation_simulations: int = 128
    evaluation_opening_moves: int = 6
    evaluation_opening_temperature: float = 1.0
    promotion_threshold: float = 0.55
    seed: int = 42
    self_play_backend: str = "cpu"
    self_play_batch_size: int = 128
    augment_symmetries: bool = False
    temperature_moves: int | None = None
    final_temperature: float = 0.0
    self_play_tactical_checks: bool = False
    self_play_fpu_reduction: float | None = None

    def __post_init__(self):
        positive_ints = (
            "games_per_iteration", "simulations", "replay_capacity", "batch_size",
            "train_steps_per_iteration", "evaluation_games", "evaluation_simulations",
            "self_play_batch_size",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.self_play_backend) is not str or self.self_play_backend not in ("cpu", "cuda"):
            raise ValueError("self_play_backend must be 'cpu' or 'cuda'")
        if self.self_play_backend == "cuda" and self.simulations > 2 ** 31 - 2:
            raise ValueError("CUDA simulations must not exceed 2147483646")
        if type(self.augment_symmetries) is not bool or type(self.self_play_tactical_checks) is not bool:
            raise ValueError("augment_symmetries and self_play_tactical_checks must be bool")
        if self.temperature_moves is not None and (
                type(self.temperature_moves) is not int or self.temperature_moves < 0):
            raise ValueError("temperature_moves must be None or a non-negative integer")
        if self.self_play_fpu_reduction is not None and (
                isinstance(self.self_play_fpu_reduction, bool)
                or not isinstance(self.self_play_fpu_reduction, (int, float))
                or not math.isfinite(self.self_play_fpu_reduction)
                or self.self_play_fpu_reduction < 0):
            raise ValueError("self_play_fpu_reduction must be None or a finite non-negative real")
        if self.self_play_backend != "cuda" and (
                self.self_play_tactical_checks or self.self_play_fpu_reduction is not None):
            raise ValueError("Training tactical checks and FPU require CUDA self-play")
        if self.evaluation_games < 2 or self.evaluation_games % 2:
            raise ValueError("evaluation_games must be even and at least two")
        if type(self.evaluation_opening_moves) is not int or self.evaluation_opening_moves < 0:
            raise ValueError("evaluation_opening_moves must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for name in (
            "c_puct", "dirichlet_alpha", "dirichlet_epsilon", "temperature",
            "learning_rate", "weight_decay", "evaluation_opening_temperature",
            "promotion_threshold",
            "final_temperature",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        for name in ("c_puct", "dirichlet_alpha", "learning_rate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("temperature", "final_temperature", "weight_decay", "evaluation_opening_temperature"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if not 0.5 <= self.promotion_threshold <= 1:
            raise ValueError("promotion_threshold must be in [0.5, 1]")


_CHECKPOINT_VERSION = 4
_GPU_CONFIG_KEYS = {"self_play_backend", "self_play_batch_size"}
_STRENGTH_CONFIG_KEYS = {
    "augment_symmetries", "temperature_moves", "final_temperature",
    "self_play_tactical_checks", "self_play_fpu_reduction",
}
_SCHEMA = {
    "format_version": FORMAT_VERSION, "board_size": BOARD_SIZE,
    "action_size": ACTION_SIZE, "feature_names": list(FEATURE_NAMES),
    "perspective": "to_play",
    "rules": {"suicide": "Loses", "own_territory_moves": False,
              "single_edge_territory": True, "stones_per_player": 41},
}
_CHECKPOINT_KEYS = {
    "checkpoint_version", "schema", "config", "model_config", "model", "champion",
    "optimizer", "replay", "progress", "generator_state", "rng", "runtime",
    "last_metrics",
}
_RUNTIME_KEYS = {"evaluation_workers"}
_PROGRESS_KEYS = {"iteration", "self_play_games", "training_steps", "champion_version"}


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Checkpoint state must contain only tensors and primitive values")


def _same_primitive(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_primitive(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _same_primitive(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _model_state(model):
    state = _cpu_copy(model.state_dict())
    if any(tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
           for tensor in state.values()):
        raise ValueError("Training models must contain finite float32 parameters")
    return state


def _restore_model(config, state):
    if not isinstance(config, dict) or set(config) != {"channels", "residual_blocks"}:
        raise ValueError("Invalid training checkpoint model configuration")
    # Constructing a validation model must not consume the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        model = PolicyValueNet(**config)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint model parameters do not match the architecture")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != expected[name].shape or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"Invalid checkpoint model parameter: {name}")
    model.load_state_dict(state, strict=True)
    return model


def _atomic_write(path, writer):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_optimizer(optimizer, payload, steps):
    expected = optimizer.state_dict()
    if not isinstance(payload, dict) or set(payload) != {"state", "param_groups"}:
        raise ValueError("Invalid optimizer checkpoint")
    groups = payload["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("Invalid optimizer parameter groups")
    if not _same_primitive(groups[0], expected["param_groups"][0]):
        raise ValueError("Optimizer settings do not match the training configuration")
    states = payload["state"]
    parameters = list(optimizer.param_groups[0]["params"])
    if (not isinstance(states, dict) or any(type(key) is not int for key in states)
            or set(states) != (set(range(len(parameters))) if steps else set())):
        raise ValueError("Optimizer parameter states are inconsistent with training steps")
    for index, state in states.items():
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid Adam state fields")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.shape != ()
                or not step.is_floating_point() or not bool(torch.isfinite(step))
                or float(step) != steps):
            raise ValueError("Adam step does not match training progress")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != parameters[index].shape or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())
                    or (key == "exp_avg_sq" and bool((value < 0).any()))):
                raise ValueError("Invalid Adam moment tensors")


def _validate_rng(payload):
    if not isinstance(payload, dict) or set(payload) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("Invalid RNG checkpoint fields")
    try:
        random.Random().setstate(payload["python"])
        state = payload["torch_cpu"]
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Invalid CPU RNG state")
        torch.Generator(device="cpu").set_state(state)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("Invalid Python/CPU RNG checkpoint state") from error
    cuda_states = payload["torch_cuda"]
    if (not isinstance(cuda_states, list) or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            or state.ndim != 1 or state.numel() == 0 for state in cuda_states)):
        raise ValueError("Invalid CUDA RNG checkpoint state")


class Trainer:
    """Run additional complete iterations; checkpoints are iteration boundaries.

    Self-play uses the current champion. The learner keeps its optimizer and
    continues training even when evaluation rejects promotion. A failed or
    interrupted iteration must be discarded by loading the last checkpoint.
    """

    def __init__(self, config=TrainingConfig(), model=None, device="cpu", evaluation_workers=1):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        if model is not None and not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet")
        if type(evaluation_workers) is not int or evaluation_workers < 1:
            raise ValueError("evaluation_workers must be a positive integer")
        self.config = config
        self.device = torch.device(device)
        # Parallel batching can change floating-point scheduling, so preserve
        # this operational setting in checkpoints even though it is not part of
        # TrainingConfig and can be explicitly overridden when loading.
        self.evaluation_workers = evaluation_workers
        if config.self_play_backend == "cuda":
            if self.device.type != "cuda":
                raise ValueError("CUDA self-play requires a CUDA Trainer device (device='cuda')")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA self-play requires an available CUDA device")
        if model is None:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.seed)
                model = PolicyValueNet()
        _model_state(model)
        # Own the supplied model: external updates cannot change a running trainer.
        self.model = deepcopy(model).to(self.device)
        self.champion = deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.iteration = self.self_play_games = self.training_steps = self.champion_version = 0
        self._at_boundary = True
        self._last_metrics = None

    def _next_seed(self):
        return int(torch.randint(2 ** 63 - 1, (1,), generator=self.generator, device="cpu").item())

    def reconfigure(self, **overrides):
        """Change learning/search settings at a saved boundary without resetting state.

        Historical counter invariants require the iteration game/update budgets
        and replay capacity to remain fixed. Architecture, RNG, replay contents,
        optimizer moments and progress are preserved. Call save_checkpoint() on
        a NEW path to persist the changed settings before continuing.
        """
        if not self._at_boundary:
            raise RuntimeError("Cannot reconfigure an incomplete iteration")
        immutable = {"games_per_iteration", "train_steps_per_iteration", "replay_capacity", "seed"}
        if immutable.intersection(overrides):
            raise ValueError("Reconfiguration cannot change games, training steps, replay capacity or seed")
        try:
            config = replace(self.config, **overrides)
        except TypeError as error:
            raise ValueError("Unknown training configuration field") from error
        if config.self_play_backend == "cuda" and self.device.type != "cuda":
            raise ValueError("CUDA self-play requires a CUDA Trainer device")
        # Validation above completes before changing any owned state.
        self.config = config
        for group in self.optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["weight_decay"] = config.weight_decay

    def _collect_games(self):
        config = self.config
        schedule_options = ({} if config.temperature_moves is None else {
            "temperature_moves": config.temperature_moves,
            "final_temperature": config.final_temperature,
        })
        if config.self_play_backend == "cuda":
            # Each chunk owns temporary GPU RNG streams seeded from the saved
            # CPU generator. No unsaved searcher survives an iteration boundary.
            for offset in range(0, config.games_per_iteration, config.self_play_batch_size):
                seed = self._next_seed()
                options = GpuPUCTOptions(
                    simulations=config.simulations, c_puct=config.c_puct, seed=seed,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_epsilon,
                    tactical_checks=config.self_play_tactical_checks,
                    fpu_reduction=config.self_play_fpu_reduction,
                )
                count = min(config.self_play_batch_size, config.games_per_iteration - offset)
                yield from collect_gpu_puct_games(
                    self.champion, count, options=options, temperature=config.temperature,
                    seed=seed, batch_size=config.self_play_batch_size, device=self.device,
                    **schedule_options,
                )
        else:
            for _ in range(config.games_per_iteration):
                seed = self._next_seed()
                options = PUCTOptions(
                    simulations=config.simulations, c_puct=config.c_puct, seed=seed,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_epsilon,
                )
                yield collect_puct_game(self.champion, options=options,
                                        temperature=config.temperature, seed=seed,
                                        **schedule_options)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_iteration(self):
        if not self._at_boundary:
            raise RuntimeError("Reload the last checkpoint after an incomplete iteration")
        self._at_boundary = False
        self._synchronize()
        started = perf_counter()
        config = self.config
        endings = {}
        winners = {"Black": 0, "White": 0}
        game_plies = []
        plies = samples = 0
        replay_store_seconds = 0.0
        pending_samples = []
        pending_games = 0

        def flush_replay():
            nonlocal replay_store_seconds, pending_games
            if not pending_samples:
                return
            replay_started = perf_counter()
            self.replay.extend(pending_samples)
            replay_store_seconds += perf_counter() - replay_started
            pending_samples.clear()
            pending_games = 0

        for data in self._collect_games():
            pending_samples.extend(data.samples)
            pending_games += 1
            self.self_play_games += 1
            samples += len(data.samples)
            plies += len(data.samples)
            game_plies.append(len(data.samples))
            endings[data.reason.name] = endings.get(data.reason.name, 0) + 1
            winners[data.winner.name] += 1
            # CUDA collection already materializes one complete game chunk.
            # Insert that chunk in one vectorized replay operation. CPU games
            # remain streaming so sequential collection does not accumulate.
            if (config.self_play_backend == "cpu"
                    or pending_games == config.self_play_batch_size):
                flush_replay()
        flush_replay()
        self._synchronize()
        self_play_seconds = perf_counter() - started
        self_play_compute_seconds = max(0.0, self_play_seconds - replay_store_seconds)
        training_started = perf_counter()
        loss_totals = dict.fromkeys(("loss", "policy_loss", "value_loss"), 0.0)
        for _ in range(config.train_steps_per_iteration):
            batch = self.replay.sample(config.batch_size, generator=self.generator,
                                       device=self.device)
            if config.augment_symmetries:
                batch = augment_batch(batch, generator=self.generator)
            losses = train_step(self.model, self.optimizer, batch)
            for key in loss_totals:
                loss_totals[key] += losses[key]
            self.training_steps += 1
        self._synchronize()
        training_seconds = perf_counter() - training_started
        evaluation_started = perf_counter()
        evaluation_options = ({"tactical_checks": True} if config.self_play_tactical_checks else {})
        evaluation = evaluate_models(
            self.model, self.champion, games=config.evaluation_games,
            simulations=config.evaluation_simulations, c_puct=config.c_puct,
            seed=self._next_seed(), opening_moves=config.evaluation_opening_moves,
            opening_temperature=config.evaluation_opening_temperature,
            workers=self.evaluation_workers,
            **evaluation_options,
        )
        promoted = evaluation.win_rate >= config.promotion_threshold
        if promoted:
            self.champion = deepcopy(self.model).eval()
            self.champion_version += 1
        self._synchronize()
        evaluation_seconds = perf_counter() - evaluation_started
        self.iteration += 1
        elapsed_seconds = perf_counter() - started
        training_samples_drawn = config.batch_size * config.train_steps_per_iteration
        retained_new_samples = min(samples, len(self.replay))
        sorted_plies = sorted(game_plies)
        p95_plies = sorted_plies[math.ceil(0.95 * len(sorted_plies)) - 1]
        metrics = {
            "metrics_schema_version": 2,
            "iteration": self.iteration, "self_play_games": self.self_play_games,
            "training_steps": self.training_steps, "champion_version": self.champion_version,
            "generated_samples": samples, "replay_size": len(self.replay),
            "mean_self_play_plies": plies / config.games_per_iteration,
            "p95_self_play_plies": p95_plies,
            "max_self_play_plies": sorted_plies[-1],
            "self_play_endings": endings,
            "self_play_winners": winners,
            **{key: value / config.train_steps_per_iteration for key, value in loss_totals.items()},
            "evaluation": {**asdict(evaluation), "win_rate": evaluation.win_rate},
            "promoted": promoted, "self_play_seconds": self_play_seconds,
            "self_play_compute_seconds": self_play_compute_seconds,
            "replay_store_seconds": replay_store_seconds,
            "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
            "elapsed_seconds": elapsed_seconds,
            "evaluation_workers": self.evaluation_workers,
            "self_play_backend": config.self_play_backend,
            "self_play_batch_size": config.self_play_batch_size,
            "games_per_iteration": config.games_per_iteration,
            "replay_capacity": config.replay_capacity,
            "retained_new_samples": retained_new_samples,
            "retained_new_sample_ratio": retained_new_samples / samples,
            "replay_turnover": samples / config.replay_capacity,
            "training_batch_size": config.batch_size,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "training_samples_drawn": training_samples_drawn,
            "training_draws_per_generated_sample": training_samples_drawn / samples,
            "training_draws_per_replay_sample": training_samples_drawn / len(self.replay),
            "augment_symmetries": config.augment_symmetries,
            "self_play_simulations": config.simulations,
            "self_play_tactical_checks": config.self_play_tactical_checks,
            "self_play_fpu_reduction": config.self_play_fpu_reduction,
            "self_play_games_per_second": config.games_per_iteration / self_play_seconds,
            "self_play_positions_per_second": samples / self_play_seconds,
            "self_play_compute_positions_per_second": (
                samples / self_play_compute_seconds if self_play_compute_seconds else 0.0),
            "replay_store_positions_per_second": (
                samples / replay_store_seconds if replay_store_seconds else 0.0),
            "iteration_games_per_second": config.games_per_iteration / elapsed_seconds,
        }
        # Reject non-finite metrics before permitting a resumable boundary save.
        json.dumps(metrics, allow_nan=False)
        self._last_metrics = deepcopy(metrics)
        self._at_boundary = True
        return metrics

    def save_checkpoint(self, path):
        if not self._at_boundary:
            raise RuntimeError("An incomplete iteration cannot be checkpointed; reload the last save")
        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION, "schema": deepcopy(_SCHEMA),
            "config": asdict(self.config), "model_config": self.model.model_config,
            "model": _model_state(self.model), "champion": _model_state(self.champion),
            "optimizer": _cpu_copy(self.optimizer.state_dict()), "replay": self.replay.state_dict(),
            "progress": {name: getattr(self, name) for name in _PROGRESS_KEYS},
            "generator_state": self.generator.get_state().clone(),
            "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
                    "torch_cuda": ([state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                                   if torch.cuda.is_available() else [])},
            "runtime": {"evaluation_workers": self.evaluation_workers},
            "last_metrics": deepcopy(self._last_metrics),
        }
        _atomic_write(path, lambda temporary: torch.save(payload, temporary))

    @classmethod
    def load_checkpoint(cls, path, device="cpu", evaluation_workers=None):
        if (evaluation_workers is not None
                and (type(evaluation_workers) is not int or evaluation_workers < 1)):
            raise ValueError("evaluation_workers must be None or a positive integer")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Unexpected training checkpoint fields")
        version = payload.get("checkpoint_version")
        if type(version) is not int or version not in (1, 2, 3, _CHECKPOINT_VERSION):
            raise ValueError("Unsupported training checkpoint version")
        expected_keys = (_CHECKPOINT_KEYS if version >= 4
                         else _CHECKPOINT_KEYS - {"runtime"})
        if set(payload) != expected_keys:
            raise ValueError("Unexpected training checkpoint fields")
        if not _same_primitive(payload["schema"], _SCHEMA):
            raise ValueError("Training checkpoint game/input schema does not match")
        raw_config = payload["config"]
        config_keys = {field.name for field in fields(TrainingConfig)}
        if version < 3:
            config_keys -= _STRENGTH_CONFIG_KEYS
        if version == 1:
            config_keys -= _GPU_CONFIG_KEYS
        if not isinstance(raw_config, dict) or set(raw_config) != config_keys:
            raise ValueError("Invalid checkpoint training configuration")
        config = TrainingConfig(**raw_config)
        progress = payload["progress"]
        if (not isinstance(progress, dict) or set(progress) != _PROGRESS_KEYS
                or any(type(value) is not int or value < 0 for value in progress.values())
                or progress["self_play_games"] != progress["iteration"] * config.games_per_iteration
                or progress["training_steps"] != progress["iteration"] * config.train_steps_per_iteration
                or progress["champion_version"] > progress["iteration"]):
            raise ValueError("Checkpoint progress counters are inconsistent")
        model = _restore_model(payload["model_config"], payload["model"])
        champion = _restore_model(payload["model_config"], payload["champion"])
        replay = ReplayBuffer.from_state_dict(payload["replay"])
        if (replay.capacity != config.replay_capacity
                or (progress["iteration"] == 0 and len(replay) != 0)
                or (progress["iteration"] > 0 and len(replay) == 0)):
            raise ValueError("Replay state is inconsistent with training progress")
        validation_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                                weight_decay=config.weight_decay)
        _validate_optimizer(validation_optimizer, payload["optimizer"], progress["training_steps"])
        try:
            private_generator = torch.Generator(device="cpu")
            generator_state = payload["generator_state"]
            if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
                raise ValueError("Invalid private generator state")
            private_generator.set_state(generator_state)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Invalid private generator checkpoint state") from error
        _validate_rng(payload["rng"])
        last_metrics = payload["last_metrics"]
        if ((progress["iteration"] == 0 and last_metrics is not None)
                or (progress["iteration"] > 0 and (
                    not isinstance(last_metrics, dict)
                    or any(last_metrics.get(name) != value for name, value in progress.items())))):
            raise ValueError("Checkpoint metrics do not match training progress")
        try:
            json.dumps(last_metrics, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid checkpoint metrics") from error
        cuda_states = payload["rng"]["torch_cuda"]
        if cuda_states and torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG configuration")
        saved_evaluation_workers = 1
        if version >= 4:
            runtime = payload["runtime"]
            if (not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS
                    or type(runtime["evaluation_workers"]) is not int
                    or runtime["evaluation_workers"] < 1):
                raise ValueError("Invalid checkpoint runtime settings")
            saved_evaluation_workers = runtime["evaluation_workers"]
        trainer = cls(config, model=model, device=device,
                      evaluation_workers=(saved_evaluation_workers
                                          if evaluation_workers is None
                                          else evaluation_workers))
        trainer.champion = champion.to(trainer.device).eval()
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.replay = replay
        trainer.generator = private_generator
        for name, value in progress.items():
            setattr(trainer, name, value)
        trainer._last_metrics = deepcopy(last_metrics)
        # Restore RNG last: validation and object construction must not disturb it.
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
        return trainer

    def export_champion(self, path):
        if not self._at_boundary:
            raise RuntimeError("Export the champion only at a completed iteration boundary")
        _atomic_write(path, lambda temporary: save_model(self.champion, temporary))

    def run(self, iterations, checkpoint_path=None, metrics_path=None, on_iteration=None,
            collect_metrics=True):
        """Run additional iterations, saving before work and after each completion."""
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if on_iteration is not None and not callable(on_iteration):
            raise TypeError("on_iteration must be callable")
        if type(collect_metrics) is not bool:
            raise TypeError("collect_metrics must be bool")
        if (checkpoint_path is not None and metrics_path is not None
                and Path(checkpoint_path).resolve() == Path(metrics_path).resolve()):
            raise ValueError("Checkpoint and metrics paths must be different")
        initial_checkpoint_seconds = 0.0
        if checkpoint_path is not None:
            initial_checkpoint_started = perf_counter()
            self.save_checkpoint(checkpoint_path)
            initial_checkpoint_seconds = perf_counter() - initial_checkpoint_started
        output = []
        for run_index in range(iterations):
            metrics = self.run_iteration()
            checkpoint_seconds = 0.0
            checkpoint_bytes = 0
            if checkpoint_path is not None:
                checkpoint_started = perf_counter()
                self.save_checkpoint(checkpoint_path)
                checkpoint_seconds = perf_counter() - checkpoint_started
                checkpoint_bytes = Path(checkpoint_path).stat().st_size
            metrics["checkpoint_seconds"] = checkpoint_seconds
            metrics["initial_checkpoint_seconds"] = (
                initial_checkpoint_seconds if run_index == 0 else 0.0)
            metrics["checkpoint_bytes"] = checkpoint_bytes
            metrics["checkpoint_written"] = checkpoint_path is not None
            metrics["elapsed_with_checkpoint_seconds"] = (
                metrics["elapsed_seconds"] + checkpoint_seconds
                + metrics["initial_checkpoint_seconds"])
            json.dumps(metrics, allow_nan=False)
            # A checkpoint cannot contain the duration of its own write. These
            # operational fields therefore live in JSONL/return/callback data;
            # checkpoint last_metrics remains the resumable iteration payload.
            if metrics_path is not None:
                target = Path(metrics_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
            if on_iteration is not None:
                on_iteration(deepcopy(metrics))
            if collect_metrics:
                output.append(metrics)
        return output
