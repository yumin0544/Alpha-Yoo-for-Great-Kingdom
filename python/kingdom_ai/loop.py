"""CPU or batched CUDA self-play, replay learning, evaluation and full resume."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import json
import hashlib
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
from .proof_replay import CertifiedTacticalReplay
from .promotion import PromotionLeague
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
    promotion_archive_dir: str | None = None
    promotion_archive_games: int = 100
    seed: int = 42
    self_play_backend: str = "cpu"
    self_play_model: str = "champion"
    self_play_batch_size: int = 128
    augment_symmetries: bool = False
    temperature_moves: int | None = None
    final_temperature: float = 0.0
    self_play_tactical_checks: bool = False
    self_play_fpu_reduction: float | None = None
    online_tactics: bool = False
    online_tactics_max_cases: int = 32
    online_tactics_max_depth: int = 20
    online_tactics_max_nodes: int = 2000000
    online_tactics_time_limit_ms: int = 2000
    online_tactics_generation_seconds: float = 30.0
    online_tactics_fraction: float = 0.25
    online_tactics_replay_capacity: int = 1024
    online_tactics_min_proof_depth: int = 3
    online_tactics_include_loss: bool = False

    def __post_init__(self):
        positive_ints = (
            "games_per_iteration", "simulations", "replay_capacity", "batch_size",
            "train_steps_per_iteration", "evaluation_games", "evaluation_simulations",
            "self_play_batch_size", "promotion_archive_games",
            "online_tactics_max_cases", "online_tactics_max_depth", "online_tactics_max_nodes",
            "online_tactics_time_limit_ms", "online_tactics_replay_capacity",
            "online_tactics_min_proof_depth",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.self_play_backend) is not str or self.self_play_backend not in ("cpu", "cuda"):
            raise ValueError("self_play_backend must be 'cpu' or 'cuda'")
        if type(self.self_play_model) is not str or self.self_play_model not in ("champion", "learner"):
            raise ValueError("self_play_model must be 'champion' or 'learner'")
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
        if type(self.online_tactics) is not bool or type(self.online_tactics_include_loss) is not bool:
            raise ValueError("online_tactics and online_tactics_include_loss must be bool")
        if not self.online_tactics_min_proof_depth <= self.online_tactics_max_depth <= 256:
            raise ValueError("Online tactics requires min_proof_depth <= max_depth <= 256")
        if self.online_tactics_time_limit_ms > 2 ** 31 - 1:
            raise ValueError("Online tactics time limit must fit the solver's signed 32-bit milliseconds")
        if self.online_tactics_max_nodes > 2 ** 64 - 1:
            raise ValueError("Online tactics node limit must fit the solver's unsigned 64-bit counter")
        for name in ("online_tactics_generation_seconds", "online_tactics_fraction"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive real")
        if self.online_tactics_fraction >= 1:
            raise ValueError("online_tactics_fraction must be strictly less than one")
        if self.online_tactics and self.batch_size < 2:
            raise ValueError("Online tactics needs batch_size >= 2 to retain ordinary replay rows")
        if self.evaluation_games < 2 or self.evaluation_games % 2:
            raise ValueError("evaluation_games must be even and at least two")
        if self.promotion_archive_games < 2 or self.promotion_archive_games % 2:
            raise ValueError("promotion_archive_games must be even and at least two")
        if self.promotion_archive_dir is not None:
            if type(self.promotion_archive_dir) is not str or not self.promotion_archive_dir.strip():
                raise ValueError("promotion_archive_dir must be None or a nonempty path string")
            object.__setattr__(self, "promotion_archive_dir", str(Path(self.promotion_archive_dir).resolve()))
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


_CHECKPOINT_VERSION = 9
_PROMOTION_CONFIG_KEYS = {"promotion_archive_dir", "promotion_archive_games"}
_ADAPTIVE_CONFIG_KEYS = {"self_play_model", "online_tactics_include_loss"}
_ONLINE_CONFIG_KEYS = {"online_tactics", "online_tactics_max_cases", "online_tactics_max_depth",
                       "online_tactics_max_nodes", "online_tactics_time_limit_ms",
                       "online_tactics_generation_seconds", "online_tactics_fraction",
                       "online_tactics_replay_capacity", "online_tactics_min_proof_depth"}
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
_CHECKPOINT_KEYS_V7 = _CHECKPOINT_KEYS | {"online_tactical_replay"}
_CHECKPOINT_KEYS_V8 = _CHECKPOINT_KEYS_V7 | {"training_budget_history"}
_CHECKPOINT_KEYS_V9 = _CHECKPOINT_KEYS_V8 | {"promotion_league"}
_RUNTIME_KEYS_V4 = {"evaluation_workers"}
_RUNTIME_KEYS = _RUNTIME_KEYS_V4 | {
    "evaluation_backend", "evaluation_leaf_batch_size", "evaluation_reuse_tree",
}
_PROGRESS_KEYS_V5 = {"iteration", "self_play_games", "training_steps", "champion_version"}
_PROGRESS_KEYS_V6 = _PROGRESS_KEYS_V5 | {"tactical_training_steps"}
_PROGRESS_KEYS = _PROGRESS_KEYS_V6 | {"normal_training_steps"}


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


def _model_digest(model):
    """Identify the exact frozen actor independently of checkpoint ZIP metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_budget_history(history, iteration, configured_steps):
    """A segment starting at k applies to completed cycles k+1 and later."""
    if not isinstance(history, list) or not history:
        raise ValueError("Invalid training budget history")
    previous = -1
    for segment in history:
        if (not isinstance(segment, dict)
                or set(segment) != {"start_iteration", "train_steps_per_iteration"}
                or type(segment["start_iteration"]) is not int
                or not previous < segment["start_iteration"] <= iteration
                or type(segment["train_steps_per_iteration"]) is not int
                or segment["train_steps_per_iteration"] < 1):
            raise ValueError("Invalid training budget history")
        previous = segment["start_iteration"]
    if (history[0]["start_iteration"] != 0
            or history[-1]["train_steps_per_iteration"] != configured_steps):
        raise ValueError("Training budget history does not match configuration")
    return sum(((history[index + 1]["start_iteration"] if index + 1 < len(history)
                 else iteration) - segment["start_iteration"])
               * segment["train_steps_per_iteration"]
               for index, segment in enumerate(history))


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

    Self-play uses the configured champion or a cycle-frozen latest learner.
    The learner keeps its optimizer and
    continues training even when evaluation rejects promotion. A failed or
    interrupted iteration must be discarded by loading the last checkpoint.
    """

    def __init__(self, config=TrainingConfig(), model=None, device="cpu", evaluation_workers=1,
                 evaluation_backend="legacy", evaluation_leaf_batch_size=8,
                 evaluation_reuse_tree=True, promotion_league_state=None,
                 promotion_archive_allow_additions=False):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        if model is not None and not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet")
        if type(evaluation_workers) is not int or evaluation_workers < 1:
            raise ValueError("evaluation_workers must be a positive integer")
        if type(evaluation_backend) is not str or evaluation_backend not in ("legacy", "batched_cpp"):
            raise ValueError("evaluation_backend must be 'legacy' or 'batched_cpp'")
        if type(evaluation_leaf_batch_size) is not int or evaluation_leaf_batch_size < 1:
            raise ValueError("evaluation_leaf_batch_size must be a positive integer")
        if type(evaluation_reuse_tree) is not bool:
            raise ValueError("evaluation_reuse_tree must be a bool")
        self.config = config
        self.promotion_league = PromotionLeague(
            config.promotion_archive_dir, games=config.promotion_archive_games,
            seed=config.seed, state=promotion_league_state,
            allow_additions=promotion_archive_allow_additions)
        self.device = torch.device(device)
        # Parallel batching can change floating-point scheduling, so preserve
        # this operational setting in checkpoints even though it is not part of
        # TrainingConfig and can be explicitly overridden when loading.
        self.evaluation_workers = evaluation_workers
        self.evaluation_backend = evaluation_backend
        self.evaluation_leaf_batch_size = evaluation_leaf_batch_size
        self.evaluation_reuse_tree = evaluation_reuse_tree
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
        self.online_tactical_replay = CertifiedTacticalReplay(config.online_tactics_replay_capacity)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.iteration = self.self_play_games = self.training_steps = self.champion_version = 0
        self.tactical_training_steps = 0
        self.normal_training_steps = 0
        self.training_budget_history = [{"start_iteration": 0,
                                        "train_steps_per_iteration": config.train_steps_per_iteration}]
        self._at_boundary = True
        self._last_metrics = None
        if config.online_tactics:
            self._require_online_solver()

    @staticmethod
    def _require_online_solver():
        import my_board_engine as engine
        if not hasattr(engine, "solve_tactics"):
            raise RuntimeError("Online tactics requires the rebuilt solve_tactics binding")

    def _next_seed(self):
        return int(torch.randint(2 ** 63 - 1, (1,), generator=self.generator, device="cpu").item())

    def train_tactical_batch(self, batch):
        """Apply one extra teacher update without inventing a self-play cycle.

        The caller supplies engine-proved targets and, optionally, replay rows.
        A failed update cannot be saved; reload the last boundary checkpoint.
        The champion and historical iteration evaluation are not refreshed here.
        """
        if not self._at_boundary:
            raise RuntimeError("Cannot train tactics during an incomplete iteration")
        self._at_boundary = False
        losses = train_step(self.model, self.optimizer, batch)
        self.training_steps += 1
        self.tactical_training_steps += 1
        if self._last_metrics is not None:
            self._last_metrics["training_steps"] = self.training_steps
            self._last_metrics["tactical_training_steps"] = self.tactical_training_steps
            self._last_metrics["tactical_finetuning_since_evaluation"] = True
        self._at_boundary = True
        return losses

    def reconfigure(self, **overrides):
        """Change learning/search settings at a saved boundary without resetting state.

        Historical game counters and replay capacity remain fixed. Update budget
        changes apply only to future cycles and are recorded in a segment ledger.
        Architecture, RNG, replay contents,
        optimizer moments and progress are preserved. Call save_checkpoint() on
        a NEW path to persist the changed settings before continuing.
        """
        if not self._at_boundary:
            raise RuntimeError("Cannot reconfigure an incomplete iteration")
        immutable = {"games_per_iteration", "replay_capacity", "seed"}
        if immutable.intersection(overrides):
            raise ValueError("Reconfiguration cannot change games, replay capacity or seed")
        try:
            config = replace(self.config, **overrides)
        except TypeError as error:
            raise ValueError("Unknown training configuration field") from error
        if config.self_play_backend == "cuda" and self.device.type != "cuda":
            raise ValueError("CUDA self-play requires a CUDA Trainer device")
        if config.online_tactics:
            self._require_online_solver()
        if (config.online_tactics_replay_capacity != self.online_tactical_replay.capacity
                and len(self.online_tactical_replay)):
            raise ValueError("Cannot resize a nonempty certified tactical replay")
        league = self.promotion_league
        if _PROMOTION_CONFIG_KEYS.intersection(overrides):
            league = PromotionLeague(config.promotion_archive_dir,
                                     games=config.promotion_archive_games, seed=config.seed)
        # Validation above completes before changing any owned state.
        if config.train_steps_per_iteration != self.config.train_steps_per_iteration:
            segment = {"start_iteration": self.iteration,
                       "train_steps_per_iteration": config.train_steps_per_iteration}
            if self.training_budget_history[-1]["start_iteration"] == self.iteration:
                self.training_budget_history[-1] = segment
            else:
                self.training_budget_history.append(segment)
        if config.online_tactics_replay_capacity != self.online_tactical_replay.capacity:
            self.online_tactical_replay = CertifiedTacticalReplay(config.online_tactics_replay_capacity)
        self.config = config
        self.promotion_league = league
        for group in self.optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["weight_decay"] = config.weight_decay

    def _collect_games(self):
        config = self.config
        # No optimizer update occurs during collection. Own a frozen learner copy
        # across ALL chunks/games; never let rejected promotion revert this actor.
        actor = (deepcopy(self.model).eval() if config.self_play_model == "learner"
                 else self.champion)
        schedule_options = ({} if config.temperature_moves is None else {
            "temperature_moves": config.temperature_moves,
            "final_temperature": config.final_temperature,
        })
        if config.online_tactics:
            schedule_options["record_history"] = True
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
                    actor, count, options=options, temperature=config.temperature,
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
                yield collect_puct_game(actor, options=options,
                                        temperature=config.temperature, seed=seed,
                                        **schedule_options)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_iteration(self):
        if not self._at_boundary:
            raise RuntimeError("Reload the last checkpoint after an incomplete iteration")
        self.promotion_league.check_unchanged()
        self._at_boundary = False
        self._synchronize()
        started = perf_counter()
        config = self.config
        actor_metadata = {"role": config.self_play_model, "iteration": self.iteration,
                          "training_steps": self.training_steps,
                          "champion_version": self.champion_version,
                          "weights_sha256": _model_digest(
                              self.model if config.self_play_model == "learner" else self.champion)}
        endings = {}
        winners = {"Black": 0, "White": 0}
        game_plies = []
        plies = samples = 0
        replay_store_seconds = 0.0
        pending_samples = []
        pending_games = 0
        miner = None
        if config.online_tactics:
            from .online_tactics import TacticalPositionMiner
            miner = TacticalPositionMiner(config.online_tactics_max_cases,
                                          generator=self.generator, iteration=self.iteration + 1)

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
            if miner is not None:
                miner.add_game(data, self.self_play_games + 1)
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
        online_report = {"enabled": config.online_tactics, "seconds": 0.0,
                         "added_samples": 0, "excluded_shallow": 0,
                         "replay_size": len(self.online_tactical_replay)}
        if miner is not None:
            import my_board_engine as engine
            from .tactical_training import collect_certified_samples
            tactical_started = perf_counter()
            cases = miner.cases()
            options = engine.TacticalSolverOptions(
                max_depth=config.online_tactics_max_depth,
                max_nodes=config.online_tactics_max_nodes,
                time_limit_ms=config.online_tactics_time_limit_ms)
            proved, generation = collect_certified_samples(
                cases, options, max_cases=config.online_tactics_max_cases,
                generation_seconds=config.online_tactics_generation_seconds,
                bound_remaining_time=True,
                **({"include_loss": True} if config.online_tactics_include_loss else {}))
            eligible = [replace(row, proof={**row.proof, "self_play_actor": deepcopy(actor_metadata)})
                        for row in proved
                        if row.proof["proof_depth"] >= config.online_tactics_min_proof_depth]
            eligible_ids = {row.case_id for row in eligible}
            for record in generation["records"]:
                if record["training_label"] and record["id"] not in eligible_ids:
                    record["training_label"] = False
                    record["exclusion_reason"] = "proof_shorter_than_minimum_depth"
            self.online_tactical_replay.extend(eligible)
            online_report.update({**generation, "sampled_games": miner.sampled_games,
                                  "candidate_count": miner.candidate_count,
                                  "added_samples": len(eligible),
                                  "excluded_shallow": len(proved) - len(eligible),
                                  "replay_size": len(self.online_tactical_replay),
                                  "seconds": perf_counter() - tactical_started,
                                  "max_depth": config.online_tactics_max_depth,
                                  "max_nodes": config.online_tactics_max_nodes,
                                  "time_limit_ms": config.online_tactics_time_limit_ms,
                                  "generation_budget_seconds": config.online_tactics_generation_seconds,
                                  "min_proof_depth": config.online_tactics_min_proof_depth})
            online_report["include_loss"] = config.online_tactics_include_loss
            online_report["actor"] = actor_metadata
        training_started = perf_counter()
        loss_totals = dict.fromkeys(("loss", "policy_loss", "value_loss"), 0.0)
        category_rows = {
            "teacher_win_value_loss": "teacher_win_rows",
            "teacher_loss_value_loss": "teacher_loss_rows",
            **{f"{source}_{color}_value_loss": f"{source}_{color}_rows"
               for source in ("normal", "teacher") for color in ("black", "white")},
        }
        weighted_losses, drawn_rows = {}, {}
        tactical_rows = (max(1, min(config.batch_size - 1,
                                   round(config.batch_size * config.online_tactics_fraction)))
                         if config.online_tactics and len(self.online_tactical_replay) else 0)
        for _ in range(config.train_steps_per_iteration):
            batch = self.replay.sample(config.batch_size - tactical_rows, generator=self.generator,
                                       device=self.device)
            if tactical_rows:
                from .training import augment_proof_batch, train_mixed_step
                teacher = self.online_tactical_replay.sample(
                    tactical_rows, generator=self.generator, device=self.device)
            if config.augment_symmetries:
                batch = augment_batch(batch, generator=self.generator)
                if tactical_rows:
                    teacher = augment_proof_batch(teacher, generator=self.generator)
            if tactical_rows:
                losses = train_mixed_step(self.model, self.optimizer, batch, teacher)
            else:
                losses = train_step(self.model, self.optimizer, batch)
                losses.update(normal_policy_loss=losses["policy_loss"],
                              normal_value_loss=losses["value_loss"])
            for key, value in losses.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + value
                if key.endswith("_rows"):
                    drawn_rows[key] = drawn_rows.get(key, 0) + value
            for key, count_key in category_rows.items():
                if key in losses:
                    weighted_losses[key] = weighted_losses.get(key, 0.0) + losses[key] * losses[count_key]
            self.training_steps += 1
            self.normal_training_steps += 1
        self._synchronize()
        training_seconds = perf_counter() - training_started
        online_report.update({"mixed_updates": config.train_steps_per_iteration if tactical_rows else 0,
                              "tactical_rows_per_batch": tactical_rows,
                              "replay_rows_per_batch": config.batch_size - tactical_rows,
                              "actual_tactical_fraction": tactical_rows / config.batch_size,
                              "normal_training_draws": (config.batch_size - tactical_rows)
                                  * config.train_steps_per_iteration,
                              "tactical_training_draws": tactical_rows * config.train_steps_per_iteration})
        evaluation_started = perf_counter()
        evaluation_options = ({"tactical_checks": True} if config.self_play_tactical_checks else {})
        evaluation_diagnostics = {}
        if self.evaluation_backend == "batched_cpp":
            evaluation_options.update(
                backend=self.evaluation_backend, leaf_batch_size=self.evaluation_leaf_batch_size,
                reuse_tree=self.evaluation_reuse_tree, diagnostics=evaluation_diagnostics,
            )
        evaluation = evaluate_models(
            self.model, self.champion, games=config.evaluation_games,
            simulations=config.evaluation_simulations, c_puct=config.c_puct,
            seed=self._next_seed(), opening_moves=config.evaluation_opening_moves,
            opening_temperature=config.evaluation_opening_temperature,
            workers=self.evaluation_workers,
            **evaluation_options,
        )
        gate_passed = evaluation.win_rate >= config.promotion_threshold
        promotion_report = self.promotion_league.skipped_report(gate_passed)
        promoted = gate_passed
        if gate_passed and config.promotion_archive_dir is not None:
            promotion_report = self.promotion_league.compare(
                self.model, self.champion, evaluator=evaluate_models,
                simulations=config.evaluation_simulations, c_puct=config.c_puct,
                opening_moves=config.evaluation_opening_moves,
                opening_temperature=config.evaluation_opening_temperature,
                workers=self.evaluation_workers, backend=self.evaluation_backend,
                leaf_batch_size=self.evaluation_leaf_batch_size,
                reuse_tree=self.evaluation_reuse_tree,
                tactical_checks=config.self_play_tactical_checks)
            promoted = promotion_report["passed"]
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
            "metrics_schema_version": 3,
            "iteration": self.iteration, "self_play_games": self.self_play_games,
            "training_steps": self.training_steps, "champion_version": self.champion_version,
            "tactical_training_steps": self.tactical_training_steps,
            "normal_training_steps": self.normal_training_steps,
            "self_play_model": config.self_play_model,
            "self_play_actor": actor_metadata,
            "online_tactics": online_report,
            "online_tactics_seconds": online_report["seconds"],
            "generated_samples": samples, "replay_size": len(self.replay),
            "mean_self_play_plies": plies / config.games_per_iteration,
            "p95_self_play_plies": p95_plies,
            "max_self_play_plies": sorted_plies[-1],
            "self_play_endings": endings,
            "self_play_winners": winners,
            **{key: value / config.train_steps_per_iteration for key, value in loss_totals.items()},
            **{key: value / drawn_rows[category_rows[key]]
               if drawn_rows[category_rows[key]] else 0.0
               for key, value in weighted_losses.items()},
            "training_draw_counts": drawn_rows,
            "evaluation": {**asdict(evaluation), "win_rate": evaluation.win_rate},
            "promotion_threshold": config.promotion_threshold,
            "promotion_league": promotion_report,
            "promoted": promoted, "self_play_seconds": self_play_seconds,
            "self_play_compute_seconds": self_play_compute_seconds,
            "replay_store_seconds": replay_store_seconds,
            "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
            "elapsed_seconds": elapsed_seconds,
            "evaluation_workers": self.evaluation_workers,
            "evaluation_backend": self.evaluation_backend,
            "evaluation_leaf_batch_size": self.evaluation_leaf_batch_size,
            "evaluation_reuse_tree": self.evaluation_reuse_tree,
            "evaluation_diagnostics": evaluation_diagnostics,
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
            "normal_training_samples_drawn": (config.batch_size - tactical_rows)
                * config.train_steps_per_iteration,
            "tactical_training_samples_drawn": tactical_rows * config.train_steps_per_iteration,
            "normal_training_draws_per_generated_sample": (config.batch_size - tactical_rows)
                * config.train_steps_per_iteration / samples,
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
            "online_tactical_replay": self.online_tactical_replay.state_dict(),
            "training_budget_history": deepcopy(self.training_budget_history),
            "promotion_league": self.promotion_league.state_dict(),
            "config": asdict(self.config), "model_config": self.model.model_config,
            "model": _model_state(self.model), "champion": _model_state(self.champion),
            "optimizer": _cpu_copy(self.optimizer.state_dict()), "replay": self.replay.state_dict(),
            "progress": {name: getattr(self, name) for name in _PROGRESS_KEYS},
            "generator_state": self.generator.get_state().clone(),
            "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
                    "torch_cuda": ([state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                                   if torch.cuda.is_available() else [])},
            "runtime": {name: getattr(self, name) for name in _RUNTIME_KEYS},
            "last_metrics": deepcopy(self._last_metrics),
        }
        _atomic_write(path, lambda temporary: torch.save(payload, temporary))

    @classmethod
    def load_checkpoint(cls, path, device="cpu", evaluation_workers=None,
                        evaluation_backend=None, evaluation_leaf_batch_size=None,
                        evaluation_reuse_tree=None, refresh_promotion_archive=False):
        if type(refresh_promotion_archive) is not bool:
            raise ValueError("refresh_promotion_archive must be a bool")
        if (evaluation_workers is not None
                and (type(evaluation_workers) is not int or evaluation_workers < 1)):
            raise ValueError("evaluation_workers must be None or a positive integer")
        if evaluation_backend is not None and (
                type(evaluation_backend) is not str or evaluation_backend not in ("legacy", "batched_cpp")):
            raise ValueError("evaluation_backend must be None, 'legacy' or 'batched_cpp'")
        if evaluation_leaf_batch_size is not None and (
                type(evaluation_leaf_batch_size) is not int or evaluation_leaf_batch_size < 1):
            raise ValueError("evaluation_leaf_batch_size must be None or a positive integer")
        if evaluation_reuse_tree is not None and type(evaluation_reuse_tree) is not bool:
            raise ValueError("evaluation_reuse_tree must be None or a bool")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Unexpected training checkpoint fields")
        version = payload.get("checkpoint_version")
        if type(version) is not int or version not in range(1, _CHECKPOINT_VERSION + 1):
            raise ValueError("Unsupported training checkpoint version")
        expected_keys = (_CHECKPOINT_KEYS_V9 if version >= 9 else _CHECKPOINT_KEYS_V8 if version >= 8
                         else _CHECKPOINT_KEYS_V7 if version >= 7
                         else _CHECKPOINT_KEYS if version >= 4
                         else _CHECKPOINT_KEYS - {"runtime"})
        if set(payload) != expected_keys:
            raise ValueError("Unexpected training checkpoint fields")
        if not _same_primitive(payload["schema"], _SCHEMA):
            raise ValueError("Training checkpoint game/input schema does not match")
        raw_config = payload["config"]
        config_keys = {field.name for field in fields(TrainingConfig)}
        if version < 9:
            config_keys -= _PROMOTION_CONFIG_KEYS
        if version < 8:
            config_keys -= _ADAPTIVE_CONFIG_KEYS
        if version < 7:
            config_keys -= _ONLINE_CONFIG_KEYS
        if version < 3:
            config_keys -= _STRENGTH_CONFIG_KEYS
        if version == 1:
            config_keys -= _GPU_CONFIG_KEYS
        if not isinstance(raw_config, dict) or set(raw_config) != config_keys:
            raise ValueError("Invalid checkpoint training configuration")
        config = TrainingConfig(**raw_config)
        if refresh_promotion_archive and (version < 9 or config.promotion_archive_dir is None):
            raise ValueError("Archive refresh requires a checkpoint with an enabled pinned promotion archive")
        progress = payload["progress"]
        expected_progress = (_PROGRESS_KEYS if version >= 8 else _PROGRESS_KEYS_V6 if version >= 6
                             else _PROGRESS_KEYS_V5)
        if (not isinstance(progress, dict) or set(progress) != expected_progress
                or any(type(value) is not int or value < 0 for value in progress.values())
                or progress["self_play_games"] != progress["iteration"] * config.games_per_iteration
                or progress["champion_version"] > progress["iteration"]):
            raise ValueError("Checkpoint progress counters are inconsistent")
        budget_history = (payload["training_budget_history"] if version >= 8 else
                          [{"start_iteration": 0,
                            "train_steps_per_iteration": config.train_steps_per_iteration}])
        normal_steps = _validate_budget_history(budget_history, progress["iteration"],
                                                config.train_steps_per_iteration)
        if (progress.get("normal_training_steps", normal_steps) != normal_steps
                or progress["training_steps"] != normal_steps + progress.get("tactical_training_steps", 0)):
            raise ValueError("Checkpoint progress counters are inconsistent with training budget history")
        model = _restore_model(payload["model_config"], payload["model"])
        champion = _restore_model(payload["model_config"], payload["champion"])
        replay = ReplayBuffer.from_state_dict(payload["replay"])
        teacher_replay = (CertifiedTacticalReplay.from_state_dict(payload["online_tactical_replay"])
                          if version >= 7 else CertifiedTacticalReplay(config.online_tactics_replay_capacity))
        if teacher_replay.capacity != config.online_tactics_replay_capacity:
            raise ValueError("Certified replay capacity does not match the configuration")
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
        saved_runtime = {
            "evaluation_workers": 1, "evaluation_backend": "legacy",
            "evaluation_leaf_batch_size": 8, "evaluation_reuse_tree": True,
        }
        if version >= 4:
            runtime = payload["runtime"]
            expected_runtime = _RUNTIME_KEYS_V4 if version == 4 else _RUNTIME_KEYS
            if (not isinstance(runtime, dict) or set(runtime) != expected_runtime
                    or type(runtime["evaluation_workers"]) is not int
                    or runtime["evaluation_workers"] < 1):
                raise ValueError("Invalid checkpoint runtime settings")
            if version >= 5 and (
                    type(runtime["evaluation_backend"]) is not str
                    or runtime["evaluation_backend"] not in ("legacy", "batched_cpp")
                    or type(runtime["evaluation_leaf_batch_size"]) is not int
                    or runtime["evaluation_leaf_batch_size"] < 1
                    or type(runtime["evaluation_reuse_tree"]) is not bool):
                raise ValueError("Invalid checkpoint runtime settings")
            saved_runtime.update(runtime)
        requested_runtime = {
            "evaluation_workers": evaluation_workers, "evaluation_backend": evaluation_backend,
            "evaluation_leaf_batch_size": evaluation_leaf_batch_size,
            "evaluation_reuse_tree": evaluation_reuse_tree,
        }
        saved_runtime.update({name: value for name, value in requested_runtime.items() if value is not None})
        trainer = cls(config, model=model, device=device, **saved_runtime,
                      promotion_league_state=payload["promotion_league"] if version >= 9 else None,
                      promotion_archive_allow_additions=refresh_promotion_archive)
        trainer.champion = champion.to(trainer.device).eval()
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.replay = replay
        trainer.online_tactical_replay = teacher_replay
        trainer.generator = private_generator
        if version < 6:
            progress = {**progress, "tactical_training_steps": 0}
            if last_metrics is not None:
                last_metrics = {**last_metrics, "tactical_training_steps": 0}
        if version < 8:
            progress = {**progress, "normal_training_steps": normal_steps}
            if last_metrics is not None:
                last_metrics = {**last_metrics, "normal_training_steps": normal_steps}
        trainer.training_budget_history = deepcopy(budget_history)
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

    def export_learner(self, path):
        """Portable latest candidate, independent of champion promotion."""
        if not self._at_boundary:
            raise RuntimeError("Export the learner only at a completed iteration boundary")
        _atomic_write(path, lambda temporary: save_model(self.model, temporary))

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
                from .metrics import compact_metric_row
                target = Path(metrics_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(compact_metric_row(metrics), ensure_ascii=False,
                                            allow_nan=False, separators=(",", ":")) + "\n")
                    handle.flush()
            if on_iteration is not None:
                on_iteration(deepcopy(metrics))
            if collect_metrics:
                output.append(metrics)
        return output
