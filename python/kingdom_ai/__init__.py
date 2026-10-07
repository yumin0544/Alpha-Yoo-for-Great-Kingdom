"""Optional PyTorch policy/value connection for my_board_engine."""

from .encoding import (
    ACTION_SIZE, BOARD_SIZE, FEATURE_NAMES, FORMAT_VERSION, INPUT_CHANNELS,
    PASS_ACTION, EncodedState, action_to_move, encode_state, move_to_action,
    terminal_value, visit_policy,
)
from .model import PolicyValueNet, masked_policy
from .inference import NeuralAgent, Prediction
from .batching import BatchedEvaluator
from .encoded_batching import EncodedBatchedEvaluator
from .gpu_rules import GpuStateBatch
from .gpu_puct import GpuPUCT, GpuPUCTOptions, GpuSearchResult
from .gpu_training import collect_gpu_puct_games
from .checkpoint import load_model, save_model
from .puct import PUCT, PUCTOptions, PUCTSearchResult, sample_visits
from .training import (
    GameData, Losses, TrainingBatch, TrainingSample, collect_mcts_game, collect_puct_game,
    make_batch, policy_value_loss, train_step,
)
from .replay import ReplayBuffer
from .evaluation import EvaluationResult, evaluate_models
from .match import MatchOptions, play_match, series_ratings
from .loop import Trainer, TrainingConfig

__all__ = [
    "ACTION_SIZE", "BOARD_SIZE", "FEATURE_NAMES", "FORMAT_VERSION", "INPUT_CHANNELS",
    "PASS_ACTION", "EncodedState", "action_to_move", "encode_state", "move_to_action",
    "terminal_value", "visit_policy", "PolicyValueNet", "masked_policy", "NeuralAgent",
    "Prediction", "BatchedEvaluator", "EncodedBatchedEvaluator", "load_model", "save_model", "GameData", "Losses", "TrainingBatch",
    "GpuStateBatch", "GpuPUCT", "GpuPUCTOptions", "GpuSearchResult",
    "collect_gpu_puct_games",
    "TrainingSample", "collect_mcts_game", "collect_puct_game", "make_batch", "policy_value_loss", "train_step",
    "PUCT", "PUCTOptions", "PUCTSearchResult", "sample_visits",
    "ReplayBuffer", "EvaluationResult", "evaluate_models", "MatchOptions", "play_match",
    "series_ratings", "Trainer", "TrainingConfig",
]
