"""Optional PyTorch policy/value connection for my_board_engine."""

from .encoding import (
    ACTION_SIZE, BOARD_SIZE, FEATURE_NAMES, FORMAT_VERSION, INPUT_CHANNELS,
    PASS_ACTION, EncodedState, action_to_move, encode_state, move_to_action,
    terminal_value, visit_policy,
)
from .model import PolicyValueNet, masked_policy
from .inference import NeuralAgent, Prediction
from .checkpoint import load_model, save_model
from .puct import PUCT, PUCTOptions, PUCTSearchResult, sample_visits
from .training import (
    GameData, Losses, TrainingBatch, TrainingSample, collect_mcts_game, collect_puct_game,
    make_batch, policy_value_loss, train_step,
)

__all__ = [
    "ACTION_SIZE", "BOARD_SIZE", "FEATURE_NAMES", "FORMAT_VERSION", "INPUT_CHANNELS",
    "PASS_ACTION", "EncodedState", "action_to_move", "encode_state", "move_to_action",
    "terminal_value", "visit_policy", "PolicyValueNet", "masked_policy", "NeuralAgent",
    "Prediction", "load_model", "save_model", "GameData", "Losses", "TrainingBatch",
    "TrainingSample", "collect_mcts_game", "collect_puct_game", "make_batch", "policy_value_loss", "train_step",
    "PUCT", "PUCTOptions", "PUCTSearchResult", "sample_visits",
]
