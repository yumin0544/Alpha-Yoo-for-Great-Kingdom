"""PyTorch evaluation for the C++ PUCT tree and visit-based move selection."""

import math

import torch
import my_board_engine as engine

from .encoding import ACTION_SIZE, _check_rules, action_to_move, move_to_action
from .inference import NeuralAgent

PUCTOptions = engine.PUCTOptions
PUCTSearchResult = engine.PUCTSearchResult


class PUCT:
    def __init__(self, model_or_agent, options=None, device=None):
        if isinstance(model_or_agent, NeuralAgent) and device is None:
            self.agent = model_or_agent
        else:
            model = model_or_agent.model if isinstance(model_or_agent, NeuralAgent) else model_or_agent
            self.agent = NeuralAgent(model, device=device)
        self._searcher = engine.PUCT(options if options is not None else PUCTOptions())

    @property
    def options(self):
        return self._searcher.options

    def _evaluate(self, state):
        prediction = self.agent.predict(state)
        return prediction.policy.tolist(), prediction.value

    def search(self, state):
        if not isinstance(state, engine.State):
            raise TypeError("Expected an engine State")
        snapshot = state.copy()
        _check_rules(snapshot)
        return self._searcher.search(snapshot, self._evaluate)


def sample_visits(result, temperature=1.0, generator=None):
    """Sample only visited root actions; temperature zero takes best_move."""
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("Temperature must be a real number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative")
    if result.best_move is None or result.simulations < 1:
        raise ValueError("A terminal or unsearched result has no move to sample")
    if temperature == 0:
        return result.best_move
    visits = [0] * ACTION_SIZE
    for item in result.moves:
        if item.visits < 0:
            raise ValueError("Visit counts cannot be negative")
        action = move_to_action(item.move)
        visits[action] = item.visits
    if max(visits) < 1:
        raise ValueError("A searched result must contain positive visit counts")
    largest_log = math.log(max(visits))
    weights = torch.tensor([
        math.exp((math.log(count) - largest_log) / temperature) if count > 0 else 0.0
        for count in visits
    ], dtype=torch.float64)
    action = int(torch.multinomial(weights, 1, generator=generator).item())
    return action_to_move(action)
