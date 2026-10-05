"""GPU completed-game targets, temperature sampling and CPU schema parity."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

from kingdom_ai.encoding import action_to_move, encode_state
from kingdom_ai.gpu_puct import GpuPUCTOptions
from kingdom_ai.gpu_rules import GPU_REASON, GpuStateBatch
from kingdom_ai.gpu_training import collect_gpu_puct_games, _visit_sampling_weights
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.replay import ReplayBuffer


class GpuTrainingValidationTest(unittest.TestCase):
    def test_invalid_arguments_are_rejected_before_cuda_setup(self):
        invalid = (
            {"games": -1}, {"games": True}, {"batch_size": 0},
            {"batch_size": 1.5}, {"seed": -1}, {"seed": 2**64},
            {"seed": True}, {"temperature": float("nan")},
            {"temperature": float("inf")}, {"temperature": -1},
            {"temperature": True}, {"temperature": "1"}, {"options": object()},
        )
        for kwargs in invalid:
            arguments = {"games": 1, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                collect_gpu_puct_games(None, **arguments)
        self.assertEqual(collect_gpu_puct_games(None, 0), [])

    def test_extreme_positive_temperatures_keep_only_visited_actions(self):
        visits = torch.tensor([[0, 1, 1_000_000_000], [0, 4, 4]], dtype=torch.int32)
        for temperature in (5e-324, 1e-300, 1.0, 1e300):
            weights = _visit_sampling_weights(visits, temperature)
            with self.subTest(temperature=temperature):
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue((weights[:, 0] == 0).all())
                self.assertTrue((weights.sum(dim=1) > 0).all())
                self.assertEqual(weights[0, 2].item(), 1.0)
                self.assertEqual(weights[1, 1:].tolist(), [1.0, 1.0])


class PassSearcher:
    """Fixture whose best move passes but whose raw visits have two actions."""

    def __init__(self, model, options, device):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

    def search(self, state):
        batch = len(state)
        alive = state.states[:, GPU_REASON] == 0
        visits = torch.zeros((batch, 82), dtype=torch.int32, device=self.device)
        visits[:, 0] = alive.to(torch.int32)
        visits[:, 81] = alive.to(torch.int32)
        actions = torch.where(alive, 81, -1).to(torch.int64)
        return SimpleNamespace(actions=actions, visits=visits, policy=visits.float() / 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
class GpuTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def assert_same_games(self, first, second):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertEqual((a.winner, a.reason, len(a.samples)),
                             (b.winner, b.reason, len(b.samples)))
            for x, y in zip(a.samples, b.samples):
                self.assertEqual((x.to_play, x.value), (y.to_play, y.value))
                for field in ("features", "legal_mask", "policy"):
                    torch.testing.assert_close(getattr(x, field), getattr(y, field),
                                               rtol=0, atol=0)

    def test_single_visit_games_match_every_cpu_position_and_final_outcome(self):
        data = collect_gpu_puct_games(
            self.make_model(), 5, batch_size=3, seed=628,
            options=GpuPUCTOptions(simulations=1, seed=628),
        )
        self.assertEqual(len(data), 5)
        replay = ReplayBuffer(512)
        for game in data:
            oracle = engine.State()
            self.assertGreater(len(game.samples), 1)
            for sample in game.samples:
                self.assertFalse(oracle.result.finished())
                encoded = encode_state(oracle)
                self.assertEqual(sample.to_play, oracle.to_play)
                self.assertEqual(sample.value, 1.0 if sample.to_play == game.winner else -1.0)
                torch.testing.assert_close(sample.features, encoded.features, rtol=0, atol=0)
                torch.testing.assert_close(sample.legal_mask, encoded.legal_mask, rtol=0, atol=0)
                self.assertEqual(sample.policy.sum().item(), 1.0)
                self.assertEqual((sample.policy > 0).sum().item(), 1)
                for field in (sample.features, sample.legal_mask, sample.policy):
                    self.assertEqual(field.device.type, "cpu")
                    self.assertFalse(field.is_inference())
                self.assertTrue(oracle.play(action_to_move(int(sample.policy.argmax()))).accepted())
            self.assertTrue(oracle.result.finished())
            self.assertEqual((game.winner, game.reason), (oracle.result.winner, oracle.result.reason))
            replay.extend(game.samples)
        self.assertGreater(len(replay), 0)

    def test_temperature_zero_keeps_raw_targets_and_pre_terminal_perspective(self):
        with patch("kingdom_ai.gpu_training.GpuPUCT", PassSearcher):
            games = collect_gpu_puct_games(self.make_model(), 2, temperature=0, batch_size=2)
        for game in games:
            self.assertEqual(game.winner, engine.Cell.White)
            self.assertEqual(game.reason, engine.EndReason.TwoPasses)
            self.assertEqual(len(game.samples), 2)
            self.assertEqual([sample.to_play for sample in game.samples],
                             [engine.Cell.Black, engine.Cell.White])
            self.assertEqual([sample.value for sample in game.samples], [-1.0, 1.0])
            for sample in game.samples:
                self.assertEqual(sample.policy[0].item(), 0.5)
                self.assertEqual(sample.policy[81].item(), 0.5)
            self.assertEqual(game.samples[1].features[8, 0, 0].item(), 0.5)

    def test_finished_lanes_append_no_samples_and_lane_order_is_preserved(self):
        terminal = engine.State()
        terminal.pass_turn()
        terminal.pass_turn()
        one_pass = engine.State()
        one_pass.pass_turn()
        inputs = [terminal, one_pass, engine.State()]
        batch = GpuStateBatch.from_engine(inputs)
        with patch("kingdom_ai.gpu_training.GpuPUCT", PassSearcher), \
                patch("kingdom_ai.gpu_training.GpuStateBatch.initial", return_value=batch):
            games = collect_gpu_puct_games(self.make_model(), 3, temperature=0, batch_size=3)
        self.assertEqual([len(game.samples) for game in games], [0, 1, 2])
        self.assertEqual(games[1].samples[0].to_play, engine.Cell.White)
        self.assertTrue(all(game.reason == engine.EndReason.TwoPasses for game in games))

    def test_seeded_noise_sampling_repeats_and_preserves_model_and_global_rngs(self):
        model = self.make_model()
        model.train()
        model.trunk[0].eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        modes = [module.training for module in model.modules()]
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        cpu_rng = torch.get_rng_state().clone()
        cuda_rngs = [state.clone() for state in torch.cuda.get_rng_state_all()]
        kwargs = dict(games=3, batch_size=2, seed=731, temperature=1.0,
                      options=GpuPUCTOptions(simulations=4, seed=731))
        first = collect_gpu_puct_games(model, **kwargs)
        second = collect_gpu_puct_games(model, **kwargs)
        self.assert_same_games(first, second)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rngs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, actual in model.state_dict().items():
            self.assertEqual(actual.device.type, "cpu")
            torch.testing.assert_close(actual, weights[name], rtol=0, atol=0)
        for parameter, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
