"""Replay actual self-play actions, including sampled non-argmax moves."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import my_board_engine as engine

from kingdom_ai.encoding import action_to_move, encode_state
from kingdom_ai.gpu_puct import GpuPUCTOptions
from kingdom_ai.gpu_rules import GPU_REASON, GpuStateBatch
from kingdom_ai.gpu_training import collect_gpu_puct_games
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.puct import PUCTOptions
from kingdom_ai.training import GameData, collect_puct_game


CAPTURE_ACTIONS = (0, 1, 10, 80, 2)


class ReplayAssertions:
    def replay_game(self, game):
        self.assertIsInstance(game.action_history, tuple)
        self.assertEqual(len(game.action_history), len(game.samples))
        state = engine.State()
        for action, sample in zip(game.action_history, game.samples):
            self.assertFalse(state.result.finished())
            self.assertIs(type(action), int)
            encoded = encode_state(state)
            self.assertEqual(sample.to_play, encoded.to_play)
            self.assertEqual(sample.value, 1. if sample.to_play == game.winner else -1.)
            torch.testing.assert_close(sample.features, encoded.features, rtol=0, atol=0)
            torch.testing.assert_close(sample.legal_mask, encoded.legal_mask, rtol=0, atol=0)
            self.assertTrue(state.play(action_to_move(action)).accepted())
        self.assertTrue(state.result.finished())
        self.assertEqual((state.result.winner, state.result.reason), (game.winner, game.reason))
        return state

    def assert_games_equal(self, first, second):
        self.assertEqual((first.winner, first.reason, len(first.samples)),
                         (second.winner, second.reason, len(second.samples)))
        for left, right in zip(first.samples, second.samples):
            self.assertEqual((left.value, left.to_play), (right.value, right.to_play))
            for field in ("features", "legal_mask", "policy"):
                torch.testing.assert_close(getattr(left, field), getattr(right, field),
                                           rtol=0, atol=0)


class CpuHistoryTest(ReplayAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_game_data_three_positional_fields_remain_compatible(self):
        self.assertIsNone(GameData([], engine.Cell.White,
                                   engine.EndReason.TwoPasses).action_history)

    def test_record_history_requires_bool_before_search_setup(self):
        for invalid in (0, 1, None, "True"):
            with self.subTest(value=invalid), patch("kingdom_ai.puct.PUCT") as searcher:
                with self.assertRaisesRegex(TypeError, "record_history"):
                    collect_puct_game(None, record_history=invalid)
                searcher.assert_not_called()
            with self.subTest(gpu_value=invalid), patch("kingdom_ai.gpu_training.GpuPUCT") as searcher:
                with self.assertRaisesRegex(TypeError, "record_history"):
                    collect_gpu_puct_games(None, 0, record_history=invalid)
                searcher.assert_not_called()

    def collect_script(self, actions, record_history=True):
        class ScriptSearcher:
            def __init__(self, model, options):
                self.step = 0

            def search(self, state):
                action = actions[self.step]
                self.step += 1
                moves = [SimpleNamespace(move=engine.Move.pass_turn(), visits=9)]
                if action != 81:
                    moves.append(SimpleNamespace(move=action_to_move(action), visits=1))
                return SimpleNamespace(moves=moves, simulations=sum(move.visits for move in moves),
                                       best_move=engine.Move.pass_turn())

        selected = iter(actions)

        def sample(weights, count, *, generator):
            action = next(selected)
            self.assertGreater(weights[action].item(), 0)
            return torch.tensor([action], dtype=torch.int64)

        with patch("kingdom_ai.puct.PUCT", ScriptSearcher), \
                patch("kingdom_ai.puct.torch.multinomial", side_effect=sample):
            return collect_puct_game(None, record_history=record_history)

    def test_sampled_capture_is_not_inferred_from_policy_argmax(self):
        game = self.collect_script(CAPTURE_ACTIONS)
        self.assertEqual(game.action_history, CAPTURE_ACTIONS)
        self.assertTrue(all(int(sample.policy.argmax()) == 81 for sample in game.samples))
        state = self.replay_game(game)
        self.assertEqual(game.reason, engine.EndReason.Capture)
        self.assertEqual(game.winner, engine.Cell.Black)
        self.assertEqual((state.remaining_stones(engine.Cell.Black),
                          state.remaining_stones(engine.Cell.White)), (38, 39))
        self.assertEqual(state.board.cells[1], engine.Cell.Empty)
        disabled = self.collect_script(CAPTURE_ACTIONS, record_history=False)
        self.assertIsNone(disabled.action_history)
        self.assert_games_equal(game, disabled)

    def test_double_pass_history_and_stocks(self):
        game = self.collect_script((81, 81))
        state = self.replay_game(game)
        self.assertEqual(game.reason, engine.EndReason.TwoPasses)
        self.assertEqual(game.winner, engine.Cell.White)
        self.assertEqual((state.remaining_stones(engine.Cell.Black),
                          state.remaining_stones(engine.Cell.White)), (41, 41))
        self.assertEqual(state.consecutive_passes, 2)

    def test_real_puct_history_is_optional_and_preserves_seeded_samples(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        kwargs = dict(options=PUCTOptions(simulations=4, seed=73, dirichlet_epsilon=.25),
                      seed=73, temperature=1., temperature_moves=6)
        before = torch.get_rng_state().clone()
        plain = collect_puct_game(model, **kwargs)
        recorded = collect_puct_game(model, **kwargs, record_history=True)
        self.assertIsNone(plain.action_history)
        self.assert_games_equal(plain, recorded)
        self.replay_game(recorded)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
class GpuHistoryTest(ReplayAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cuda_sampling_keeps_exact_actions_and_lane_lengths(self):
        instances = []
        batches = []
        initial = GpuStateBatch.initial

        class ScriptSearcher:
            def __init__(self, model, options, device):
                self.device = torch.device("cuda", torch.cuda.current_device())
                self.step = 0
                instances.append(self)

            def search(self, state):
                alive = state.states[:, GPU_REASON] == 0
                desired = torch.full((len(state),), 81, dtype=torch.int64, device=self.device)
                desired[0] = CAPTURE_ACTIONS[self.step % len(CAPTURE_ACTIONS)]
                self.step += 1
                self.selected = torch.where(alive, desired, -1)
                visits = torch.zeros((len(state), 82), dtype=torch.int32, device=self.device)
                visits[:, 81] = alive.to(torch.int32) * 9
                visits.scatter_add_(1, desired[:, None], alive.to(torch.int32)[:, None])
                return SimpleNamespace(actions=torch.where(alive, 81, -1), visits=visits,
                                       policy=visits.float() / 10.)

        def choose_actions(weights, count, *, generator):
            selected = instances[-1].selected
            # Inactive lanes use a dummy pass for multinomial, discarded by the collector.
            return selected.clamp_min(0)[:, None]

        def retain_batch(*args, **kwargs):
            batch = initial(*args, **kwargs)
            batches.append(batch)
            return batch

        with patch("kingdom_ai.gpu_training.GpuPUCT", ScriptSearcher), \
                patch("kingdom_ai.gpu_training.torch.multinomial", side_effect=choose_actions), \
                patch("kingdom_ai.gpu_training.GpuStateBatch.initial", side_effect=retain_batch):
            games = collect_gpu_puct_games(None, 3, batch_size=2, record_history=True)
        self.assertEqual([game.action_history for game in games],
                         [CAPTURE_ACTIONS, (81, 81), CAPTURE_ACTIONS])
        self.assertEqual([len(game.samples) for game in games], [5, 2, 5])
        self.assertTrue(all(int(sample.policy.argmax()) == 81
                            for game in games for sample in game.samples))
        records = [record for batch in batches for record in batch.snapshot()]
        for game, record in zip(games, records):
            state = self.replay_game(game)
            self.assertEqual(record["cells"], [int(cell) for cell in state.board.cells])
            self.assertEqual(record["ownership"], [int(owner) for owner in state.ownership])
            self.assertEqual(record["black_stock"], state.remaining_stones(engine.Cell.Black))
            self.assertEqual(record["white_stock"], state.remaining_stones(engine.Cell.White))
            self.assertEqual(record["passes"], state.consecutive_passes)

    def test_actual_cuda_puct_recording_preserves_seeded_games_and_rng(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        kwargs = dict(games=3, batch_size=2, seed=92, temperature=1., temperature_moves=3,
                      options=GpuPUCTOptions(simulations=3, seed=92))
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        plain = collect_gpu_puct_games(model, **kwargs)
        recorded = collect_gpu_puct_games(model, **kwargs, record_history=True)
        for left, right in zip(plain, recorded):
            self.assertIsNone(left.action_history)
            self.assert_games_equal(left, right)
            self.replay_game(right)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
