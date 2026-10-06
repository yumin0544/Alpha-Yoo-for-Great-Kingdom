"""GPU self-play integration, replay learning, and completed-boundary resume."""

from dataclasses import asdict, replace
import copy
import importlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai import ACTION_SIZE, PASS_ACTION, GameData, PolicyValueNet, TrainingSample, encode_state
from kingdom_ai.evaluation import EvaluationResult
from kingdom_ai.gpu_puct import GpuPUCTOptions
from kingdom_ai.loop import Trainer, TrainingConfig


loop = importlib.import_module(Trainer.__module__)


def small_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(127)
        return PolicyValueNet(channels=8, residual_blocks=1)


def small_config(**changes):
    values = dict(
        games_per_iteration=2, simulations=2, dirichlet_epsilon=0.25,
        self_play_backend="cuda", self_play_batch_size=2, temperature=1.0,
        replay_capacity=256, batch_size=4, train_steps_per_iteration=1,
        evaluation_games=2, evaluation_simulations=2,
        evaluation_opening_moves=2, seed=914,
    )
    values.update(changes)
    return TrainingConfig(**values)


def pass_game():
    game = engine.State()
    positions = []
    while not game.result.finished():
        encoded = encode_state(game)
        policy = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        policy[PASS_ACTION] = 1
        positions.append((encoded, policy))
        game.pass_turn()
    return GameData([
        TrainingSample(encoded.features, encoded.legal_mask, policy,
                       1.0 if encoded.to_play == game.result.winner else -1.0,
                       encoded.to_play)
        for encoded, policy in positions
    ], game.result.winner, game.result.reason)


def evaluation(wins):
    return EvaluationResult(games=2, wins=wins, losses=2 - wins,
                            wins_as_black=int(wins == 2), wins_as_white=int(wins >= 1),
                            total_plies=4, endings={"TwoPasses": 2})


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


class EqualityMixin:
    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.device, right.device)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                with self.subTest(key=key):
                    self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def trainer_state(self, trainer):
        return cpu_tree({
            "model": trainer.model.state_dict(), "champion": trainer.champion.state_dict(),
            "optimizer": trainer.optimizer.state_dict(), "replay": trainer.replay.state_dict(),
            "generator": trainer.generator.get_state(), "iteration": trainer.iteration,
            "self_play_games": trainer.self_play_games, "training_steps": trainer.training_steps,
            "champion_version": trainer.champion_version,
        })


class GpuTrainingConfigurationTest(EqualityMixin, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="kingdom-gpu-config-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def test_backend_and_batch_validation_and_cuda_requires_cuda_device(self):
        defaults = TrainingConfig()
        self.assertEqual(defaults.self_play_backend, "cpu")
        self.assertEqual(defaults.self_play_batch_size, 128)
        for value in ("gpu", "CUDA", "", None, True, 1):
            with self.subTest(backend=value), self.assertRaises((ValueError, TypeError)):
                TrainingConfig(self_play_backend=value)
        for value in (0, -1, True, 1.5, "12"):
            with self.subTest(batch_size=value), self.assertRaises((ValueError, TypeError)):
                TrainingConfig(self_play_batch_size=value)
        with patch.object(loop, "collect_gpu_puct_games") as generate, \
                self.assertRaisesRegex(ValueError, "CUDA|cuda"):
            Trainer(config=small_config(), model=small_model(), device="cpu")
        generate.assert_not_called()

    def test_version_one_completed_checkpoint_restores_cpu_defaults(self):
        config = small_config(self_play_backend="cpu", self_play_batch_size=3)
        trainer = Trainer(config=config, model=small_model(), device="cpu")
        with patch.object(loop, "collect_puct_game", side_effect=lambda *a, **k: pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            trainer.run(1)
        source = self.path / "current.pt"
        trainer.save_checkpoint(source)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        self.assertEqual(payload["checkpoint_version"], 3)
        payload["checkpoint_version"] = 1
        payload["config"].pop("self_play_backend")
        payload["config"].pop("self_play_batch_size")
        for field in loop._STRENGTH_CONFIG_KEYS:
            payload["config"].pop(field)
        legacy = self.path / "legacy.pt"
        torch.save(payload, legacy)

        restored = Trainer.load_checkpoint(legacy, device="cpu")
        self.assertEqual(restored.config.self_play_backend, "cpu")
        self.assertEqual(restored.config.self_play_batch_size, 128)
        self.assert_tree_equal(self.trainer_state(trainer), self.trainer_state(restored))
        with patch.object(loop, "collect_puct_game", side_effect=lambda *a, **k: pass_game()), \
                patch.object(loop, "evaluate_models", return_value=evaluation(0)):
            restored.run(1)
        self.assertEqual(restored.iteration, 2)
        self.assertEqual(restored.self_play_games, 4)

    def test_checkpoint_versions_require_their_own_strict_config_fields(self):
        trainer = Trainer(config=small_config(self_play_backend="cpu"),
                          model=small_model(), device="cpu")
        target = self.path / "checkpoint.pt"
        trainer.save_checkpoint(target)
        original = torch.load(target, map_location="cpu", weights_only=True)
        variants = []
        for field in ("self_play_backend", "self_play_batch_size"):
            damaged = copy.deepcopy(original)
            damaged["config"].pop(field)
            variants.append(damaged)
        damaged = copy.deepcopy(original)
        damaged["checkpoint_version"] = 1
        variants.append(damaged)  # New settings must not be silently accepted as v1.
        damaged = copy.deepcopy(original)
        damaged["checkpoint_version"] = 4
        variants.append(damaged)
        for index, damaged in enumerate(variants):
            torch.save(damaged, target)
            with self.subTest(variant=index), self.assertRaises((ValueError, TypeError)):
                Trainer.load_checkpoint(target)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
class GpuTrainerIntegrationTest(EqualityMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        cls.previous_deterministic = torch.backends.cudnn.deterministic
        cls.previous_benchmark = torch.backends.cudnn.benchmark
        cls.previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        cls.previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        torch.backends.cudnn.deterministic = cls.previous_deterministic
        torch.backends.cudnn.benchmark = cls.previous_benchmark
        torch.backends.cudnn.allow_tf32 = cls.previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_matmul_tf32

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="kingdom-gpu-trainer-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.config = small_config()
        self.initial_python = random.getstate()
        self.initial_torch = torch.get_rng_state().clone()
        self.initial_cuda = [value.clone() for value in torch.cuda.get_rng_state_all()]
        self.addCleanup(random.setstate, self.initial_python)
        self.addCleanup(torch.set_rng_state, self.initial_torch)
        self.addCleanup(torch.cuda.set_rng_state_all, self.initial_cuda)

    def trainer(self):
        return Trainer(config=self.config, model=small_model(), device="cuda")

    def test_strength_options_actual_gpu_training_and_exact_resume(self):
        self.config = replace(self.config, augment_symmetries=True,
                              temperature_moves=2, final_temperature=0.25,
                              self_play_tactical_checks=True, self_play_fpu_reduction=0.0)
        self.test_actual_gpu_selfplay_training_evaluation_and_exact_resume()

    def test_actual_gpu_selfplay_training_evaluation_and_exact_resume(self):
        collect = loop.collect_gpu_puct_games
        update = loop.train_step
        generated = []
        batches = []

        def observe_games(*args, **kwargs):
            games = collect(*args, **kwargs)
            records = []
            for game in games:
                self.assertTrue(game.samples)
                self.assertIn(game.winner, (engine.Cell.Black, engine.Cell.White))
                self.assertNotEqual(game.reason, engine.EndReason.None_)
                records.append({"winner": game.winner, "reason": game.reason,
                                "samples": [cpu_tree(asdict(sample)) for sample in game.samples]})
                for sample in game.samples:
                    self.assertEqual(sample.features.device.type, "cpu")
                    self.assertEqual(sample.legal_mask.device.type, "cpu")
                    self.assertEqual(sample.policy.device.type, "cpu")
                    self.assertEqual(sample.value, 1.0 if sample.to_play == game.winner else -1.0)
                    self.assertTrue(torch.equal(sample.policy[~sample.legal_mask],
                                                torch.zeros_like(sample.policy[~sample.legal_mask])))
                    self.assertAlmostEqual(sample.policy.sum().item(), 1.0, places=6)
            generated.append(records)
            return games

        def observe_update(model, optimizer, batch):
            self.assertEqual(next(model.parameters()).device.type, "cuda")
            for tensor in (batch.features, batch.legal_mask, batch.policy, batch.value):
                self.assertEqual(tensor.device.type, "cuda")
            batches.append(cpu_tree(asdict(batch)))
            return update(model, optimizer, batch)

        with patch.object(loop, "collect_gpu_puct_games", side_effect=observe_games), \
                patch.object(loop, "train_step", side_effect=observe_update):
            uninterrupted = self.trainer()
            starting_model = cpu_tree(uninterrupted.model.state_dict())
            metrics = uninterrupted.run(2)
            expected = self.trainer_state(uninterrupted)
            expected_games, expected_batches = generated[:], batches[:]
            generated.clear()
            batches.clear()
            expected_python = [random.random() for _ in range(4)]
            expected_torch = torch.rand(4)
            expected_cuda = torch.rand(4, device="cuda").cpu()

            random.setstate(self.initial_python)
            torch.set_rng_state(self.initial_torch)
            torch.cuda.set_rng_state_all(self.initial_cuda)
            interrupted = self.trainer()
            checkpoint = self.path / "resume.pt"
            interrupted.run(1, checkpoint_path=checkpoint)
            # Loading must restore both training state and global RNG after unrelated use.
            random.random()
            torch.rand(7)
            torch.rand(7, device="cuda")
            resumed = Trainer.load_checkpoint(checkpoint, device="cuda")
            self.assertEqual(asdict(resumed.config), asdict(self.config))
            self.assert_tree_equal(self.trainer_state(interrupted), self.trainer_state(resumed))
            resumed.run(1)

        self.assertEqual(resumed.iteration, 2)
        self.assertEqual(resumed.self_play_games, 4)
        self.assertEqual(resumed.training_steps, 2)
        self.assert_tree_equal(expected_games, generated)
        self.assert_tree_equal(expected_batches, batches)
        self.assert_tree_equal(expected, self.trainer_state(resumed))
        self.assertTrue(any(not torch.equal(value, expected["model"][name])
                            for name, value in starting_model.items()))
        self.assertGreater(len(resumed.replay), 0)
        self.assertLessEqual(len(resumed.replay), self.config.replay_capacity)
        for value in resumed.replay.state_dict().values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
        self.assertEqual(metrics[-1]["evaluation"]["games"], 2)
        self.assertEqual(sum(metrics[-1]["self_play_endings"].values()), 2)
        self.assertEqual(metrics[-1]["generated_samples"],
                         sum(len(game["samples"]) for game in expected_games[-1]))
        self.assertEqual(expected_python, [random.random() for _ in range(4)])
        torch.testing.assert_close(expected_torch, torch.rand(4), rtol=0, atol=0)
        torch.testing.assert_close(expected_cuda, torch.rand(4, device="cuda").cpu(), rtol=0, atol=0)

    def test_batched_options_seeds_and_next_iteration_champion_after_promotion(self):
        self.config = replace(self.config, games_per_iteration=5, self_play_batch_size=2,
                              promotion_threshold=1.0)
        trainer = self.trainer()
        initial_champion = trainer.champion
        starting_weights = cpu_tree(initial_champion.state_dict())
        observed = []

        def generate(model, games, *, options, temperature, seed, batch_size, device):
            self.assertIsInstance(options, GpuPUCTOptions)
            self.assertEqual(options.simulations, self.config.simulations)
            self.assertEqual(options.c_puct, self.config.c_puct)
            self.assertEqual(options.dirichlet_alpha, self.config.dirichlet_alpha)
            self.assertEqual(options.dirichlet_epsilon, self.config.dirichlet_epsilon)
            self.assertEqual(options.seed, seed)
            self.assertEqual(temperature, self.config.temperature)
            self.assertEqual(batch_size, 2)
            self.assertEqual(torch.device(device).type, "cuda")
            self.assertIs(type(seed), int)
            self.assertTrue(0 <= seed < 2 ** 64)
            observed.append((model, games, seed, cpu_tree(model.state_dict())))
            return [pass_game() for _ in range(games)]

        with patch.object(loop, "collect_gpu_puct_games", side_effect=generate), \
                patch.object(loop, "collect_puct_game") as cpu_collect, \
                patch.object(loop, "evaluate_models", side_effect=[evaluation(0), evaluation(2),
                                                                   evaluation(0)]):
            rejected = trainer.run_iteration()
            self.assertFalse(rejected["promoted"])
            self.assertIs(trainer.champion, initial_champion)
            self.assert_tree_equal(starting_weights, cpu_tree(trainer.champion.state_dict()))
            promoted = trainer.run_iteration()
            self.assertTrue(promoted["promoted"])
            updated_champion = trainer.champion
            updated_weights = cpu_tree(updated_champion.state_dict())
            self.assertIsNot(updated_champion, initial_champion)
            self.assert_tree_equal(updated_weights, cpu_tree(trainer.model.state_dict()))
            trainer.run_iteration()
            cpu_collect.assert_not_called()

        self.assertEqual([row[1] for row in observed], [2, 2, 1] * 3)
        self.assertEqual(len({row[2] for row in observed}), 9)
        self.assertTrue(all(row[0] is initial_champion for row in observed[:6]))
        for model, _, _, weights in observed[6:]:
            self.assertIs(model, updated_champion)
            self.assert_tree_equal(updated_weights, weights)
        self.assertEqual(trainer.iteration, 3)
        self.assertEqual(trainer.self_play_games, 15)
        self.assertEqual(trainer.training_steps, 3)
        self.assertEqual(trainer.champion_version, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
