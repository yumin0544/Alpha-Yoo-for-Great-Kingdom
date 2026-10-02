"""Equal-budget color pairing, completed games and read-only evaluation."""

from dataclasses import asdict, FrozenInstanceError
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.evaluation import EvaluationResult, evaluate_models
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.puct import sample_visits


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(791)
            self.candidate = PolicyValueNet(channels=4, residual_blocks=0)
            self.reference = PolicyValueNet(channels=4, residual_blocks=0)
        self.reference.load_state_dict(self.candidate.state_dict())

    def test_identical_models_have_half_wins_with_paired_sampled_openings(self):
        result = evaluate_models(
            self.candidate, self.reference, games=4, simulations=4,
            seed=271, opening_moves=6, opening_temperature=1.0,
        )
        self.assertEqual(result.games, 4)
        self.assertEqual(result.wins, 2)
        self.assertEqual(result.losses, 2)
        self.assertEqual(result.win_rate, 0.5)
        self.assertEqual(result.wins_as_black + result.wins_as_white, result.wins)
        self.assertEqual(sum(result.endings.values()), result.games)
        self.assertTrue(set(result.endings) <= {"Capture", "Suicide", "TwoPasses"})
        self.assertTrue(all(count % 2 == 0 for count in result.endings.values()))
        self.assertGreaterEqual(result.total_plies, 2 * result.games)
        self.assertLessEqual(result.total_plies, (2 * engine.CELL_COUNT + 2) * result.games)
        json.dumps(asdict(result))
        with self.assertRaises(FrozenInstanceError):
            result.wins = 0

    def test_evaluation_is_repeatable_and_preserves_weights_gradients_modes_and_rng(self):
        self.candidate.train()
        self.candidate.policy_head.eval()
        self.reference.eval()
        self.reference.value_head.train()
        modes = {module: module.training for model in (self.candidate, self.reference)
                 for module in model.modules()}
        models = (self.candidate, self.reference)
        weights = [{name: value.clone() for name, value in model.state_dict().items()}
                   for model in models]
        for model in models:
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.25)
        rng = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        options = dict(games=2, simulations=2, seed=418, opening_moves=3)
        first = evaluate_models(*models, **options)
        second = evaluate_models(*models, **options)
        self.assertEqual(first, second)
        self.assertEqual(torch.get_num_threads(), threads)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for model, expected in zip(models, weights):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            for parameter in model.parameters():
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25),
                                           rtol=0, atol=0)
        self.assertEqual({module: module.training for module in modes}, modes)

    def test_actor_colors_budgets_seeds_and_opening_schedule(self):
        constructions = []
        searches = []
        samples = []

        class PassSearcher:
            def __init__(searcher, model, options):
                searcher.model = model
                searcher.options = options
                constructions.append((model, options))

            def search(searcher, state):
                searches.append((searcher.model, state.to_play))
                move = engine.Move.pass_turn()
                return SimpleNamespace(
                    best_move=move, simulations=searcher.options.simulations,
                    moves=[SimpleNamespace(move=move, visits=searcher.options.simulations)],
                )

        def record_sample(result, *, temperature, generator):
            samples.append((temperature, generator.initial_seed()))
            return sample_visits(result, temperature=temperature, generator=generator)

        with patch("kingdom_ai.evaluation.PUCT", PassSearcher), \
                patch("kingdom_ai.evaluation.sample_visits", record_sample):
            result = evaluate_models(
                self.candidate, self.reference, games=4, simulations=13,
                c_puct=2.0, seed=2 ** 64 - 1, opening_moves=1, opening_temperature=0.5,
            )
        self.assertEqual(result.wins, 2)
        self.assertEqual(result.wins_as_black, 0)
        self.assertEqual(result.wins_as_white, 2)
        self.assertEqual(result.total_plies, 8)
        self.assertEqual(result.endings, {"TwoPasses": 4})
        self.assertEqual(searches, [
            (self.candidate, engine.Cell.Black), (self.reference, engine.Cell.White),
            (self.reference, engine.Cell.Black), (self.candidate, engine.Cell.White),
        ] * 2)
        self.assertEqual(len(constructions), 8)
        for _, options in constructions:
            self.assertEqual(options.simulations, 13)
            self.assertEqual(options.c_puct, 2.0)
            self.assertEqual(options.time_limit_ms, 0)
            self.assertEqual(options.dirichlet_epsilon, 0)
        self.assertEqual([options.seed for _, options in constructions],
                         [2 ** 64 - 1] * 4 + [0] * 4)
        self.assertEqual(samples, [(0.5, 2 ** 64 - 1), (0.0, 2 ** 64 - 1)] * 2
                         + [(0.5, 0), (0.0, 0)] * 2)

    def test_zero_opening_moves_uses_deterministic_selection(self):
        with patch("kingdom_ai.evaluation.sample_visits", wraps=sample_visits) as sample:
            result = evaluate_models(self.candidate, self.reference, games=2,
                                     simulations=1, opening_moves=0)
        self.assertEqual(result.win_rate, 0.5)
        self.assertGreater(sample.call_count, 0)
        self.assertTrue(all(call.kwargs["temperature"] == 0 for call in sample.call_args_list))

    def test_modes_restore_when_search_raises(self):
        self.candidate.train()
        self.candidate.policy_head.eval()
        modes = {module: module.training for module in self.candidate.modules()}

        def fail(*args, **kwargs):
            self.candidate.eval()
            raise LookupError("evaluation failed")

        with patch("kingdom_ai.evaluation.PUCT", side_effect=fail), \
                self.assertRaisesRegex(LookupError, "evaluation failed"):
            evaluate_models(self.candidate, self.reference, games=2, simulations=1)
        self.assertEqual({module: module.training for module in modes}, modes)

    def test_invalid_options_fail_before_search(self):
        invalid = {
            "games": (0, 1, 3, -2, True, 2.0),
            "simulations": (0, -1, True, 1.0),
            "c_puct": (0, -1, float("nan"), float("inf"), True, "1.5"),
            "seed": (-1, 2 ** 64, True, 0.0),
            "opening_moves": (-1, True, 1.0),
            "opening_temperature": (-1, float("nan"), float("inf"), True, "1"),
        }
        with patch("kingdom_ai.evaluation.PUCT") as searcher:
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(option=name, value=value), \
                            self.assertRaises((TypeError, ValueError)):
                        evaluate_models(self.candidate, self.reference, **{name: value})
            searcher.assert_not_called()
        for candidate, reference in ((None, self.reference), (self.candidate, object())):
            with self.assertRaises(TypeError):
                evaluate_models(candidate, reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
