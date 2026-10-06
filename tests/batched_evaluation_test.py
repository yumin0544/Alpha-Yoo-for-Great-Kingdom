"""Native leaf evaluation routing, model isolation and subtree lifecycle."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.encoding import encode_state
from kingdom_ai.evaluation import evaluate_models
from kingdom_ai.model import PolicyValueNet


class BatchedEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(124)
            self.candidate = PolicyValueNet(channels=4, residual_blocks=0)
        self.reference = deepcopy(self.candidate)

    def test_routes_models_and_advances_both_private_trees_after_every_move(self):
        constructions = []
        searches = []
        with torch.no_grad():
            self.candidate.policy_head[-1].bias[0] = 10
            self.reference.policy_head[-1].bias[0] = -10

        class PassSearcher:
            def __init__(searcher, options, *, leaf_batch_size, reuse_tree):
                searcher.options = options
                searcher.leaf_batch_size = leaf_batch_size
                searcher.reuse_tree = reuse_tree
                searcher.advanced = []
                searcher.stats = {"advance_calls": 0, "reuse_hits": 0,
                                  "network_evaluations": 0, "inference_batches": 0,
                                  "max_observed_batch_size": 0, "reused_nodes": 0,
                                  "inherited_visits": 0}
                constructions.append(searcher)

            def search(searcher, state, evaluator):
                searches.append((float(evaluator.model.policy_head[-1].bias[0].detach()), state.to_play))
                encoded = encode_state(state)
                evaluator(encoded.features.numpy()[None], encoded.legal_mask.numpy()[None])
                searcher.stats.update(network_evaluations=1, inference_batches=1,
                                      max_observed_batch_size=1)
                move = engine.Move.pass_turn()
                return SimpleNamespace(best_move=move, simulations=searcher.options.simulations,
                                       moves=[SimpleNamespace(move=move,
                                                              visits=searcher.options.simulations)])

            def advance(searcher, move):
                searcher.advanced.append(move)
                searcher.stats["advance_calls"] += 1
                return False

        diagnostics = {}
        with patch.object(engine, "BatchedPUCT", PassSearcher, create=True):
            result = evaluate_models(self.candidate, self.reference, games=4, simulations=13,
                                     seed=2**64 - 1, workers=1, backend="batched_cpp",
                                     leaf_batch_size=3, reuse_tree=True, diagnostics=diagnostics)
        self.assertEqual(result.win_rate, 0.5)
        self.assertEqual(result.total_plies, 8)
        self.assertEqual(searches, [(10., engine.Cell.Black), (-10., engine.Cell.White),
                                    (-10., engine.Cell.Black), (10., engine.Cell.White)] * 2)
        self.assertEqual(len(constructions), 8)
        for searcher in constructions:
            self.assertEqual(len(searcher.advanced), 2)
            self.assertTrue(all(move.is_pass() for move in searcher.advanced))
            self.assertEqual(searcher.options.simulations, 13)
            self.assertEqual(searcher.options.time_limit_ms, 0)
            self.assertEqual(searcher.options.dirichlet_epsilon, 0)
            self.assertEqual(searcher.leaf_batch_size, 3)
            self.assertTrue(searcher.reuse_tree)
        self.assertEqual([searcher.options.seed for searcher in constructions],
                         [2**64 - 1] * 4 + [0] * 4)
        self.assertEqual(diagnostics["search"]["advance_calls"], 16)
        self.assertEqual(diagnostics["search"]["network_evaluations"], 8)
        self.assertEqual(diagnostics["search"]["max_observed_batch_size"], 1)
        self.assertEqual(diagnostics["candidate_inference"]["network_evaluations"], 4)
        self.assertEqual(diagnostics["reference_inference"]["network_evaluations"], 4)

    def test_invalid_options_fail_before_new_searcher_construction(self):
        invalid = {
            "backend": (None, "unknown", 1),
            "leaf_batch_size": (0, -1, True, 1.0),
            "reuse_tree": (None, 0, 1, "true"),
            "inference_wait_ms": (-1, float("nan"), float("inf"), True, "0"),
            "diagnostics": ([], 1, True),
        }
        with patch.object(engine, "BatchedPUCT", create=True) as searcher:
            for key, values in invalid.items():
                for value in values:
                    with self.subTest(option=key, value=value), \
                            self.assertRaises((TypeError, ValueError)):
                        options = {"backend": "batched_cpp", key: value}
                        evaluate_models(self.candidate, self.reference, **options)
            searcher.assert_not_called()

    @unittest.skipUnless(hasattr(engine, "BatchedPUCT"), "native BatchedPUCT is not built")
    def test_native_leaf_batches_are_used_without_python_state_encoding(self):
        diagnostics = {}
        with patch("kingdom_ai.encoding.encode_state", side_effect=AssertionError("Python encoding")):
            result = evaluate_models(self.candidate, self.reference, games=2, simulations=8,
                                     seed=926, workers=1, backend="batched_cpp", leaf_batch_size=4,
                                     reuse_tree=True, diagnostics=diagnostics)
        self.assertEqual(result.win_rate, 0.5)
        self.assertGreater(diagnostics["search"]["network_evaluations"], 0)
        self.assertGreater(diagnostics["search"]["max_observed_batch_size"], 1)
        self.assertGreater(diagnostics["search"]["advance_calls"], 0)
        self.assertGreater(diagnostics["search"]["reuse_hits"], 0)
        inferred = sum(diagnostics[key]["network_evaluations"]
                       for key in ("candidate_inference", "reference_inference"))
        self.assertEqual(inferred, diagnostics["search"]["network_evaluations"])
        self.assertTrue(all(diagnostics[key]["max_observed_batch_size"] <= 4
                            for key in ("candidate_inference", "reference_inference")))

    @unittest.skipUnless(hasattr(engine, "BatchedPUCT"), "native BatchedPUCT is not built")
    def test_native_pairs_are_repeatable_and_preserve_source_models_and_rng(self):
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
        options = dict(games=4, simulations=8, seed=271, opening_moves=3, workers=4,
                       backend="batched_cpp", leaf_batch_size=4, reuse_tree=True,
                       tactical_checks=True)
        first = evaluate_models(*models, **options)
        second = evaluate_models(*models, **options)
        self.assertEqual(first, second)
        self.assertEqual(first.win_rate, .5)
        self.assertEqual({module: module.training for module in modes}, modes)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for model, expected in zip(models, weights):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            for parameter in model.parameters():
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, .25),
                                           rtol=0, atol=0)

    @unittest.skipUnless(hasattr(engine, "BatchedPUCT"), "native BatchedPUCT is not built")
    def test_native_model_failure_closes_services_and_restores_modes(self):
        modes = {module: module.training for model in (self.candidate, self.reference)
                 for module in model.modules()}
        with patch.object(PolicyValueNet, "forward", side_effect=LookupError("native inference failed")), \
                self.assertRaisesRegex(LookupError, "native inference failed"):
            evaluate_models(self.candidate, self.reference, games=4, simulations=8, workers=4,
                            backend="batched_cpp", leaf_batch_size=4)
        self.assertEqual({module: module.training for module in modes}, modes)

    @unittest.skipUnless(hasattr(engine, "BatchedPUCT") and torch.cuda.is_available(),
                         "native BatchedPUCT or CUDA is unavailable")
    def test_cuda_native_identical_models_keep_color_pair_balance(self):
        candidate = self.candidate.to("cuda")
        reference = self.reference.to("cuda")
        result = evaluate_models(candidate, reference, games=4, simulations=8, workers=4,
                                 seed=821, backend="batched_cpp", leaf_batch_size=4,
                                 tactical_checks=True)
        self.assertEqual(result.win_rate, .5)
        self.assertTrue(all(parameter.device.type == "cuda" for model in (candidate, reference)
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
