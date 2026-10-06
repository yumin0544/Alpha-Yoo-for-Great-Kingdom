"""CPU collection temperature boundaries keep the original visit targets."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import my_board_engine as engine
import torch

from kingdom_ai.training import collect_puct_game


class TemperatureScheduleTest(unittest.TestCase):
    def test_opening_threshold_zero_and_none_keep_raw_targets_and_value_perspective(self):
        search = SimpleNamespace(
            simulations=2,
            moves=[SimpleNamespace(move=engine.Move.place(0, 0), visits=1),
                   SimpleNamespace(move=engine.Move.pass_turn(), visits=1)],
        )
        cases = ((None, 0., [1., 1.]), (0, 0., [0., 0.]),
                 (1, 0., [1., 0.]), (1, .25, [1., .25]), (2, .25, [1., 1.]))
        for limit, final, expected in cases:
            temperatures = []

            def choose(result, *, temperature, generator):
                temperatures.append(temperature)
                return engine.Move.pass_turn()

            with self.subTest(limit=limit, final=final), \
                    patch("kingdom_ai.puct.PUCT") as searcher, \
                    patch("kingdom_ai.puct.sample_visits", side_effect=choose):
                searcher.return_value.search.return_value = search
                game = collect_puct_game(None, temperature=1., temperature_moves=limit,
                                         final_temperature=final)
            self.assertEqual(temperatures, expected)
            self.assertEqual(game.reason, engine.EndReason.TwoPasses)
            self.assertEqual([sample.value for sample in game.samples], [-1., 1.])
            for sample in game.samples:
                self.assertEqual(sample.policy[0].item(), .5)
                self.assertEqual(sample.policy[81].item(), .5)
                self.assertAlmostEqual(sample.policy.sum().item(), 1.)

    def test_bad_schedule_is_rejected_before_search_setup(self):
        invalid = (
            {"temperature_moves": -1}, {"temperature_moves": True},
            {"temperature_moves": 1.5}, {"temperature_moves": "2"},
            {"final_temperature": -1}, {"final_temperature": float("nan")},
            {"final_temperature": float("inf")}, {"final_temperature": True},
            {"final_temperature": "0"},
        )
        before = torch.get_rng_state().clone()
        with patch("kingdom_ai.puct.PUCT") as searcher:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    collect_puct_game(None, **kwargs)
            searcher.assert_not_called()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
