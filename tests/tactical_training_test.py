"""Proof-only labels, family-level holdout and replay-preserving fine-tuning."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import my_board_engine as engine

from kingdom_ai.encoding import encode_state, move_to_action
from kingdom_ai.model import PolicyValueNet
from kingdom_ai.loop import Trainer, TrainingConfig
from kingdom_ai.replay import ReplayBuffer
from kingdom_ai.tactical_training import (
    CertifiedTacticalSample, collect_certified_samples, fine_tune_tactics,
    model_digest, split_tactical_samples, tactical_metrics,
)
from kingdom_ai.training import TrainingSample


def position(moves=()):
    state = engine.State()
    for row, col in moves:
        if not state.place(row, col).accepted():
            raise AssertionError("Invalid fixture")
    return state


def sample(case_id, state, family_id=None):
    encoded = encode_state(state)
    action = move_to_action(state.legal_moves()[0])
    policy = torch.zeros(82)
    policy[action] = 1.0
    target = TrainingSample(encoded.features, encoded.legal_mask, policy, 1.0, encoded.to_play)
    return CertifiedTacticalSample(case_id, family_id or case_id, "unit_test", target,
                                   {"outcome": "WIN", "winning_actions": [action]})


class TacticalTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.model = PolicyValueNet(channels=4, residual_blocks=0)
        self.rows = [sample("opening", position()),
                     sample("one_stone", position([(0, 0)])),
                     sample("two_stones", position([(0, 0), (8, 8)]))]

    def test_unknown_and_loss_are_never_imitation_labels(self):
        state = position()
        cases = [{"id": label, "state": state} for label in ("WIN", "LOSS", "UNKNOWN")]
        outcomes = iter(("WIN", "LOSS", "UNKNOWN"))
        def solver(snapshot, options):
            result = SimpleNamespace(outcome=next(outcomes), winning_moves=[snapshot.legal_moves()[0]],
                                     nodes=12, completed_depth=3, budget_exhausted=False)
            return result
        rows, report = collect_certified_samples(cases, object(),
            load_position=lambda case: case["state"], solver=solver)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample.value, 1.0)
        self.assertEqual(report["excluded_unknown"], 1)
        self.assertEqual(report["excluded_loss"], 1)
        self.assertEqual(report["records"][1]["training_label"], False)

    def test_invalid_proof_winning_actions_fail_closed(self):
        case = {"id": "invalid", "state": position()}
        for moves in ([], [engine.Move.place(4, 4)],
                      [engine.Move.place(0, 0), engine.Move.place(0, 0)]):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                collect_certified_samples([case], object(),
                    load_position=lambda row: row["state"],
                    solver=lambda state, options: SimpleNamespace(outcome="WIN", winning_moves=moves))

    def test_generation_max_cases_and_original_state_preserved(self):
        state = position()
        before = str(state.board)
        calls = []
        def solver(snapshot, options):
            calls.append(snapshot)
            move = snapshot.legal_moves()[0]
            snapshot.play(move)
            return SimpleNamespace(outcome="UNKNOWN", winning_moves=[])
        _, report = collect_certified_samples(
            [{"id": str(index), "state": state} for index in range(5)], object(),
            load_position=lambda row: row["state"], solver=solver, max_cases=2)
        self.assertEqual(report["cases_solved"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(str(state.board), before)
        self.assertEqual(state.to_play, engine.Cell.Black)

    def test_split_groups_related_families_and_d4_duplicates(self):
        first = sample("left", position([(0, 0)]), "left_family")
        sibling = sample("right", position([(0, 8)]), "right_family")
        third = sample("paired", position([(0, 0), (8, 8)]), "left_family")
        other = sample("empty", position(), "other_family")
        train, heldout = split_tactical_samples([first, sibling, third, other], seed=7)
        groups = [{row.case_id for row in train}, {row.case_id for row in heldout}]
        self.assertTrue(any({"left", "right", "paired"} <= group for group in groups))
        self.assertEqual({row.family_id for row in train} & {row.family_id for row in heldout}, set())
        again = split_tactical_samples([first, sibling, third, other], seed=7)
        self.assertEqual([row.case_id for row in train], [row.case_id for row in again[0]])

    def test_one_family_reports_no_heldout_not_fake_generalization(self):
        train, heldout = split_tactical_samples([self.rows[0]])
        self.assertEqual(len(train), 1)
        self.assertEqual(heldout, [])
        self.assertIsNone(tactical_metrics(self.model, heldout)["certified_top1"])

    def test_copy_training_preserves_original_flags_weights_grads_and_rng(self):
        self.model.train()
        self.model.trunk[1].eval()
        first_parameter = next(self.model.parameters())
        first_parameter.grad = torch.full_like(first_parameter, 0.25)
        flags = [module.training for module in self.model.modules()]
        digest = model_digest(self.model)
        rng = torch.get_rng_state().clone()
        candidate, report = fine_tune_tactics(self.model, self.rows[:2], self.rows[2:],
                                             steps=8, batch_size=4, learning_rate=0.01)
        self.assertIsNot(candidate, self.model)
        self.assertEqual(model_digest(self.model), digest)
        self.assertNotEqual(model_digest(candidate), digest)
        self.assertEqual(flags, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(first_parameter.grad, torch.full_like(first_parameter, 0.25)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(report["actual_tactical_fraction"], 1.0)
        self.assertEqual(report["before"]["heldout"]["cases"], 1)
        self.assertFalse(report["promoted"])

    def test_replay_mixing_does_not_overwrite_or_extend_replay(self):
        replay = ReplayBuffer(8)
        replay.extend([row.sample for row in self.rows])
        before = deepcopy(replay.state_dict())
        _, report = fine_tune_tactics(self.model, self.rows[:2], self.rows[2:],
                                     replay=replay, tactical_fraction=0.25, steps=3, batch_size=8)
        self.assertEqual(report["tactical_rows_per_batch"], 2)
        self.assertEqual(report["replay_rows_per_batch"], 6)
        after = replay.state_dict()
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]) if isinstance(before[key], torch.Tensor)
                            else before[key] == after[key])

    def test_train_heldout_overlap_rejected_before_model_mutation(self):
        original = model_digest(self.model)
        variants = [self.rows[0], sample("renamed", position(), "renamed_family")]
        for heldout in ([self.rows[0]], [variants[1]]):
            with self.subTest(heldout=heldout), self.assertRaisesRegex(ValueError, "disjoint"):
                fine_tune_tactics(self.model, [self.rows[0]], heldout, steps=1)
        self.assertEqual(model_digest(self.model), original)

    def test_bad_inputs_rejected_before_training(self):
        for kwargs in ({"steps": 0}, {"steps": True}, {"batch_size": 0}, {"learning_rate": float("nan")},
                       {"tactical_fraction": 0}, {"tactical_fraction": 1.1},
                       {"copy_model": True, "update_callback": lambda batch: {}},
                       {"generator": torch.Generator(device="cpu").get_state()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fine_tune_tactics(self.model, self.rows, **kwargs)
        corrupted = CertifiedTacticalSample("bad", "bad", "test", self.rows[0].sample,
                                            {"outcome": "UNKNOWN", "winning_actions": [0]})
        with self.assertRaisesRegex(ValueError, "certified WIN"):
            fine_tune_tactics(self.model, [corrupted])

    def test_external_update_callback_owns_updates(self):
        seen = []
        def update(batch):
            seen.append(batch)
            return {"loss": 1.0, "policy_loss": 0.5, "value_loss": 0.5}
        digest = model_digest(self.model)
        candidate, report = fine_tune_tactics(self.model, self.rows, steps=2, batch_size=4,
                                             copy_model=False, update_callback=update)
        self.assertIs(candidate, self.model)
        self.assertEqual(len(seen), 2)
        self.assertEqual(digest, model_digest(self.model))
        self.assertEqual(report["last_loss"]["loss"], 1.0)

    @unittest.skipUnless(hasattr(Trainer, "train_tactical_batch"), "Requires checkpoint-v6 integration")
    def test_trainer_callback_roundtrip_preserves_counters_and_champion(self):
        config = TrainingConfig(games_per_iteration=2, simulations=2, replay_capacity=8,
                                batch_size=2, train_steps_per_iteration=1,
                                evaluation_games=2, evaluation_simulations=2)
        trainer = Trainer(config, model=self.model)
        champion_digest = model_digest(trainer.champion)
        trainer.reconfigure(learning_rate=0.0001)
        _, report = fine_tune_tactics(trainer.model, self.rows, steps=2, batch_size=4,
            learning_rate=0.0001, generator=trainer.generator, replay=trainer.replay,
            copy_model=False, update_callback=trainer.train_tactical_batch)
        self.assertEqual(trainer.iteration, 0)
        self.assertEqual(trainer.self_play_games, 0)
        self.assertEqual(trainer.training_steps, 2)
        self.assertEqual(trainer.tactical_training_steps, 2)
        self.assertEqual(trainer.champion_version, 0)
        self.assertEqual(model_digest(trainer.champion), champion_digest)
        self.assertEqual(report["sampling_rng"], "caller_generator")
        with tempfile.TemporaryDirectory(prefix="kingdom-tactics-") as directory:
            path = Path(directory) / "latest.pt"
            trainer.save_checkpoint(path)
            restored = Trainer.load_checkpoint(path)
        self.assertEqual(restored.training_steps, 2)
        self.assertEqual(restored.tactical_training_steps, 2)
        self.assertEqual(restored.iteration, 0)
        self.assertEqual(model_digest(restored.champion), champion_digest)
        self.assertEqual(model_digest(restored.model), model_digest(trainer.model))
        self.assertTrue(torch.equal(restored.generator.get_state(), trainer.generator.get_state()))

    def test_cli_source_modes_are_mutually_exclusive(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "train_tactics.py"
        spec = importlib.util.spec_from_file_location("train_tactics_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--resume-training", "a.pt", "--initial-model", "b.pt", "--output", "new"])
        args = parser.parse_args(["--initial-model", "a.pt", "--output", "new", "--dry-run"])
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
