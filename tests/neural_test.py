"""Neural interface and C++-generated training integration checks.

Run with the optional learning dependencies installed:
    python tests/neural_test.py
"""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import my_board_engine as engine
from kingdom_ai.checkpoint import load_model, save_model
from kingdom_ai.encoding import (
    ACTION_SIZE,
    BOARD_SIZE,
    FEATURE_NAMES,
    FORMAT_VERSION,
    INPUT_CHANNELS,
    PASS_ACTION,
    action_to_move,
    encode_state,
    move_to_action,
    terminal_value,
    visit_policy,
)
from kingdom_ai.inference import NeuralAgent
from kingdom_ai.model import PolicyValueNet, masked_policy
from kingdom_ai.training import (
    collect_mcts_game,
    make_batch,
    policy_value_loss,
    train_step,
)


def make_board(black=(), white=(), neutral=None):
    board = engine.Board(neutral=neutral)
    for points, color in ((black, engine.Cell.Black), (white, engine.Cell.White)):
        for row, col in points:
            if not board.place(engine.Position(row, col), color):
                raise AssertionError("Invalid neural test fixture")
    return board


def houses_board():
    return make_board(
        ((1, 2), (2, 1), (2, 3), (3, 2)),
        ((5, 6), (6, 5), (6, 7), (7, 6)),
        engine.Position(4, 4),
    )


def capture_state():
    state = engine.State(
        make_board(((1, 2), (2, 1), (3, 2)), ((2, 2),)), engine.Cell.Black
    )
    if not state.place(2, 3).accepted():
        raise AssertionError("Capture fixture did not accept its winning move")
    return state


def suicide_state():
    return engine.State(
        make_board(((0, 1),), ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))),
        engine.Cell.Black,
    )


class NeuralInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_input_and_action_contract(self):
        self.assertEqual((FORMAT_VERSION, BOARD_SIZE, INPUT_CHANNELS), (1, 9, 10))
        self.assertEqual((ACTION_SIZE, PASS_ACTION), (82, 81))
        self.assertIsInstance(FEATURE_NAMES, tuple)
        self.assertEqual(len(FEATURE_NAMES), INPUT_CHANNELS)
        self.assertEqual(len(set(FEATURE_NAMES)), INPUT_CHANNELS)
        encoded = encode_state(engine.State())
        self.assertEqual(encoded.features.shape, (10, 9, 9))
        self.assertEqual(encoded.features.dtype, torch.float32)
        self.assertEqual(encoded.legal_mask.shape, (82,))
        self.assertEqual(encoded.legal_mask.dtype, torch.bool)
        self.assertEqual(encoded.to_play, engine.Cell.Black)
        self.assertEqual(encoded.features.device.type, "cpu")
        self.assertTrue(torch.isfinite(encoded.features).all())
        self.assertEqual(encoded.features[2, 4, 4].item(), 1.0)
        self.assertEqual(encoded.features[2].sum().item(), 1.0)
        self.assertFalse(encoded.legal_mask[40].item())
        self.assertTrue(encoded.legal_mask[PASS_ACTION].item())
        self.assertEqual(encoded.legal_mask.sum().item(), 81)

    def test_relative_colors_and_absolute_komi_identity(self):
        board = make_board(((0, 1), (3, 4), (8, 7)), ((6, 2),),
                           engine.Position(2, 7))
        black = encode_state(engine.State(board, engine.Cell.Black)).features
        white = encode_state(engine.State(board, engine.Cell.White)).features
        torch.testing.assert_close(black[0], white[1], rtol=0, atol=0)
        torch.testing.assert_close(black[1], white[0], rtol=0, atol=0)
        torch.testing.assert_close(black[2], white[2], rtol=0, atol=0)
        torch.testing.assert_close(black[3], white[4], rtol=0, atol=0)
        torch.testing.assert_close(black[4], white[3], rtol=0, atol=0)
        self.assertEqual(black[0].sum().item(), 3)
        self.assertEqual(black[1].sum().item(), 1)
        self.assertEqual(black[0, 3, 4].item(), 1)
        self.assertEqual(black[1, 6, 2].item(), 1)
        self.assertEqual(black[2, 2, 7].item(), 1)
        torch.testing.assert_close(black[5], torch.ones((9, 9)), rtol=0, atol=0)
        torch.testing.assert_close(white[5], torch.zeros((9, 9)), rtol=0, atol=0)
        torch.testing.assert_close(black[6], torch.full((9, 9), 38 / 41))
        torch.testing.assert_close(black[7], torch.full((9, 9), 40 / 41))
        torch.testing.assert_close(black[6], white[7], rtol=0, atol=0)
        torch.testing.assert_close(black[7], white[6], rtol=0, atol=0)

    def test_house_features_and_legal_mask_match_engine(self):
        for player in (engine.Cell.Black, engine.Cell.White):
            with self.subTest(player=player):
                state = engine.State(houses_board(), player)
                encoded = encode_state(state)
                own, enemy = ((2, 2), (6, 6)) if player == engine.Cell.Black else (
                    (6, 6), (2, 2)
                )
                self.assertEqual(encoded.features[3, own[0], own[1]].item(), 1)
                self.assertEqual(encoded.features[4, enemy[0], enemy[1]].item(), 1)
                self.assertEqual(encoded.features[3].sum().item(), 1)
                self.assertEqual(encoded.features[4].sum().item(), 1)
                self.assertFalse(encoded.legal_mask[20].item())
                self.assertFalse(encoded.legal_mask[60].item())
                expected = torch.zeros(ACTION_SIZE, dtype=torch.bool)
                for move in state.legal_moves():
                    expected[move_to_action(move)] = True
                torch.testing.assert_close(encoded.legal_mask, expected, rtol=0, atol=0)
                torch.testing.assert_close(encoded.features[9],
                                           expected[:81].reshape(9, 9).float(),
                                           rtol=0, atol=0)

    def test_stone_stocks_and_pass_counter_follow_state(self):
        state = engine.State()
        self.assertTrue(state.place(0, 0).accepted())
        self.assertTrue(state.pass_turn().accepted())
        encoded = encode_state(state)
        self.assertEqual(encoded.to_play, engine.Cell.Black)
        torch.testing.assert_close(encoded.features[6], torch.full((9, 9), 40 / 41))
        torch.testing.assert_close(encoded.features[7], torch.ones((9, 9)))
        torch.testing.assert_close(encoded.features[8], torch.full((9, 9), 0.5))
        self.assertTrue(state.place(0, 1).accepted())
        next_encoded = encode_state(state)
        self.assertEqual(next_encoded.to_play, engine.Cell.White)
        torch.testing.assert_close(next_encoded.features[6], torch.ones((9, 9)))
        torch.testing.assert_close(next_encoded.features[7],
                                   torch.full((9, 9), 39 / 41))
        torch.testing.assert_close(next_encoded.features[8], torch.zeros((9, 9)))

    def test_neutral_variants_are_represented(self):
        empty = encode_state(engine.State(neutral=None))
        self.assertEqual(empty.features[2].sum().item(), 0)
        self.assertEqual(empty.legal_mask.sum().item(), ACTION_SIZE)
        shifted = encode_state(engine.State(neutral=engine.Position(0, 8)))
        self.assertEqual(shifted.features[2, 0, 8].item(), 1)
        self.assertEqual(shifted.features[2, 4, 4].item(), 0)
        self.assertFalse(shifted.legal_mask[8].item())

    def test_exhausted_default_stone_stock_allows_only_pass(self):
        board = make_board(tuple(divmod(action, 9) for action in range(41)))
        state = engine.State(board, engine.Cell.Black)
        self.assertEqual(state.remaining_stones(engine.Cell.Black), 0)
        encoded = encode_state(state)
        self.assertEqual(encoded.features[6].sum().item(), 0)
        self.assertEqual(encoded.features[9].sum().item(), 0)
        self.assertEqual(encoded.legal_mask.sum().item(), 1)
        self.assertTrue(encoded.legal_mask[PASS_ACTION].item())
        agent = NeuralAgent(PolicyValueNet(channels=8, residual_blocks=1))
        prediction = agent.predict(state)
        self.assertEqual(prediction.best_action, PASS_ACTION)
        self.assertTrue(prediction.best_move.is_pass())
        self.assertEqual(prediction.policy[PASS_ACTION].item(), 1)

    def test_unrepresented_analysis_rules_are_rejected(self):
        agent = NeuralAgent(PolicyValueNet(channels=8, residual_blocks=1))
        for rules in (
            engine.GameRules(suicide_rule=engine.SuicideRule.Forbidden),
            engine.GameRules(allow_own_territory_moves=True),
            engine.GameRules(allow_single_edge_territory=False),
            engine.GameRules(stones_per_player=40),
        ):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                encode_state(engine.State(rules))
            terminal = engine.State(rules)
            terminal.pass_turn()
            terminal.pass_turn()
            self.assertTrue(terminal.result.finished())
            with self.subTest(terminal_rules=rules), self.assertRaises(ValueError):
                agent.predict(terminal)

    def test_suicide_is_a_legal_action_with_immediate_loss(self):
        state = suicide_state()
        encoded = encode_state(state)
        self.assertTrue(encoded.legal_mask[10].item())
        self.assertTrue(state.place(1, 1).accepted())
        self.assertEqual(state.result.reason, engine.EndReason.Suicide)
        self.assertEqual(state.result.winner, engine.Cell.White)
        self.assertEqual(state.to_play, engine.Cell.White)
        self.assertEqual(terminal_value(state), 1.0)
        terminal = encode_state(state)
        self.assertFalse(terminal.legal_mask.any().item())
        self.assertEqual(terminal.features[9].sum().item(), 0)

    def test_terminal_value_uses_current_player_for_all_end_reasons(self):
        captured = capture_state()
        self.assertEqual(captured.result.reason, engine.EndReason.Capture)
        self.assertEqual(captured.to_play, engine.Cell.White)
        self.assertEqual(terminal_value(captured), -1.0)
        passed = engine.State()
        passed.pass_turn()
        passed.pass_turn()
        self.assertEqual(passed.result.reason, engine.EndReason.TwoPasses)
        self.assertEqual(passed.to_play, engine.Cell.Black)
        self.assertEqual(terminal_value(passed), -1.0)
        torch.testing.assert_close(encode_state(passed).features[8], torch.ones((9, 9)))
        with self.assertRaises(ValueError):
            terminal_value(engine.State())

    def test_action_mapping_bijection_and_bad_coordinates(self):
        for action in range(ACTION_SIZE):
            move = action_to_move(action)
            self.assertEqual(move_to_action(move), action)
            if action == PASS_ACTION:
                self.assertTrue(move.is_pass())
            else:
                self.assertEqual((move.point.row, move.point.col), divmod(action, 9))
        for value in (-1, 82, True, False, 1.5, "1"):
            with self.subTest(action=value), self.assertRaises((TypeError, ValueError)):
                action_to_move(value)
        for row, col in ((-1, 0), (0, -1), (9, 0), (0, 9)):
            with self.subTest(row=row, col=col), self.assertRaises(ValueError):
                move_to_action(engine.Move.place(row, col))

    def test_mcts_visit_targets_normalize_actual_root_counts(self):
        state = engine.State()
        before = state.board.to_string()
        search = engine.MCTS(engine.MCTSOptions(simulations=64, seed=239)).search(state)
        actual = visit_policy(search, state)
        expected = torch.zeros(ACTION_SIZE, dtype=torch.float32)
        for item in search.moves:
            expected[move_to_action(item.move)] = item.visits / 64
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(actual.sum().item(), 1)
        self.assertTrue((actual[~encode_state(state).legal_mask] == 0).all())
        self.assertEqual(state.board.to_string(), before)

    def test_mcts_visit_targets_reject_empty_illegal_and_negative_counts(self):
        state = engine.State()
        invalid = (
            SimpleNamespace(moves=[], simulations=0),
            SimpleNamespace(moves=[SimpleNamespace(
                move=engine.Move.place(0, 0), visits=0)], simulations=0),
            SimpleNamespace(moves=[SimpleNamespace(
                move=engine.Move.place(4, 4), visits=1)], simulations=1),
            SimpleNamespace(moves=[SimpleNamespace(
                move=engine.Move.place(0, 0), visits=-1)], simulations=-1),
            SimpleNamespace(moves=[
                SimpleNamespace(move=engine.Move.place(0, 0), visits=-1),
                SimpleNamespace(move=engine.Move.place(0, 1), visits=2),
            ], simulations=1),
        )
        for search in invalid:
            with self.subTest(search=search), self.assertRaises(ValueError):
                visit_policy(search, state)
        terminal = capture_state()
        with self.assertRaises(ValueError):
            visit_policy(engine.MCTS().search(terminal), terminal)

    def test_batched_model_shapes_ranges_and_gradients(self):
        torch.manual_seed(81)
        model = PolicyValueNet(channels=8, residual_blocks=1)
        features = torch.stack([
            encode_state(engine.State()).features,
            encode_state(engine.State(houses_board(), engine.Cell.White)).features,
        ])
        logits, values = model(features)
        self.assertEqual(logits.shape, (2, 82))
        self.assertEqual(values.shape, (2,))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((values.abs() <= 1).all())
        (logits.square().mean() + (values - 1).square().mean()).backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))

    def test_masked_policy_zeros_illegal_actions_and_terminal_rows(self):
        logits = torch.zeros((3, ACTION_SIZE))
        logits[0, 40] = 1e20  # An occupied neutral point cannot dominate the policy.
        logits[1, 0] = 1e20
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, 0] = True
        mask[0, PASS_ACTION] = True
        mask[1, 0] = True
        probabilities = masked_policy(logits, mask)
        self.assertEqual(probabilities.shape, logits.shape)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue((probabilities[~mask] == 0).all())
        torch.testing.assert_close(probabilities[0, [0, PASS_ACTION]],
                                   torch.tensor([0.5, 0.5]), rtol=0, atol=0)
        self.assertEqual(probabilities[1, 0].item(), 1)
        self.assertEqual(probabilities[2].sum().item(), 0)

    def test_masked_policy_rejects_invalid_values_and_shapes(self):
        mask = torch.ones((1, ACTION_SIZE), dtype=torch.bool)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            logits = torch.zeros((1, ACTION_SIZE))
            logits[0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                masked_policy(logits, mask)
        with self.assertRaises((TypeError, ValueError)):
            masked_policy(torch.zeros((1, ACTION_SIZE)), mask.float())
        with self.assertRaises(ValueError):
            masked_policy(torch.zeros((1, 81)), mask)

    def test_neural_agent_masks_actions_and_preserves_input(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        agent = NeuralAgent(model)
        state = engine.State(houses_board(), engine.Cell.Black)
        before = state.board.to_string()
        prediction = agent.predict(state)
        mask = encode_state(state).legal_mask
        self.assertEqual(prediction.policy.shape, (ACTION_SIZE,))
        self.assertEqual(prediction.policy.device.type, "cpu")
        self.assertFalse(prediction.policy.requires_grad)
        self.assertTrue(torch.isfinite(prediction.policy).all())
        self.assertTrue((prediction.policy[~mask] == 0).all())
        self.assertAlmostEqual(prediction.policy.sum().item(), 1, places=6)
        self.assertGreaterEqual(prediction.value, -1)
        self.assertLessEqual(prediction.value, 1)
        self.assertEqual(prediction.best_action, int(prediction.policy.argmax()))
        self.assertEqual(move_to_action(prediction.best_move), prediction.best_action)
        self.assertTrue(state.is_legal(prediction.best_move))
        self.assertEqual(state.board.to_string(), before)

    def test_neural_agent_terminal_predictions_are_exact(self):
        agent = NeuralAgent(PolicyValueNet(channels=8, residual_blocks=1))
        suicide = suicide_state()
        suicide.place(1, 1)
        for state in (capture_state(), suicide):
            with self.subTest(reason=state.result.reason):
                prediction = agent.predict(state)
                self.assertEqual(prediction.policy.sum().item(), 0)
                self.assertEqual(prediction.value, terminal_value(state))
                self.assertIsNone(prediction.best_action)
                self.assertIsNone(prediction.best_move)

    def test_neural_inference_restores_model_training_mode(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        agent = NeuralAgent(model)
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                prediction = agent.predict(engine.State())
                self.assertEqual(model.training, training)
                self.assertFalse(prediction.policy.requires_grad)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_full_mcts_game_supplies_player_relative_training_targets(self):
        game = collect_mcts_game(simulations=8, seed=773)
        self.assertIn(game.winner, (engine.Cell.Black, engine.Cell.White))
        self.assertIn(game.reason, (engine.EndReason.Capture, engine.EndReason.Suicide,
                                   engine.EndReason.TwoPasses))
        self.assertGreaterEqual(len(game.samples), 2)
        for index, sample in enumerate(game.samples):
            with self.subTest(index=index):
                self.assertEqual(sample.features.shape, (10, 9, 9))
                self.assertEqual(sample.legal_mask.shape, (82,))
                self.assertEqual(sample.policy.shape, (82,))
                self.assertEqual(sample.value, 1.0 if sample.to_play == game.winner else -1.0)
                self.assertEqual(sample.to_play,
                                 engine.Cell.Black if index % 2 == 0 else engine.Cell.White)
                self.assertAlmostEqual(sample.policy.sum().item(), 1, places=6)
                self.assertTrue((sample.policy[~sample.legal_mask] == 0).all())
        batch = make_batch(game.samples)
        count = len(game.samples)
        self.assertEqual(batch.features.shape, (count, 10, 9, 9))
        self.assertEqual(batch.legal_mask.shape, (count, 82))
        self.assertEqual(batch.policy.shape, (count, 82))
        self.assertEqual(batch.value.shape, (count,))
        self.assertEqual(batch.legal_mask.dtype, torch.bool)
        self.assertEqual(batch.features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch.features).all())
        with self.assertRaises(ValueError):
            make_batch([])

    def test_policy_and_value_loss_matches_small_hand_calculation(self):
        mask = torch.zeros((1, ACTION_SIZE), dtype=torch.bool)
        mask[0, 0] = True
        mask[0, PASS_ACTION] = True
        target_policy = torch.zeros((1, ACTION_SIZE))
        target_policy[0, 0] = 0.25
        target_policy[0, PASS_ACTION] = 0.75
        for logit in (0.0, torch.finfo(torch.float32).min):
            with self.subTest(logit=logit):
                logits = torch.full((1, ACTION_SIZE), logit, requires_grad=True)
                values = torch.tensor([0.25], requires_grad=True)
                losses = policy_value_loss(logits, values, target_policy,
                                           torch.tensor([1.0]), mask)
                self.assertAlmostEqual(losses.policy.item(), math.log(2), places=6)
                self.assertAlmostEqual(losses.value.item(), 0.75 ** 2, places=6)
                self.assertAlmostEqual(losses.total.item(),
                                       math.log(2) + 0.75 ** 2, places=6)
                losses.total.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertTrue((logits.grad[~mask] == 0).all())
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertNotEqual(values.grad.item(), 0)

    def test_training_update_changes_parameters_with_finite_losses(self):
        torch.manual_seed(2041)
        batch = make_batch(collect_mcts_game(simulations=8, seed=41).samples)
        model = PolicyValueNet(channels=8, residual_blocks=1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        metrics = train_step(model, optimizer, batch)
        self.assertTrue({"loss", "policy_loss", "value_loss"}.issubset(metrics))
        self.assertTrue(all(math.isfinite(metrics[key])
                            for key in ("loss", "policy_loss", "value_loss")))
        self.assertGreaterEqual(metrics["loss"], 0)
        changed = [not torch.equal(before[name], value.detach())
                   for name, value in model.named_parameters()]
        self.assertTrue(any(changed))
        self.assertTrue(all(torch.isfinite(parameter).all()
                            for parameter in model.parameters()))

    def test_training_loss_rejects_invalid_targets(self):
        logits = torch.zeros((1, ACTION_SIZE))
        values = torch.zeros(1)
        mask = encode_state(engine.State()).legal_mask.unsqueeze(0)
        valid_policy = torch.zeros_like(logits)
        valid_policy[0, PASS_ACTION] = 1
        invalid_policies = []
        illegal = torch.zeros_like(logits)
        illegal[0, 40] = 1
        invalid_policies.append(illegal)
        invalid_policies.append(torch.zeros_like(logits))
        negative = valid_policy.clone()
        negative[0, 0] = -1
        negative[0, PASS_ACTION] = 2
        invalid_policies.append(negative)
        nonfinite = valid_policy.clone()
        nonfinite[0, 0] = float("nan")
        invalid_policies.append(nonfinite)
        for policy in invalid_policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                policy_value_loss(logits, values, policy, torch.ones(1), mask)
        for target in (torch.tensor([1.1]), torch.tensor([float("nan")])):
            with self.subTest(target=target), self.assertRaises(ValueError):
                policy_value_loss(logits, values, valid_policy, target, mask)
        with self.assertRaises(ValueError):
            policy_value_loss(logits, values, valid_policy, torch.ones(1),
                              torch.zeros_like(mask))

    def test_checkpoint_round_trip_preserves_outputs_and_architecture(self):
        torch.manual_seed(98)
        model = PolicyValueNet(channels=8, residual_blocks=1).eval()
        features = torch.stack([
            encode_state(engine.State()).features,
            encode_state(engine.State(houses_board(), engine.Cell.White)).features,
        ])
        with torch.inference_mode():
            expected = model(features)
        with tempfile.TemporaryDirectory(prefix="kingdom-model-test-") as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            self.assertTrue(path.is_file())
            restored = load_model(path, device="cpu").eval()
            self.assertEqual(restored.model_config, model.model_config)
            with torch.inference_mode():
                actual = restored(features)
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_checkpoint_rejects_incompatible_or_missing_metadata(self):
        model = PolicyValueNet(channels=8, residual_blocks=1)
        with tempfile.TemporaryDirectory(prefix="kingdom-schema-test-") as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            incompatible = dict(payload)
            incompatible["format_version"] = FORMAT_VERSION + 1
            torch.save(incompatible, path)
            with self.assertRaises(ValueError):
                load_model(path)
            incompatible = dict(payload)
            incompatible["feature_names"] = list(reversed(FEATURE_NAMES))
            torch.save(incompatible, path)
            with self.assertRaises(ValueError):
                load_model(path)
            torch.save({"state_dict": model.state_dict()}, path)
            with self.assertRaises(ValueError):
                load_model(path)
            with self.assertRaises(ValueError):
                save_model(model.double(), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
