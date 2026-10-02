"""Connect a PyTorch policy/value model to the C++ game and perform one update."""

import argparse
from pathlib import Path

import my_board_engine as engine
import torch

from kingdom_ai import (
    NeuralAgent,
    PolicyValueNet,
    collect_mcts_game,
    load_model,
    make_batch,
    save_model,
    train_step,
)


def positive_integer(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("탐색 횟수는 1 이상이어야 합니다.")
    return count


def move_text(move: engine.Move | None) -> str:
    if move is None:
        return "종료된 대국"
    if move.is_pass():
        return "패스"
    point = move.point
    return f"{point.row + 1}행 {point.col + 1}열"


def player_name(player: engine.Cell) -> str:
    return "선공 흑/파랑" if player == engine.Cell.Black else "후공 백/주황"


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch 모델과 C++ 엔진 연결 검증")
    parser.add_argument(
        "--simulations", type=positive_integer, default=64,
        help="학습 예시를 만드는 순수 MCTS의 한 수당 탐색 횟수 (기본 64)",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드 (기본 42)")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/policy_value.pt"),
        help="저장할 모델 파일 (기본 models/policy_value.pt)",
    )
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64:
        parser.error("시드는 0 이상, 2^64 미만이어야 합니다.")

    # Limit this small CPU demonstration to one inference thread.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = PolicyValueNet(channels=32, residual_blocks=2)
    game = engine.State()
    prediction = NeuralAgent(model, device="cpu").predict(game)
    print("초기 가중치로 모델과 C++ 엔진을 연결합니다.")
    print(f"추천 수: {move_text(prediction.best_move)}")
    print(f"현재 차례의 모델 평가값: {prediction.value:+.4f} (아직 학습되지 않은 값)")
    if prediction.best_move is None:
        raise RuntimeError("진행 중인 대국에서 모델이 추천 수를 반환하지 않았습니다.")
    trial = game.copy()
    if not trial.play(prediction.best_move).accepted():
        raise RuntimeError("모델의 추천 수를 C++ 엔진이 거부했습니다.")
    print("C++ 엔진이 추천 수를 수락했습니다.")

    print(f"\n순수 MCTS 대국 1판을 생성합니다 (매 수 {args.simulations}회 탐색).")
    data = collect_mcts_game(simulations=args.simulations, seed=args.seed)
    print(f"종료: {len(data.samples)}수, 승자 {player_name(data.winner)}, 사유 {data.reason}")
    batch = make_batch(data.samples, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train_step(model, optimizer, batch)
    print(
        "학습 1회 완료: "
        f"전체 손실 {losses['loss']:.6f}, "
        f"정책 손실 {losses['policy_loss']:.6f}, "
        f"가치 손실 {losses['value_loss']:.6f}"
    )

    save_model(model, args.checkpoint)
    restored = load_model(args.checkpoint, device="cpu")
    after = NeuralAgent(model, device="cpu").predict(game)
    reloaded = NeuralAgent(restored, device="cpu").predict(game)
    torch.testing.assert_close(after.policy, reloaded.policy, rtol=0, atol=0)
    if after.value != reloaded.value:
        raise RuntimeError("저장 전후 모델의 평가값이 일치하지 않습니다.")
    print(f"모델 저장·복원 결과 일치: {args.checkpoint.resolve()}")
    print("1판과 학습 1회로 연결을 확인한 예제이며, 대국 실력이 검증된 모델은 아닙니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
