"""Use the PyTorch policy/value model in the C++ PUCT tree search."""

import argparse
from pathlib import Path
from time import perf_counter

import my_board_engine as engine
import torch

from kingdom_ai import (
    PUCT,
    PUCTOptions,
    PolicyValueNet,
    collect_puct_game,
    load_model,
    move_to_action,
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
    parser = argparse.ArgumentParser(description="C++ PUCT와 PyTorch 정책·가치 모델 대국")
    parser.add_argument(
        "--simulations", type=positive_integer, default=128,
        help="한 수당 완료 시뮬레이션 수 (기본 128)",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드 (기본 42)")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="읽어 사용할 모델 파일; 생략하면 새 무작위 초기 모델 사용",
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="양쪽 모두 PUCT로 종료까지 대국하고 학습 자료 생성",
    )
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64:
        parser.error("시드는 0 이상, 2^64 미만이어야 합니다.")

    # Only the example changes the global inference-thread setting.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.checkpoint is None:
        model = PolicyValueNet(channels=32, residual_blocks=2)
        print("새 모델의 무작위 초기 가중치를 사용합니다. 대국 실력은 아직 검증되지 않았습니다.")
    else:
        model = load_model(args.checkpoint, device="cpu")
        print(f"저장된 모델을 사용합니다: {args.checkpoint.resolve()}")

    options = PUCTOptions(
        simulations=args.simulations,
        seed=args.seed,
        dirichlet_epsilon=0.25 if args.self_play else 0.0,
    )
    if args.self_play:
        started = perf_counter()
        data = collect_puct_game(model, options=options, temperature=1.0, seed=args.seed)
        elapsed = perf_counter() - started
        print(
            f"대국 종료: {len(data.samples)}수, 승자 {player_name(data.winner)}, "
            f"사유 {data.reason}"
        )
        print(f"학습 표본 {len(data.samples)}개 생성, 이 실행의 경과 시간 {elapsed:.3f}초")
        print("정책 목표는 방문 비율, 가치 목표는 착수 전 플레이어의 최종 승패입니다.")
        return 0

    game = engine.State()
    result = PUCT(model, options=options, device="cpu").search(game)
    print(f"추천 수: {move_text(result.best_move)}")
    print(
        f"완료 시뮬레이션 {result.simulations}회, 트리 상태 {result.nodes}개, "
        f"신경망 평가 {result.network_evaluations}회, 경과 시간 {result.elapsed_seconds:.3f}초"
    )
    print(f"추천 행동의 현재 플레이어 평가값: {result.best_value:+.4f}")
    candidates = sorted(
        result.moves, key=lambda item: (-item.visits, -item.value, move_to_action(item.move))
    )
    for item in candidates[:5]:
        print(
            f"  {move_text(item.move)}: 방문 {item.visits}회, "
            f"사전 확률 {item.prior:.4f}, 평가값 {item.value:+.4f}"
        )
    if result.best_move is None or not game.play(result.best_move).accepted():
        raise RuntimeError("PUCT가 C++ 엔진이 수락하는 추천 수를 반환하지 않았습니다.")
    print("\n추천 수를 적용한 보드:")
    print(game.board.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
