"""Call the C++ engine and pure MCTS from Python; display coordinates from 1."""

import argparse

import my_board_engine as engine


def positive_integer(value: str) -> int:
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("탐색 횟수는 1 이상이어야 합니다.")
    return count


def player_name(player: engine.Cell) -> str:
    return "선공 흑/파랑" if player == engine.Cell.Black else "후공 백/주황"


def move_text(move: engine.Move) -> str:
    if move.is_pass():
        return "패스"
    point = move.point
    return f"{point.row + 1}행 {point.col + 1}열"


def print_search(ply: int, player: engine.Cell, result: engine.SearchResult) -> None:
    move = result.best_move
    if move is None:
        raise RuntimeError("진행 중인 대국에 추천 수가 없습니다.")
    print(
        f"{ply}수 {player_name(player)}: {move_text(move)} | "
        f"탐색 {result.simulations}회, 노드 {result.nodes}개, "
        f"추정 승률 {result.win_rate:.1%}, {result.elapsed_seconds:.3f}초"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python에서 C++ 엔진과 MCTS 호출")
    parser.add_argument(
        "--simulations", type=positive_integer, default=1000,
        help="한 수당 탐색 횟수 (기본 1000)",
    )
    parser.add_argument("--self-play", action="store_true", help="양쪽 AI로 종료까지 대국")
    args = parser.parse_args()

    game = engine.State()
    searcher = engine.MCTS(engine.MCTSOptions(simulations=args.simulations, seed=42))
    print("초기 보드 (x=선공, o=후공, v=중립 돌; 표시 좌표는 1부터 시작):")
    print(game.board.to_string())

    max_plies = 2 * len(game.board.cells) + 2 if args.self_play else 1
    for ply in range(1, max_plies + 1):
        if game.result.finished():
            break
        player = game.to_play
        recommendation = searcher.search(game)
        print_search(ply, player, recommendation)
        outcome = game.play(recommendation.best_move)
        if not outcome.accepted():
            raise RuntimeError(f"추천 수가 거부되었습니다: {outcome.error}")

    print("\n현재 보드:")
    print(game.board.to_string())
    score = game.score()
    print(f"현재 집: 선공 {score.black}칸, 후공 {score.white}칸")
    result = game.result
    if result.finished():
        reasons = {
            engine.EndReason.Capture: "상대 돌 포획",
            engine.EndReason.Suicide: "자충수",
            engine.EndReason.TwoPasses: "연속 패스",
        }
        print(f"승자: {player_name(result.winner)} ({reasons[result.reason]})")
    elif args.self_play:
        raise RuntimeError("대국이 최대 수순 안에 종료되지 않았습니다.")
    else:
        print(f"추천 수 적용 완료. 다음 차례: {player_name(game.to_play)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
