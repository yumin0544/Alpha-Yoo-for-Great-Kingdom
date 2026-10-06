"""Play a saved policy/value model as a human, using the verified game rules."""

import argparse
from pathlib import Path
from time import perf_counter

import my_board_engine as engine
import torch

from kingdom_ai import (
    GpuPUCT, GpuPUCTOptions, GpuStateBatch, PUCT, PUCTOptions,
    action_to_move, load_model,
)
from kingdom_ai.tactics import analyze_tactics, select_tactical_move


HELP = (
    "행 열: 3 4 -> 3행 4열 (1~9) | pass 또는 패스 | help 또는 도움말 | q 또는 종료\n"
    "x=선공 흑/파랑, o=후공 백/주황, #=중립 돌, B/W=완성된 집, .=빈칸\n"
    "완성된 집에는 누구도 놓을 수 없습니다. 포획은 즉시 승리, 자충수는 즉시 패배입니다."
)
ERRORS = {
    "Occupied": "이미 돌이 있는 칸입니다.",
    "OwnTerritory": "자기 집 안에는 놓을 수 없습니다.",
    "OpponentTerritory": "상대 집 안에는 놓을 수 없습니다.",
    "NoStones": "남은 돌이 없습니다. 패스해 주세요.",
    "OutOfBounds": "행과 열은 1~9 사이여야 합니다.",
    "GameOver": "이미 끝난 대국입니다.",
}
REASONS = {"Capture": "상대 돌 포획", "Suicide": "자충수", "TwoPasses": "연속 패스"}
QUIT = {"q", "quit", "exit", "종료"}


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return number


def player_name(player):
    return "선공 흑/파랑" if player == engine.Cell.Black else "후공 백/주황"


def move_text(move):
    return "패스" if move.is_pass() else f"{move.point.row + 1}행 {move.point.col + 1}열"


def parse_move(command):
    if command in ("pass", "패스"):
        return engine.Move.pass_turn()
    parts = command.split()
    if len(parts) != 2:
        raise ValueError("행과 열을 입력하세요. 예: 3 4")
    try:
        row, col = map(int, parts)
    except ValueError:
        raise ValueError("행과 열은 1~9 사이의 정수여야 합니다.") from None
    if not 1 <= row <= 9 or not 1 <= col <= 9:
        raise ValueError("행과 열은 1~9 사이여야 합니다.")
    return engine.Move.place(row - 1, col - 1)


def print_board(game):
    cells, owners = game.board.cells, game.ownership
    stones = {engine.Cell.Black: "x", engine.Cell.White: "o", engine.Cell.Neutral: "#"}
    territories = {engine.Cell.Black: "B", engine.Cell.White: "W"}
    print("\n     1 2 3 4 5 6 7 8 9")
    for row in range(9):
        points = [stones.get(cells[row * 9 + col],
                             territories.get(owners[row * 9 + col], ".")) for col in range(9)]
        print(f"{row + 1:2} | " + " ".join(points))
    score = game.score()
    print(f"남은 돌: 흑 {game.remaining_stones(engine.Cell.Black)} / "
          f"백 {game.remaining_stones(engine.Cell.White)} | 집: 흑 {score.black} / 백 {score.white}")


class Opponent:
    def __init__(self, model, device, simulations, seed, *, tactical_checks=True,
                 fpu_reduction=0.0):
        self.device = torch.device(device)
        if type(tactical_checks) is not bool:
            raise TypeError("tactical_checks must be a bool")
        self.tactical_checks = tactical_checks
        if self.device.type == "cuda":
            self.searcher = GpuPUCT(model, GpuPUCTOptions(
                simulations=simulations, seed=seed, dirichlet_epsilon=0.0,
                tactical_checks=tactical_checks, fpu_reduction=fpu_reduction,
            ), device=self.device)
        elif self.device.type == "cpu":
            self.searcher = PUCT(model, PUCTOptions(
                simulations=simulations, seed=seed, dirichlet_epsilon=0.0,
            ), device=self.device)
        else:
            raise ValueError("대국 장치는 cpu 또는 cuda여야 합니다.")

    def choose(self, game):
        if self.device.type == "cuda":
            result = self.searcher.search(GpuStateBatch.from_engine([game], device=self.device))
            return action_to_move(int(result.actions[0].item()))
        choices = analyze_tactics(game) if self.tactical_checks else None
        if choices is not None and choices.winning_actions:
            return action_to_move(min(choices.winning_actions))
        result = self.searcher.search(game)
        move = select_tactical_move(result, choices) if choices is not None else result.best_move
        if move is None:
            raise RuntimeError("진행 중인 대국에서 AI가 수를 반환하지 않았습니다.")
        return move


def play(opponent, human):
    print(HELP)
    print(f"사람: {player_name(human)} | AI: "
          f"{player_name(engine.Cell.White if human == engine.Cell.Black else engine.Cell.Black)}")
    while True:
        game = engine.State()
        print_board(game)
        while not game.result.finished():
            human_turn = game.to_play == human
            if human_turn:
                try:
                    command = input("내 수 > ").strip().lower()
                except EOFError:
                    return 0
                if command in QUIT:
                    return 0
                if command in ("help", "도움말"):
                    print(HELP)
                    continue
                try:
                    move = parse_move(command)
                except ValueError as error:
                    print(error)
                    continue
            else:
                print("AI가 생각 중입니다...", flush=True)
                started = perf_counter()
                move = opponent.choose(game)
                print(f"AI: {move_text(move)} ({perf_counter() - started:.2f}초)")
            outcome = game.play(move)
            if not outcome.accepted():
                if not human_turn:
                    raise RuntimeError(f"AI 착수가 거부됐습니다: {outcome.error.name}")
                print(ERRORS.get(outcome.error.name, outcome.error.name))
                continue
            print_board(game)
        winner = "사람" if game.result.winner == human else "AI"
        print(f"\n대국 종료: {winner} 승리, {player_name(game.result.winner)} "
              f"({REASONS[game.result.reason.name]})")
        while True:
            try:
                command = input("r: 새 대국 | q: 종료 > ").strip().lower()
            except EOFError:
                return 0
            if command in QUIT:
                return 0
            if command in ("r", "restart", "다시"):
                break


def main():
    parser = argparse.ArgumentParser(description="저장한 학습 모델과 사람 대 AI 대국")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="대국용 기준 모델 best.pt (학습 재개 파일 latest.pt와 다릅니다)")
    parser.add_argument("--human", choices=("black", "white"), default="black",
                        help="사람의 색: black 선공, white 후공 (기본 black)")
    parser.add_argument("--device", default="cpu", help="cpu 또는 cuda, cuda:0 (기본 cpu)")
    parser.add_argument("--simulations", type=positive_integer, default=128,
                        help="AI의 한 수당 탐색 횟수 (기본 128)")
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tactical-checks", action=argparse.BooleanOptionalAction, default=True,
                        help="즉시 승리와 상대의 다음 한 수 승리를 검사 (기본 켬)")
    parser.add_argument("--fpu-reduction", type=float, default=0.0,
                        help="GPU 미방문 수의 초기 평가 감소량 (기본 0.0)")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 64:
        parser.error("시드는 0 이상, 2^64 미만이어야 합니다.")
    torch.set_num_threads(args.threads)
    try:
        model = load_model(args.checkpoint, device="cpu")
        opponent = Opponent(model, args.device, args.simulations, args.seed,
                            tactical_checks=args.tactical_checks,
                            fpu_reduction=args.fpu_reduction)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.error(f"모델을 읽거나 대국을 준비하지 못했습니다: {error}")
    print(f"대국 모델: {args.checkpoint.resolve()}")
    human = engine.Cell.Black if args.human == "black" else engine.Cell.White
    try:
        return play(opponent, human)
    except KeyboardInterrupt:
        print("\n대국을 종료합니다.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
