"""두 저장 모델의 흑백 교대 대결과 결과 표시."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

# Run new repository code with the already-installed C++ extension.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import my_board_engine as engine
import torch

from kingdom_ai.checkpoint import load_model
from kingdom_ai.match import MatchOptions, board_rows, play_match, series_ratings
from kingdom_ai.model import PolicyValueNet


REASONS = {"Capture": "상대 돌 포획", "Suicide": "자충수", "TwoPasses": "연속 패스"}
COLORS = {"Black": "흑/선공", "White": "백/후공"}


class MatchCancelled(Exception):
    """Cooperative UI stop, checked after each completed search."""


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    digest.update(json.dumps(model.model_config, sort_keys=True).encode("utf-8"))
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_participant(path, device):
    path = path.resolve()
    before = file_digest(path)
    try:
        model = load_model(path, device=device)
    except (ValueError, KeyError) as error:
        raise ValueError(f"{path.name}: {error}. best.pt 또는 save_model로 저장한 추론용 "
                         "모델이 필요합니다. latest.pt 전체 학습 저장은 지원하지 않습니다.") from error
    if file_digest(path) != before:
        raise RuntimeError(f"읽는 동안 모델 파일이 변경됐습니다: {path}")
    return model, {"path": str(path), "checkpoint_sha256": before,
                   "weights_sha256": model_digest(model), "model_config": model.model_config}


def print_board(rows):
    print("     1 2 3 4 5 6 7 8 9")
    for row, points in enumerate(rows, 1):
        print(f"{row:2} | " + " ".join(points))
    print("x=흑, o=백, #=중립 돌, B/W=확정 집, .=빈칸", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, help="A의 추론용 best.pt")
    parser.add_argument("--model-b", type=Path, help="B의 추론용 best.pt")
    parser.add_argument("--demo", action="store_true", help="미학습 소형 모델 두 개로 실행 확인")
    parser.add_argument("--name-a", default="모델 A")
    parser.add_argument("--name-b", default="모델 B")
    parser.add_argument("--games", type=int, default=20, help="흑백 교대를 위한 짝수 판수")
    parser.add_argument("--simulations", type=int, default=128, help="양쪽의 수당 동일 탐색 횟수")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-moves", type=int, default=6)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--tactical-checks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cpu", help="cpu 또는 cuda[:장치 번호] 신경망 추론")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="매 판 기록을 저장할 새 JSONL 파일")
    parser.add_argument("--progress-file", type=Path, help="UI용 현재 판·수 진행 정보")
    parser.add_argument("--stop-file", type=Path, help="이 파일이 생기면 현재 탐색 후 중단")
    parser.add_argument("--show-board", action="store_true", help="매 판 최종 보드 표시")
    parser.add_argument("--watch", action="store_true", help="매 수와 진행 보드 표시")
    parser.add_argument("--rating-a", type=float, default=1500.0)
    parser.add_argument("--rating-b", type=float, default=1500.0)
    parser.add_argument("--k", type=float, default=32.0, help="완료 시리즈 전체의 레이팅 갱신 가중치")
    args = parser.parse_args(argv)
    if args.demo:
        if args.model_a is not None or args.model_b is not None:
            parser.error("--demo와 모델 파일은 함께 지정할 수 없습니다.")
    elif args.model_a is None or args.model_b is None:
        parser.error("--model-a와 --model-b를 모두 지정하거나 --demo를 사용하세요.")
    if not args.name_a.strip() or not args.name_b.strip() or args.name_a == args.name_b:
        parser.error("두 모델의 표시 이름은 비어 있지 않고 서로 달라야 합니다.")
    if args.threads < 1:
        parser.error("--threads는 1 이상이어야 합니다.")
    if args.output is not None and args.output.exists():
        parser.error("결과 파일이 이미 있습니다. 새 --output 경로를 사용하세요.")
    if args.progress_file is not None and args.progress_file.exists():
        parser.error("진행 파일이 이미 있습니다. 새 --progress-file 경로를 사용하세요.")
    try:
        options = MatchOptions(games=args.games, simulations=args.simulations,
                               c_puct=args.c_puct, seed=args.seed,
                               opening_moves=args.opening_moves,
                               opening_temperature=args.opening_temperature,
                               tactical_checks=args.tactical_checks)
        series_ratings(args.rating_a, args.rating_b, 0, args.games, args.k)
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("--device는 cpu 또는 cuda여야 합니다.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("사용 가능한 CUDA 장치가 없습니다.")
    except (ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    output = None
    completed = 0
    terminal_complete = False
    started = None
    names = {"a": args.name_a, "b": args.name_b}
    try:
        if args.demo:
            models, identities = [], []
            with torch.random.fork_rng(devices=[]):
                for seed in (11, 29):
                    torch.manual_seed(seed)
                    model = PolicyValueNet(channels=4, residual_blocks=0).to(device).eval()
                    models.append(model)
                    identities.append({"source": "untrained_demo", "initialization_seed": seed,
                                       "model_config": model.model_config,
                                       "weights_sha256": model_digest(model)})
            print("데모: 미학습 모델 두 개로 프로그램 동작을 확인합니다.", flush=True)
        else:
            print("두 모델을 읽는 중입니다...", flush=True)
            left, left_identity = load_participant(args.model_a, device)
            right, right_identity = load_participant(args.model_b, device)
            models, identities = [left, right], [left_identity, right_identity]
        for key, identity in zip(("a", "b"), identities):
            identity.update(id=key, name=names[key])
        session = {
            "type": "session", "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "participants": identities,
            "protocol": {**asdict(options), "device": str(device), "backend": "legacy",
                         "workers": 1, "threads": args.threads, "time_limit_ms": 0,
                         "dirichlet_epsilon": 0.0,
                         "rules": {"board_size": 9, "neutral": [4, 4],
                                   "suicide": "Loses", "stones_per_player": 41,
                                   "own_territory_moves": False, "single_edge_territory": True,
                                   "black_territory_margin": 3},
                         "initial_ratings": {"a": args.rating_a, "b": args.rating_b},
                         "k_per_series": args.k},
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(),
                        "engine_sha256": file_digest(engine.__file__)},
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("x", encoding="utf-8")

        def emit(record):
            if output is not None:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()

        emit(session)
        print(f"{args.name_a} vs {args.name_b} | {args.games}판 | "
              f"수당 {args.simulations}회 탐색 | {device}", flush=True)
        active_index = None
        last_progress = 0.0
        current_plies = 0

        def on_move(index, actor, move, state):
            nonlocal active_index, last_progress, current_plies
            if args.stop_file is not None and args.stop_file.exists():
                raise MatchCancelled("사용자 중단")
            now = perf_counter()
            current_plies = current_plies + 1 if index == active_index else 1
            if args.progress_file is not None and (index != active_index or now - last_progress >= 1):
                args.progress_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.progress_file.with_suffix(".tmp")
                temporary.write_text(json.dumps({"index": index, "plies": current_plies,
                                                "elapsed_seconds": now - started}), encoding="utf-8")
                try:
                    os.replace(temporary, args.progress_file)
                except PermissionError:
                    # On Windows a reader can briefly prevent replacement.
                    # Progress may skip an update; durable game results do not.
                    temporary.unlink(missing_ok=True)
                last_progress = now
            if active_index != index:
                active_index = index
                black = args.name_a if index % 2 else args.name_b
                white = args.name_b if index % 2 else args.name_a
                print(f"[{index}/{args.games}] 대국 시작: 흑 {black} / 백 {white}", flush=True)
            if args.watch:
                move_text = "패스" if move.is_pass() else f"{move.point.row + 1}행 {move.point.col + 1}열"
                print(f"{COLORS[actor.name]}: {move_text}")
                print_board(board_rows(state))

        def on_game(record):
            nonlocal completed
            emit({"type": "game", **record})
            completed += 1
            print(f"[{record['index']}/{args.games}] {names[record['winner']]} 승리 "
                  f"({COLORS[record['winner_color']]}, {REASONS[record['reason']]}, "
                  f"{record['plies']}수) | 집 흑 {record['territory']['black']} / "
                  f"백 {record['territory']['white']}", flush=True)
            if args.show_board:
                print_board(record["board"])

        started = perf_counter()
        result = play_match(*models, options, on_game=on_game, on_move=on_move)
        ratings = series_ratings(args.rating_a, args.rating_b, result["wins_a"], args.games, args.k)
        summary = {key: value for key, value in result.items() if key != "records"}
        emit({"type": "summary", "status": "complete", **summary,
              "ratings": ratings, "elapsed_seconds": perf_counter() - started})
        terminal_complete = True
        print("\n대결 완료", flush=True)
        for key in ("a", "b"):
            wins = result["wins_" + key]
            black_wins = result["wins_a_as_black"] if key == "a" else args.games // 2 - result["wins_a_as_white"]
            white_wins = result["wins_a_as_white"] if key == "a" else args.games // 2 - result["wins_a_as_black"]
            print(f"{names[key]}: {wins}승 {args.games - wins}패, 승률 {wins / args.games:.1%} "
                  f"(흑 {black_wins}승 {black_wins / (args.games // 2):.1%} / "
                  f"백 {white_wins}승 {white_wins / (args.games // 2):.1%})")
        print("종료 사유: " + ", ".join(f"{REASONS[key]} {value}판" for key, value in result["endings"].items()))
        print(f"내부 시리즈 레이팅 (전체 대결에 K={args.k:g} 1회 적용): "
              f"{args.name_a} {ratings['before']['a']:.1f} → {ratings['after']['a']:.1f}, "
              f"{args.name_b} {ratings['before']['b']:.1f} → {ratings['after']['b']:.1f}")
        if args.output is not None:
            print(f"대국 기록: {args.output.resolve()}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        cancelled = isinstance(error, (KeyboardInterrupt, MatchCancelled))
        message = "사용자 중단" if cancelled else str(error)
        if terminal_complete:
            notice = f"대결은 완료됐지만 결과 표시가 중단됐습니다: {message}."
            if output is not None:
                notice += " 저장된 완료 결과와 레이팅은 유효합니다."
            print(notice, file=sys.stderr)
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        if output is not None:
            try:
                output.write(json.dumps({"type": "aborted", "status": "cancelled" if cancelled else "failed",
                                         "completed_games": completed, "error": message,
                                         "elapsed_seconds": perf_counter() - started if started is not None else 0.0},
                                        ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
            except OSError:
                pass
        print(f"대결 중단: {message}. 완료 {completed}판; 레이팅은 적용하지 않습니다.", file=sys.stderr)
        return 130 if cancelled else 1
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
