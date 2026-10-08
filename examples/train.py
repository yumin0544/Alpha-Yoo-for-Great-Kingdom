"""Run or resume neural self-play reinforcement learning from the repository."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

# Use this checkout's logging/config changes even if an older wheel is installed.
# The native my_board_engine extension still comes from the existing environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kingdom_ai import PolicyValueNet, Trainer, TrainingConfig, load_model


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return number


def main():
    parser = argparse.ArgumentParser(description="자가 대국·자료 버퍼·학습·평가·재개")
    parser.add_argument("--iterations", type=positive_integer, default=1,
                        help="이번 실행에서 추가로 진행할 반복 수 (기본 1)")
    parser.add_argument("--output", type=Path, default=Path("runs/rl"),
                        help="latest.pt, best.pt, candidate.pt, metrics.jsonl 저장 폴더")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="전체 학습 체크포인트 latest.pt")
    source.add_argument("--initial-model", type=Path, help="새 학습에 사용할 기존 모델 가중치")
    parser.add_argument("--reconfigure", action="store_true",
                        help="--resume과 새 --output으로 설정 변경; 진행·optimizer·버퍼 유지")
    parser.add_argument("--prepare-only", action="store_true",
                        help="모델·설정·버퍼를 저장하고 대국을 시작하지 않음")
    parser.add_argument("--refresh-promotion-archive", action="store_true",
                        help="--resume --reconfigure로 새 과거 챔피언만 추가하고 평가 캐시 초기화")
    parser.add_argument("--device", default="cpu", help="cpu 또는 사용 가능한 cuda 장치")
    parser.add_argument("--self-play-backend", choices=("cpu", "cuda"),
                        help="자가 대국 규칙·탐색 장치 (기본 cpu); cuda는 --device cuda 필요")
    parser.add_argument("--self-play-model", choices=("champion", "learner"),
                        help="자가 대국 모델; 새 학습 기본 champion, 재개 시 저장값 복원")
    parser.add_argument("--augment-symmetries", action=argparse.BooleanOptionalAction, default=None,
                        help="학습 위치·정책에 8가지 회전/반사 중 하나를 무작위 적용")
    parser.add_argument("--self-play-tactical-checks", action=argparse.BooleanOptionalAction,
                        default=None, help="CUDA 루트에서 즉시 승리·패배와 한 수 포획 위협 확인")
    parser.add_argument("--self-play-fpu-reduction", type=float,
                        help="CUDA 미방문 수 가치=신경망 가치-지정값 (미지정은 기존 Q=0)")
    parser.add_argument("--online-tactics", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="매 사이클 새 자가 대국 위치의 깊은 승패 증명과 전술 혼합 학습")
    parser.add_argument("--online-tactics-include-loss", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="새 LOSS 증명을 가치 전용 자료로 수집 (기본 꺼짐; 기존 자료 보존)")
    parser.add_argument("--online-tactics-max-cases", type=positive_integer, default=None,
                        help="사이클당 CPU 전술 탐색 위치 상한 (새 학습 기본 32)")
    parser.add_argument("--online-tactics-max-depth", type=positive_integer, default=None,
                        help="양쪽 착수를 합친 전술 증명 최대 깊이 (새 학습 기본 20수; 재개는 저장값)")
    parser.add_argument("--online-tactics-max-nodes", type=positive_integer, default=None,
                        help="위치당 전술 증명 노드 상한 (새 학습 기본 2000000)")
    parser.add_argument("--online-tactics-time-limit-ms", type=positive_integer, default=None,
                        help="위치당 CPU 전술 증명 시간 예산 ms (새 학습 기본 2000)")
    parser.add_argument("--online-tactics-generation-seconds", type=float, default=None,
                        help="사이클당 새 전술 자료 생성 시간 예산 (새 학습 기본 30초)")
    parser.add_argument("--online-tactics-fraction", type=float, default=None,
                        help="증명 버퍼가 있을 때 미니배치 전술 비율 (새 학습 기본 0.25)")
    parser.add_argument("--online-tactics-replay-capacity", type=positive_integer, default=None,
                        help="일반 replay와 분리된 증명 위치 FIFO 용량 (새 학습 기본 1024)")
    parser.add_argument("--online-tactics-min-proof-depth", type=positive_integer, default=None,
                        help="혼합 학습에 채택할 최소 증명 수순 길이 (새 학습 기본 3수)")
    parser.add_argument("--temperature-moves", type=int,
                        help="자가 대국 초반 몇 수까지 기존 온도 사용; 이후 --final-temperature")
    parser.add_argument("--threads", type=positive_integer, default=1,
                        help="이 실행의 PyTorch CPU 연산 스레드 수 (기본 1)")
    parser.add_argument("--promotion-archive-dir", type=str,
                        help="1차 통과 후 비교할 champion_<버전>v/best.pt 폴더; 읽기 전용")
    parser.add_argument("--promotion-archive-games", type=positive_integer,
                        help="후보와 현 챔피언이 각각 과거 버전마다 두는 판수 (새 학습 기본 100)")
    parser.add_argument("--evaluation-workers", type=positive_integer,
                        help="평가 병렬 worker 수; 새 학습 기본 1, 재개 시 저장값 복원")
    parser.add_argument("--evaluation-backend", choices=("legacy", "batched_cpp"),
                        help="평가 탐색 경로; batched_cpp는 C++ leaf 배치·입력 생성·트리 재사용")
    parser.add_argument("--evaluation-leaf-batch-size", type=positive_integer,
                        help="한 탐색에서 동시에 준비할 leaf 수 (새 학습 기본 8)")
    parser.add_argument("--evaluation-reuse-tree", action=argparse.BooleanOptionalAction,
                        default=None, help="batched_cpp 평가에서 실제 착수 후 하위 트리 재사용")
    parser.add_argument("--channels", type=positive_integer, help="새 모델 채널 수 (기본 32)")
    parser.add_argument("--residual-blocks", type=int, help="새 모델 잔차 블록 수 (기본 2)")
    integer_flags = {
        "games-per-iteration": "games_per_iteration", "simulations": "simulations",
        "replay-capacity": "replay_capacity", "batch-size": "batch_size",
        "train-steps": "train_steps_per_iteration", "eval-games": "evaluation_games",
        "eval-simulations": "evaluation_simulations",
        "self-play-batch-size": "self_play_batch_size",
    }
    real_flags = {
        "c-puct": "c_puct", "dirichlet-alpha": "dirichlet_alpha",
        "dirichlet-epsilon": "dirichlet_epsilon", "temperature": "temperature",
        "learning-rate": "learning_rate", "weight-decay": "weight_decay",
        "promotion-threshold": "promotion_threshold",
        "eval-opening-temperature": "evaluation_opening_temperature",
        "final-temperature": "final_temperature",
    }
    for flag, field in integer_flags.items():
        parser.add_argument(
            "--" + flag, dest=field, type=positive_integer,
            help=("한 사이클 갱신 수; 재개 변경은 --reconfigure와 새 --output 필요"
                  if flag == "train-steps" else None),
        )
    for flag, field in real_flags.items():
        parser.add_argument("--" + flag, dest=field, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-opening-moves", dest="evaluation_opening_moves", type=int)
    args = parser.parse_args()
    if any(part.casefold() == "past-version" for part in args.output.resolve().parts):
        parser.error("past-version은 읽기 전용입니다. 결과는 별도 --output 폴더에 저장하세요.")
    config_names = set(asdict(TrainingConfig()))
    overrides = {name: getattr(args, name) for name in config_names
                 if getattr(args, name) is not None}
    if args.reconfigure and not args.resume:
        parser.error("--reconfigure에는 --resume이 필요합니다.")
    if args.refresh_promotion_archive and not (args.resume and args.reconfigure):
        parser.error("--refresh-promotion-archive에는 --resume --reconfigure와 새 --output이 필요합니다.")
    if args.resume and (args.channels is not None or args.residual_blocks is not None):
        parser.error("재개 시 모델 구성을 변경할 수 없습니다.")
    if args.resume and overrides and not args.reconfigure:
        parser.error("재개 시 학습 설정과 모델 구성은 체크포인트에서 복원하므로 변경할 수 없습니다.")
    if args.initial_model and (args.channels is not None or args.residual_blocks is not None):
        parser.error("기존 모델을 읽을 때 채널 수와 잔차 블록 수를 지정할 수 없습니다.")
    latest = args.output / "latest.pt"
    best = args.output / "best.pt"
    candidate = args.output / "candidate.pt"
    metrics = args.output / "metrics.jsonl"
    artifacts = (latest, best, candidate, metrics)
    if args.reconfigure and (
            any(path.exists() for path in artifacts)
            or latest.resolve() == args.resume.resolve()):
        parser.error("설정 변경 시 기존 기록을 보존하도록 새 --output 폴더를 사용하세요.")
    if not args.resume and any(path.exists() for path in artifacts):
        parser.error("출력 폴더에 기존 학습 기록이 있습니다. --resume 또는 새 --output을 사용하세요.")
    if args.resume and latest.exists() and latest.resolve() != args.resume.resolve():
        parser.error("다른 학습의 latest.pt가 있는 출력 폴더입니다. 새 --output을 사용하세요.")
    if args.resume and not latest.exists() and any(path.exists() for path in (best, candidate, metrics)):
        parser.error("출력 폴더에 재개 파일 없는 학습 기록이 있습니다. 새 --output을 사용하세요.")
    torch.set_num_threads(args.threads)
    try:
        if args.resume:
            trainer = Trainer.load_checkpoint(
                args.resume, device=args.device,
                evaluation_workers=args.evaluation_workers,
                evaluation_backend=args.evaluation_backend,
                evaluation_leaf_batch_size=args.evaluation_leaf_batch_size,
                evaluation_reuse_tree=args.evaluation_reuse_tree,
                **({"refresh_promotion_archive": True} if args.refresh_promotion_archive else {}),
            )
            if args.reconfigure:
                trainer.reconfigure(**overrides)
        else:
            config = TrainingConfig(**overrides)
            if args.initial_model:
                model = load_model(args.initial_model, device="cpu")
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(config.seed)
                    model = PolicyValueNet(
                        channels=32 if args.channels is None else args.channels,
                        residual_blocks=2 if args.residual_blocks is None else args.residual_blocks,
                    )
            trainer = Trainer(
                config, model=model, device=args.device,
                evaluation_workers=(1 if args.evaluation_workers is None
                                    else args.evaluation_workers),
                evaluation_backend=args.evaluation_backend or "legacy",
                evaluation_leaf_batch_size=(8 if args.evaluation_leaf_batch_size is None
                                            else args.evaluation_leaf_batch_size),
                evaluation_reuse_tree=(True if args.evaluation_reuse_tree is None
                                       else args.evaluation_reuse_tree),
            )
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.error(str(error))
    if (trainer.config.promotion_archive_dir is not None
            and args.output.resolve().is_relative_to(Path(trainer.config.promotion_archive_dir))):
        parser.error("승격 평가 원본 폴더에는 결과를 저장할 수 없습니다. 별도 --output을 사용하세요.")
    print(
        f"시작: 완료 반복 {trainer.iteration}, 자가 대국 {trainer.self_play_games}판, "
        f"자료 생성 모델 {trainer.config.self_play_model}, "
        f"평가 {trainer.evaluation_backend}, worker {trainer.evaluation_workers}개, "
        f"leaf 배치 {trainer.evaluation_leaf_batch_size}, 트리 재사용 {trainer.evaluation_reuse_tree}"
    )
    print(json.dumps(asdict(trainer.config), ensure_ascii=False))
    print("평가 승률은 현재 기준 모델에 대한 결과이며, 절대 기력은 별도 검증해야 합니다.")
    if trainer.config.promotion_archive_dir is not None:
        print(f"승격: 현 챔피언 상대 {trainer.config.promotion_threshold:.0%} 이상 → "
              f"과거 버전마다 두 모델 각각 {trainer.config.promotion_archive_games}판 → "
              "후보의 평균 승률이 더 높을 때만 교체 (동률 유지)", flush=True)
        if args.refresh_promotion_archive:
            print("과거 상대 목록을 갱신했습니다. 새 버전이 추가되면 기존 평가 캐시는 초기화됩니다.", flush=True)

    def report(row):
        trainer.export_champion(best)
        trainer.export_learner(candidate)
        evaluation = row["evaluation"]
        print(
            f"반복 {row['iteration']}: 자가 대국 누적 {row['self_play_games']}판, "
            f"버퍼 {row['replay_size']}개, 손실 {row['loss']:.5f}, "
            f"평가 {evaluation['wins']}/{evaluation['games']} "
            f"({evaluation['win_rate']:.1%}), 승격 {row['promoted']}, "
            f"자가 대국 {row['self_play_backend']} "
            f"{row['self_play_games_per_second']:.2f}판/초, "
            f"전체 반복 {row['iteration_games_per_second']:.2f}판/초, "
            f"위치 {row['self_play_positions_per_second']:.1f}개/초, "
            f"자가 대국 계산 {row['self_play_compute_seconds']:.2f}초, "
            f"버퍼 저장 {row['replay_store_seconds']:.2f}초, "
            f"학습 {row['training_seconds']:.2f}초, "
            f"평가 {row['evaluation_seconds']:.2f}초, "
            f"체크포인트 {row['checkpoint_seconds']:.2f}초, "
            f"경과 {row['elapsed_with_checkpoint_seconds']:.2f}초", flush=True,
        )
        online_tactics = row.get("online_tactics", {})
        league = row.get("promotion_league", {})
        if league.get("evaluated", False):
            print(f"과거 버전 비교: 후보 {league['candidate_mean_win_rate']:.1%}, "
                  f"현 챔피언 {league['reference_mean_win_rate']:.1%}, "
                  f"상대 {league['opponent_count']}개, 실제 추가 대국 {league['actual_games_played']}판, "
                  f"현 챔피언 결과 재사용 {league['reference_cache_hit']}, "
                  f"{league['seconds']:.2f}초", flush=True)
            for result in league["results"]:
                print(f"  champion_{result['version']}v: "
                      f"후보 {result['candidate']['wins']}/{result['candidate']['games']} "
                      f"({result['candidate']['win_rate']:.1%}), "
                      f"현 챔피언 {result['reference']['wins']}/{result['reference']['games']} "
                      f"({result['reference']['win_rate']:.1%})", flush=True)
        if online_tactics.get("enabled", False):
            print(
                f"온라인 전술: 새 증명 {online_tactics['added_samples']}개, "
                f"증명 버퍼 {online_tactics['replay_size']}개, "
                f"혼합 갱신 {online_tactics['mixed_updates']}회, "
                f"미니배치 전술 {online_tactics['tactical_rows_per_batch']}개·"
                f"일반 {online_tactics['replay_rows_per_batch']}개, "
                f"teacher {online_tactics['seconds']:.2f}초", flush=True,
            )

    try:
        if not args.prepare_only:
            trainer.run(args.iterations, checkpoint_path=latest, metrics_path=metrics,
                        on_iteration=report, collect_metrics=False)
        else:
            trainer.save_checkpoint(latest)
            trainer.export_champion(best)
            trainer.export_learner(candidate)
            print("설정과 학습 상태를 저장했습니다. 추가 학습은 시작하지 않았습니다.")
    except KeyboardInterrupt:
        if latest.exists():
            print(f"\n중단했습니다. 마지막 완료·저장한 반복부터 재개: {latest.resolve()}")
            print("진행 중이던 반복의 대국과 학습은 체크포인트를 읽은 뒤 다시 실행합니다.")
        else:
            print("\n초기 체크포인트 저장 전에 중단했습니다. 같은 명령으로 새로 시작하세요.")
        return 130
    print(f"학습 재개 파일: {latest.resolve()}")
    print(f"기준 모델 파일: {best.resolve()}")
    print(f"학습 후보 파일 (승격과 별개): {candidate.resolve()}")
    print(f"기록 파일{' (첫 반복 완료 시 생성)' if args.prepare_only else ''}: {metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
