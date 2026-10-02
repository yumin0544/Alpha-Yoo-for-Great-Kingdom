"""Run or resume neural self-play reinforcement learning from the repository."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

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
                        help="latest.pt, best.pt, metrics.jsonl 저장 폴더")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="전체 학습 체크포인트 latest.pt")
    source.add_argument("--initial-model", type=Path, help="새 학습에 사용할 기존 모델 가중치")
    parser.add_argument("--device", default="cpu", help="cpu 또는 사용 가능한 cuda 장치")
    parser.add_argument("--threads", type=positive_integer, default=1,
                        help="이 실행의 PyTorch CPU 연산 스레드 수 (기본 1)")
    parser.add_argument("--channels", type=positive_integer, help="새 모델 채널 수 (기본 32)")
    parser.add_argument("--residual-blocks", type=int, help="새 모델 잔차 블록 수 (기본 2)")
    integer_flags = {
        "games-per-iteration": "games_per_iteration", "simulations": "simulations",
        "replay-capacity": "replay_capacity", "batch-size": "batch_size",
        "train-steps": "train_steps_per_iteration", "eval-games": "evaluation_games",
        "eval-simulations": "evaluation_simulations",
    }
    real_flags = {
        "c-puct": "c_puct", "dirichlet-alpha": "dirichlet_alpha",
        "dirichlet-epsilon": "dirichlet_epsilon", "temperature": "temperature",
        "learning-rate": "learning_rate", "weight-decay": "weight_decay",
        "promotion-threshold": "promotion_threshold",
        "eval-opening-temperature": "evaluation_opening_temperature",
    }
    for flag, field in integer_flags.items():
        parser.add_argument("--" + flag, dest=field, type=positive_integer)
    for flag, field in real_flags.items():
        parser.add_argument("--" + flag, dest=field, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-opening-moves", dest="evaluation_opening_moves", type=int)
    args = parser.parse_args()
    config_names = set(asdict(TrainingConfig()))
    overrides = {name: getattr(args, name) for name in config_names
                 if getattr(args, name) is not None}
    if args.resume and (overrides or args.channels is not None or args.residual_blocks is not None):
        parser.error("재개 시 학습 설정과 모델 구성은 체크포인트에서 복원하므로 변경할 수 없습니다.")
    if args.initial_model and (args.channels is not None or args.residual_blocks is not None):
        parser.error("기존 모델을 읽을 때 채널 수와 잔차 블록 수를 지정할 수 없습니다.")
    latest = args.output / "latest.pt"
    best = args.output / "best.pt"
    metrics = args.output / "metrics.jsonl"
    if not args.resume and any(path.exists() for path in (latest, best, metrics)):
        parser.error("출력 폴더에 기존 학습 기록이 있습니다. --resume 또는 새 --output을 사용하세요.")
    if args.resume and latest.exists() and latest.resolve() != args.resume.resolve():
        parser.error("다른 학습의 latest.pt가 있는 출력 폴더입니다. 새 --output을 사용하세요.")
    if args.resume and not latest.exists() and (best.exists() or metrics.exists()):
        parser.error("출력 폴더에 재개 파일 없는 학습 기록이 있습니다. 새 --output을 사용하세요.")
    torch.set_num_threads(args.threads)
    try:
        if args.resume:
            trainer = Trainer.load_checkpoint(args.resume, device=args.device)
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
            trainer = Trainer(config, model=model, device=args.device)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        parser.error(str(error))
    print(f"시작: 완료 반복 {trainer.iteration}, 자가 대국 {trainer.self_play_games}판")
    print(json.dumps(asdict(trainer.config), ensure_ascii=False))
    print("평가 승률은 현재 기준 모델에 대한 결과이며, 절대 기력은 별도 검증해야 합니다.")

    def report(row):
        trainer.export_champion(best)
        evaluation = row["evaluation"]
        print(
            f"반복 {row['iteration']}: 자가 대국 누적 {row['self_play_games']}판, "
            f"버퍼 {row['replay_size']}개, 손실 {row['loss']:.5f}, "
            f"평가 {evaluation['wins']}/{evaluation['games']} "
            f"({evaluation['win_rate']:.1%}), 승격 {row['promoted']}, "
            f"경과 {row['elapsed_seconds']:.2f}초", flush=True,
        )

    try:
        trainer.save_checkpoint(latest)
        trainer.export_champion(best)
        trainer.run(args.iterations, checkpoint_path=latest, metrics_path=metrics,
                    on_iteration=report, collect_metrics=False)
    except KeyboardInterrupt:
        print(f"\n중단했습니다. 마지막 완료·저장한 반복부터 재개: {latest.resolve()}")
        print("진행 중이던 반복의 대국과 학습은 체크포인트를 읽은 뒤 다시 실행합니다.")
        return 130
    print(f"학습 재개 파일: {latest.resolve()}")
    print(f"기준 모델 파일: {best.resolve()}")
    print(f"기록 파일: {metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
