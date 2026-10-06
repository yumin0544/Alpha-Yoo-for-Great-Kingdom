"""Summarize Trainer metrics.jsonl throughput, bottlenecks and color results."""

import argparse
import json
from pathlib import Path

from kingdom_ai.metrics import load_metric_rows, summarize_metrics


def duration(seconds):
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}시간 {minutes}분 {seconds}초"


def percent(value):
    return "N/A" if value is None else f"{value:.1%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Trainer metrics.jsonl")
    parser.add_argument("--recent", type=int, default=20, help="최근 요약 반복 수 (기본 20)")
    parser.add_argument("--target-iterations", type=int, help="남은 시간 추정용 목표 반복")
    parser.add_argument("--json", action="store_true", help="사람용 요약 대신 JSON 출력")
    args = parser.parse_args()
    try:
        rows, duplicates = load_metric_rows(args.metrics)
        summary = summarize_metrics(
            rows, recent=args.recent, target_iterations=args.target_iterations,
            duplicate_rows=duplicates,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    overall, recent = summary["overall"], summary["recent"]
    phases = summary["phase_fractions"]
    evaluation = summary["evaluation"]
    recent_evaluation = evaluation["recent"]
    replay = summary["replay"]
    print(f"반복 {summary['first_iteration']}~{summary['last_iteration']} "
          f"({summary['rows']}개, 최근 {recent['iterations']}개 요약)")
    if summary["duplicate_rows_replaced"]:
        print(f"중복 반복 {summary['duplicate_rows_replaced']}개는 마지막 기록으로 대체")
    print(f"위치 처리량: 전체 {overall['positions_per_self_play_second']:.1f}개/초, "
          f"최근 {recent['positions_per_self_play_second']:.1f}개/초")
    print(f"대국 처리량: 전체 {overall['games_per_self_play_second']:.2f}판/초, "
          f"최근 {recent['games_per_self_play_second']:.2f}판/초; "
          f"최근 평균 {recent['mean_self_play_plies']:.2f}수")
    print(f"단계 비중: 자가대국 {phases['self_play']:.1%}, "
          f"학습 {phases['training']:.1%}, 평가 {phases['evaluation']:.1%}")
    detail = summary["self_play_detail"]
    if detail["covered_iterations"]:
        print(
            f"자가대국 내부: 계산·자료 구성 {percent(detail['compute_fraction'])}, "
            f"replay 저장 {percent(detail['replay_store_fraction'])} "
            f"({detail['covered_iterations']}/{summary['rows']}개 반복 기록)"
        )
    print(f"후보 평가: {evaluation['candidate_wins']}/{evaluation['games']} "
          f"({percent(evaluation['candidate_win_rate'])}), 승격 {evaluation['promotions']}/"
          f"{summary['rows']}; 흑 {percent(evaluation['black_win_rate'])}, "
          f"백 {percent(evaluation['white_win_rate'])}")
    print(f"최근 평가: 흑 {percent(recent_evaluation['black_win_rate'])}, "
          f"백 {percent(recent_evaluation['white_win_rate'])}, "
          f"승격 {recent_evaluation['promotions']}/{recent['iterations']}")
    if evaluation["latest_workers"] is not None:
        print(
            f"최근 평가 worker: {evaluation['latest_workers']}개 "
            f"({evaluation['worker_timing_covered_iterations']}/{summary['rows']}개 반복 기록)"
        )
    print(f"버퍼: {replay['latest_size']} 위치, 최근 생성량 기준 약 "
          f"{replay['estimated_recent_iteration_window']:.2f}사이클")
    checkpoint = summary["checkpoint"]
    if checkpoint["rows_with_timing"] < summary["rows"]:
        print(f"체크포인트 시간: {checkpoint['rows_with_timing']}/{summary['rows']}개 반복만 기록됨")
    else:
        size = checkpoint["latest_bytes"]
        size_text = f", 최근 {size / 2**20:.1f}MiB" if size else ""
        print(f"체크포인트 누적: {duration(checkpoint['seconds'])}{size_text}")
    if summary["projection"] is not None:
        projection = summary["projection"]
        print(f"목표 {projection['target_iterations']}회까지 남은 "
              f"{projection['remaining_iterations']}회: 약 {duration(projection['estimated_seconds'])} "
              "(최근 기록 시간 기준; 미측정 저장·시작·콜백 시간 제외)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
