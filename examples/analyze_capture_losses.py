"""완료된 모델 대국 JSONL에서 색별 포획·자충수·패스 패배율을 읽기 전용 집계."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kingdom_ai.capture_metrics import analyze_capture_losses


def _percentage(value):
    return "해당 없음(패배 0판)" if value is None else f"{value:.2%}"


def print_metrics(metrics, label):
    print(f"{label}: {metrics['games']}판, {metrics['wins']}승 {metrics['losses']}패, "
          f"승률 {_percentage(metrics['win_rate'])}")
    labels = {"capture_losses": "포획 패배", "suicide_losses": "자충수 패배",
              "capture_or_suicide_losses": "포획+자충수 패배", "pass_losses": "연속 패스/집 패배"}
    for field, text in labels.items():
        print(f"  {text}: {metrics[field]}판 | 전체 경기 중 "
              f"{_percentage(metrics['loss_rates_all_games'][field])} | 전체 패배 중 "
              f"{_percentage(metrics['loss_shares_all_losses'][field])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="model_match.py가 저장한 완료 results.jsonl")
    parser.add_argument("--model", choices=("a", "b"), default="a", help="표시할 모델(기본 a)")
    parser.add_argument("--json", action="store_true", help="두 모델의 모든 집계와 선택 모델을 JSON으로 표시")
    args = parser.parse_args(argv)
    try:
        report = analyze_capture_losses(args.results)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report["selected_model"] = args.model
    if args.json:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    metrics = report["models"][args.model]
    print(f"모델 {args.model.upper()} ({metrics['name']}) | 완료 기록 {report['source']}")
    print_metrics(metrics, "전체")
    for color, label in (("Black", "흑/선공"), ("White", "백/후공")):
        print_metrics(metrics["colors"][color], label)
    print("포획 패배만 줄고 패스 패배가 늘었는지 승률과 함께 비교하세요. "
          "관측된 포획 패배 0판은 미래의 포획 위험 0을 증명하지 않습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
