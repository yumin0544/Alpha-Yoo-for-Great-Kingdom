# 포획·자충 패배 회피 학습 (2026-10-12)

## 확인한 문제

`runs/matches/20261012-103331-087c1fb9/results.jsonl`의 완료된 1,000판에서
211v는 185v에 692승 308패(69.2%)였다. 패배는 포획 256판, 자충수 14판,
연속 패스 점수 판정 38판이다. 사용자에게 보인 270판은 포획과 자충수의
합계이며, 전체 경기의 27%다. 승률 향상과 돌을 잃지 않는 능력은 구분한다.

이 270판을 기존 C++ 규칙으로 재생했을 때 마지막 자기 차례에는 모두 즉시
검사의 회피 수가 없었다. 그 앞선 착수는 기존 2수 검사에서는 통과했지만
이후 방어 불가능한 상태로 이어졌다. 마지막 아타리에만 정답을 주거나 기존
깊이 20을 30으로 바꾸는 것만으로 원인 착수의 정책이 직접 교정되지는 않는다.
최근 100사이클의 기존 증명 2,570개 중 1,487개(57.9%)는 예산 소진으로
UNKNOWN이었다. 증명된 LOSS는 기존 버퍼에서 가치 -1만 학습하며, 패배 PV를
좋은 방어 정책으로 흉내 내지 않는 기존 계약은 유지한다.

## 새 회피 teacher

`--online-tactics-capture-defense`를 명시적으로 켜면 기존 GPU 자가 대국이
완료된 후 다음 처리를 추가한다. `engine/`의 게임 규칙은 변경하지 않는다.

1. 포획·자충수로 끝난 새 대국의 기록을 균등 reservoir로 보존한다. 최종 패배
   색이 실제로 착수했던 마지막 30 ply 구간의 착수 **직전** 위치를 추출한다.
   실패 직전·더 이른 방어·구간 초반을 층화해 이미 늦은 마지막 수만 반복하지 않는다.
2. 전체 실제 수순을 C++ `State`로 재생해 영구 집 소유권·재고·차례를 복원한다.
   기록의 패배 결과만으로 후보 착수를 패배 수라고 단정하지 않는다.
3. 후보 착수 직후 상대 차례의 강제 승리를 증명한다. 최대 깊이 30은 양쪽
   착수를 합친 상한이며, 후보 착수 1수 + 이후 최대 29수다. 깊이
   1·3·5·9·15·21·29의 반복 심화가 짧은 함정을 먼저 찾는다. 모든 단계는
   위치당 노드·시간 예산을 공유하며, 전체 생성 시간의 남은 예산도 적용한다.
4. 상대 WIN이 증명된 실제 착수만 별도 `CaptureDefenseReplay`에 저장한다.
   자충수 등의 즉시 패배는 기존 규칙 판정으로 1수 증명을 사용한다.
   이미 모든 수가 즉시 패배하는 부모 위치는 회피 정책 자료에서 제외한다.
5. `-log(1 - 위험 수들의 정책 확률 합)`을 안정적인 logsumexp 차이로 학습한다.
   부모 위치의 가치 정답이나 안전 대안 정책을 만들지 않는다. UNKNOWN은
   증명된 위험 수가 아니며, 남은 수를 안전하다고 선언하지 않는다.

기존 WIN 정책·LOSS 가치 teacher와 새 회피 teacher는 동일한 전술 표본 예산을
공유한다. 준비한 배치 512·전술 5%에서는 일반 486개, 기존 증명 13개,
회피 13개를 사용한다. 한쪽 버퍼가 비면 있는 쪽에 전술 26개를 배정하며,
둘 다 비면 일반 자료만 학습한다. 사이클당 Adam 갱신 512회는 늘리지 않는다.
회피 행에는 가치 손실이 없고, 전체 손실 분모는 세 자료의 전체 행 수다.

## 준비한 상태와 실행

원본은 `runs/tactics-depth20-512steps-league185-2026-10-08/latest.pt`의
완료 1,858사이클·champion 211이다. 별도
`runs/capture-defense30-2026-10-12/latest.pt`를 준비했다. 일반/기존 전술 버퍼,
모델·champion·Adam·난수·진행 횟수·학습 예산 이력은 유지한다.
새 회피 버퍼는 비어 있으며 첫 추가 사이클부터 채워진다.

변경한 전술 설정은 깊이 30, 위치당 노드 4,000,000·시간 4초, 두 teacher의
생성 구간 합계 예산 60초다. 회피 teacher에 우선 30초를 배정하고, 남은 시간은
기존 WIN/LOSS teacher에 쓴다. 위치당 예산을 반복 심화마다 다시 주지 않는다.
기록 재생·OS scheduling 등 때문에 엄밀한 전체 wall-clock 상한은 아니다.

`past-version`에 추가되어 있던 211v도 명시적으로 상대 목록에 반영했다.
과거 상대는 106·128·171·177·185·211이며, 기존 평균 승률 캐시는 비웠다.
55% 1차 조건과 각 버전 100판 후 후보 평균이 더 높을 때만 승격하는 조건은
그대로다. 기존 상대 파일을 대체·수정·삭제하지 않는다.

이번 준비에서는 실제 추가 학습을 시작하지 않았다. 아래는 **추가 10사이클**
(1,859~1,868)을 시작하는 명령이다.

```powershell
.\.venv\Scripts\python.exe examples\train.py `
  --resume runs\capture-defense30-2026-10-12\latest.pt `
  --device cuda --iterations 10 --output runs\capture-defense30-2026-10-12
```

다른 최신 상태에 연결할 때는 기존 학습 파일이 없는 새 출력 폴더를 사용한다.
이번에 이미 준비한 경로에는 아래 준비 명령을 다시 실행하지 않는다.

```powershell
.\.venv\Scripts\python.exe examples\train.py `
  --resume runs\tactics-depth20-512steps-league185-2026-10-08\latest.pt `
  --device cuda --reconfigure --prepare-only --refresh-promotion-archive `
  --online-tactics-capture-defense --online-tactics-max-depth 30 `
  --online-tactics-max-nodes 4000000 --online-tactics-time-limit-ms 4000 `
  --online-tactics-generation-seconds 60 --output runs\new-capture-defense-experiment
```

## 확인할 지표와 한계

`metrics.jsonl`의 `online_tactics.capture_defense`에서 새 반례 수·UNKNOWN·이미
불가피한 패배 제외·버퍼 크기를 확인한다. `defense_policy_loss`,
`defense_bad_probability`, `training_draw_counts.defense_rows`는 실제 회피 자료의
학습량과 위험 수 확률을 나타낸다. 전체 대국의 포획 패배율은 아니다.
전체 증명 기록은 체크포인트에 유지하며, 긴 기록은 JSONL에서
`record_summary`로 축약한다. v10은 새 버퍼를 저장하고 v1~v9는 회피 꺼짐·빈
버퍼로 복원한다. 기존 학습 동작과 저장된 깊이는 명시적 설정 변경 없이 유지한다.

```powershell
.\.venv\Scripts\python.exe examples\analyze_training.py `
  runs\capture-defense30-2026-10-12\metrics.jsonl --recent 10
.\.venv\Scripts\python.exe examples\analyze_capture_losses.py `
  runs\matches\20261012-103331-087c1fb9\results.jsonl --model a
```

학습 후에는 새 독립 1,000판 결과에서 승률·포획 패배·자충 패배·점수 패배를
함께 비교한다. 포획을 피하려고 패스로 지는 비율만 늘어난 것을 개선으로
간주하지 않는다. 기존 1,000판은 변경 전 기준이고, 도구는 불완전하거나
모순된 기록을 완료된 평가처럼 보고하지 않는다. `--json`도 지원한다.

이 구현은 **매 GPU 착수에서 모든 수를 30수 읽는 기능이 아니다.** 실행 중
PUCT 예산과 기존 얕은 즉사 검사는 유지하며, 깊은 증명은 학습용 자료 생성에
사용한다. 콘솔에서 실행 중 증명 선택도 원하면 기존 `play_ai.py`의
`--tactical-depth 30`을 별도로 사용할 수 있지만, 역시 제한 예산의 UNKNOWN과
추가 CPU 비용이 존재한다. 기존 solver는 전체 게임의 강제 승패를 증명한다.
대표 PV가 포획으로 끝나도 모든 응수가 포획으로 끝난다는 증명은 아니다.
30수 상한·학습 연결이 포획 패배 0회나 기력 향상을 보장하지 않는다.

실제 4수 함정 및 1수 자충수를 기존 C++ 규칙으로 증명하고, 새 버퍼 저장·복원·
혼합 손실까지 연결하는 회귀 테스트를 추가했다. UNKNOWN 배제, 총 탐색 예산,
대칭 마스크, 가치 정답 미생성, 기존 체크포인트와 정확한 재개도 검증한다.
작은 테스트 모델의 위험 수 확률 감소는 큰 모델의 독립 대국 기력 실측과 다르다.

통합 회귀 테스트 174개가 통과했다. 준비한 체크포인트의 모델·기존 버퍼·Adam·
난수·진행 횟수·예산 이력을 원본과 정확히 비교하고 CUDA 일반 재개도 확인했다.
원본 `latest.pt`와 `past-version` 26개 파일의 SHA-256·크기·수정 시각이 작업
전후 같았으며, `engine/`는 변경하지 않았다. 장기 추가 학습과 새 1,000판
기력 평가는 이번 구현 작업에서 실행하지 않았다.
