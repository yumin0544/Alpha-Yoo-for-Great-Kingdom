# 정체 구간의 전술 비율·학습량·자료 생성 모델 비교

온라인 전술 25%를 5%로 줄이는 효과, 갱신 수 증가, 최신 learner 자료 생성은
서로 다른 변경이다. 같은 체크포인트에서 분기하고 한 번에 바꾸는 항목을
구분한다. 아래 명령은 준비·실험 방법이며 장기 학습 완료나 기력 향상 실측을
의미하지 않는다. 기존 기본값은 champion actor·온라인 LOSS 제외·전술 비율 25%다.

## 공통 출발점과 순서

공통 원본은 `runs/tactics-online-2026-10-07-1931i/latest.pt`의 완료 515사이클,
champion 버전 171 상태다. learner·champion·Adam·두 replay·난수 상태를 이어받고
모든 실험의 학습률을 `3e-5`로 맞춘다. 25% 대조군만 이전 학습률을 그대로
쓰면 전술 비율과 학습률 효과가 섞인다. 각 실험은 원본을 덮어쓰지 않는
새 출력 폴더를 사용한다. 판수 2,048·자가 대국/평가 탐색 예산·평가 상대와 기준,
teacher 예산·자료 증강·온도 스케줄은 공통 원본 설정을 유지한다.
이 원본은 버전 7의 WIN-only 증명 버퍼다. LOSS 수집을 끄는 것은 기존 LOSS를
삭제하거나 학습에서 제외하는 기능이 아니므로, LOSS가 이미 들어 있는 다른
체크포인트에서 A/B/C를 시작하면 같은 WIN-only 대조 조건이 아니다.

| 단계 | 전술 비율 | 미니배치 × 갱신 | actor | 새 LOSS 수집 | 비교 목적 |
| --- | ---: | ---: | --- | --- | --- |
| A | 25% | 512 × 128 | champion | 제외 | 같은 낮은 학습률의 대조군 |
| B | 5% | 512 × 128 | champion | 제외 | A와 비교해 전술 혼합 비율 분리 |
| C | 5% | 512 × 256 | champion | 제외 | B와 비교해 갱신 수 분리 |
| D | 5% | 512 × 256 | champion | 포함 | C와 비교해 LOSS 가치 학습 분리 |
| E | 5% | 512 × 256 | learner | 포함 | D와 비교해 최신 후보 자료 생성 분리 |

각 단계는 앞 단계의 학습 결과에서 재개하는 것이 아니라 **같은 515사이클
원본에서 별도로 분기**한다. 새로운 actor가 강해졌다고 가정하지 않는다.
learner는 한 사이클 시작 때 고정한 사본이며 같은 자료 생성 안에서 갱신하지
않는다. learner로 자가 대국하더라도 기준 모델 승격 조건과 `best.pt`는 유지한다.

## 학습 없이 설정 먼저 준비하기

다음 PowerShell 명령은 `--prepare-only`로 변경한 경계 상태만 저장한다.
실제 자가 대국·가중치 갱신은 시작하지 않는다. 출력 폴더에 이미 학습 기록이나
후보 파일이 있으면 오류로 중단하므로 재실행에는 다른 새 폴더를 선택한다.

```powershell
$plateauSource = "runs\tactics-online-2026-10-07-1931i\latest.pt"
$plateauCommon = @("--resume", $plateauSource, "--device", "cuda", "--reconfigure", "--prepare-only", "--online-tactics", "--learning-rate", "3e-5", "--batch-size", "512")
.\.venv\Scripts\python.exe examples\train.py @plateauCommon --online-tactics-fraction 0.25 --train-steps 128 --self-play-model champion --no-online-tactics-include-loss --output runs\plateau-A-25pct-128-champion
.\.venv\Scripts\python.exe examples\train.py @plateauCommon --online-tactics-fraction 0.05 --train-steps 128 --self-play-model champion --no-online-tactics-include-loss --output runs\plateau-B-05pct-128-champion
.\.venv\Scripts\python.exe examples\train.py @plateauCommon --online-tactics-fraction 0.05 --train-steps 256 --self-play-model champion --no-online-tactics-include-loss --output runs\plateau-C-05pct-256-champion
.\.venv\Scripts\python.exe examples\train.py @plateauCommon --online-tactics-fraction 0.05 --train-steps 256 --self-play-model champion --online-tactics-include-loss --output runs\plateau-D-05pct-256-defense
.\.venv\Scripts\python.exe examples\train.py @plateauCommon --online-tactics-fraction 0.05 --train-steps 256 --self-play-model learner --online-tactics-include-loss --output runs\plateau-E-05pct-256-learner
```

실제로 한 사이클만 비교하려면 각 준비 폴더의 `latest.pt`에서 같은 추가 반복 수로
재개한다. A/B는 각각 완료한 같은 수의 사이클을 비교한 뒤 C/D/E를 검토한다.
처음부터 다섯 실험을 동시에 GPU에서 실행하지 않는다. 예를 들어 B의 실제
1사이클 실행은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\plateau-B-05pct-128-champion\latest.pt --iterations 1 --output runs\plateau-B-05pct-128-champion
```

반복 경계의 갱신 수 변경은 체크포인트 버전 8의 `training_budget_history`에
기록한다. 지난 515사이클의 갱신 수를 `515 × 256`으로 다시 계산하지 않는다.
구간별 실제 일반/혼합 갱신과 별도 전술 갱신을 합쳐 전체 Adam step을 검증한다.
판수·일반 replay 용량·초기 seed는 이 설정 변경으로 바꿀 수 없다.

## 표본 수와 LOSS의 의미

512개 미니배치의 25%는 전술 128개·일반 384개다. 설정 5%는 반올림으로
전술 26개·일반 486개(실제 약 5.08%)다. 두 버퍼는 복원추출하므로 이 수치는
서로 다른 고유 위치 수가 아니다. 자료가 부족해 새 증명을 얻지 못해도 이전
증명 버퍼를 사용할 수 있고, 증명 버퍼가 없으면 전부 일반 replay로 학습한다.

완전히 증명된 `WIN`은 현재 플레이어 관점 가치 `+1`과 증명된 승리 수의 정책을
학습한다. `LOSS` 포함 시에는 가치 `-1`만 학습하고 해당 정책 손실을 0으로
마스킹한다. 패배 증명의 PV를 좋은 방어 수라고 가르치지 않는다. 실제 반격이
승리로 증명된 경우만 `WIN` 정책 목표가 된다. `UNKNOWN`·예산으로 끝난
미완료 증명을 안전/패배 정답으로 사용하지 않는다.

전체 정책·가치 손실의 분모는 전체 미니배치 행 수다. LOSS가 늘어날 때 WIN
정책 행만으로 평균을 다시 내어 WIN의 정책 가중치가 커지는 것을 막는다.
teacher 비율은 증명 WIN과 LOSS를 합한 전체 전술 표본 비율이며, WIN의 정책
갱신 비율과 같지 않을 수 있다. LOSS 채택을 켜도 5% 전체 예산 자체는 유지한다.

## 비교 지표와 모델 선택

- 합산 loss만 비교하지 말고 일반 policy/value, teacher policy/value,
  teacher WIN/LOSS value와 흑백별 가치 오차를 나누어 본다.
- 실제 전술 행 수·WIN/LOSS 행 수·teacher 시간·새 증명 수와 버퍼 크기를 기록한다.
  teacher는 시간 제한 탐색이므로 같은 RNG여도 CPU 부하에 따라 채택 수가 달라질 수 있다.
- 자가 대국의 흑백 승수·평균 수순·포획/연속 패스/자충수 종료 비율과 처리량을
  함께 비교한다. 긴 대국이나 작은 loss 자체가 더 높은 기력을 의미하지 않는다.
- 매 사이클의 champion 상대 평가뿐 아니라 저장한 candidate끼리 같은 탐색·
  배색·판수로 별도 대국한다. 학습에 넣지 않은 증명 위치도 별도로 확인한다.
  40판 한 번의 승률이나 승격 한 번만으로 장기 개선을 단정하지 않는다.

`teacher_win_value_loss`, `teacher_loss_value_loss`와 흑백별 value loss는 해당
분류에 실제로 추출된 행 수로 가중한 MSE다. 분류가 없는 미니배치의 0을
동일한 무게로 평균내어 오차를 작게 만들지 않는다. 같은 행 이름의 최상위
`*_rows` 지표는 갱신당 평균이고, `training_draw_counts`는 그 사이클 전체
갱신의 추출 횟수 합계다. 평균 행 수와 총 횟수를 혼동하지 않는다. 모두
복원추출 횟수이며 서로 다른 고유 표본 수를 뜻하지 않는다.

CLI는 마지막 완료한 learner를 `candidate.pt`로 내보내며 champion은 `best.pt`다.
학습을 이어갈 때는 추론용 파일이 아닌 전체 `latest.pt`를 사용한다.
`candidate.pt`의 직접 대국은 최신 후보의 상태를 보여 주는 것이며 자동 승격의
증거가 아니다. E를 최종 선택하더라도 champion 파일을 후보로 임의 교체하지 않는다.

공통 515사이클 원본에서 추론용 가중치만 별도로 추출한 파일은
`runs/plateau-candidates-2026-10-07/candidate-515.pt`와 `champion-171.pt`다.
각 `.manifest.json`에 원본·역할·진행 횟수·모델 해시를 기록하며, 추출은 원본
학습 상태를 진행하거나 champion을 바꾸지 않는다. 이 파일은 대국용이고
Adam·replay를 이어받는 실험에는 여전히 원본 `latest.pt`가 필요하다.

실행·저장 형식은 [학습 루프](training_loop.md), 증명 정답의 안전 조건은
[전술 학습](tactical_curriculum.md), 후보 추출은 `examples/export_candidate.py`를 참고한다.

## 실제 준비·검증 기록: 2026-10-07

### 학습 전 후보와 champion의 200판 기준 평가

원본의 learner 515와 champion 171을 추론용 파일로 별도 추출하고 새로운
시드 `20261007515`로 200판을 완료했다. 양쪽 매 수 탐색 128회,
`batched_cpp`·CUDA·worker 12·leaf 8·트리 재사용 켬·전술 검사 켬,
처음 6수 온도 1·이후 최다 방문 선택을 사용했다. 루트 잡음과 시간 제한은 껐다.
기본 확정 규칙을 사용하고 후보의 흑/백을 100판씩 교대했다.

| 후보 색상 | 후보 승수 / 판수 | 관측 승률 |
| --- | ---: | ---: |
| 흑/선공 | 47 / 100 | 47.0% |
| 백/후공 | 30 / 100 | 30.0% |
| 전체 | 77 / 200 | 38.5% |

총 8,334수, 포획 143판·연속 패스 41판·자충수 16판이었다. 대국 시간은
91.908초였으며 모델 읽기·준비 시간과 분리한다. 이는 이번 조건에서 A/B 전의
출발 기준이지 절대 기력 수치가 아니다. 관측 승률은 저장된 승격 기준 60%보다
낮으며, 이 별도 대국 프로그램은 평가만 하고 learner·champion을 교체하지 않는다.
각 사이클의 40판 평가나 다른 seed의 결과와 합쳐 한 대국 시리즈로 취급하지 않는다.

원본 대국별 수순·종료·설정·요약은
[baseline-200.jsonl](../runs/plateau-candidates-2026-10-07/baseline-200.jsonl)에 있다.
추출 출처와 파일 해시는
[후보 manifest](../runs/plateau-candidates-2026-10-07/candidate-515.manifest.json)와
[champion manifest](../runs/plateau-candidates-2026-10-07/champion-171.manifest.json)에 있다.
다른 도구가 출력한 tensor 지문은 해시 구성 방식이 다를 수 있으므로 출처를
연결할 때는 같은 방식의 지문 또는 추론용 **파일 SHA-256**을 비교한다.

### E 설정 저장 확인: 아직 새 사이클은 실행하지 않음

실제 `runs/plateau-E-05pct-256-learner/latest.pt`를 `--prepare-only`로 만들고
버전 8 복원을 확인했다. learner actor·학습률 `3e-5`·미니배치 512·갱신 256·
전술 설정 5%·새 LOSS 수집 켬을 저장했다. 완료 반복은 여전히 515,
전체 Adam 갱신 66,176회 = 일반/혼합 65,920회 + 별도 전술 256회이고,
기존 증명 위치 194개를 유지했다. 새로운 516사이클의 대국·학습·평가를
이미 완료했다는 뜻이 아니다.

저장한 예산 이력은 시작 0에서 128회, 시작 515에서 256회다. 과거 515사이클의
진행 횟수와 Adam step을 변경하지 않았다. 원본 전체 체크포인트 SHA-256은
`540ac083a39cccd98f43c13eaf91276272f8cd51d77898f046eebb5539bbfdea`이며
추출·설정 준비 후에도 동일함을 확인했다. 원본·champion을 보존하고,
`candidate.pt`는 최신 후보 추론 파일로만 분리했다.

### A/B 각 1사이클 실제 실행

같은 515사이클 원본에서 A(25%)와 B(5%)를 각각 별도 폴더로 시작하고
새 1사이클을 끝까지 완료했다. 두 실행 모두 learner·champion·Adam·RNG를
이어받고 학습률 `3e-5`, champion actor, 미니배치 512·갱신 128, 새 LOSS 수집
꺼짐을 사용했다. 자료 생성 actor의 가중치 해시도 같았다.

각 실행은 2,048판에서 77,700개 위치를 생성했고, 새 WIN 증명 8개를 채택해
증명 버퍼가 194개에서 202개가 됐다. 완료 반복은 둘 다 516, 전체 Adam 갱신
66,304회 = 일반/혼합 66,048회 + 별도 전술 256회였다. champion 버전 171을
유지했고 두 실행 모두 승격되지 않았다.

| 항목 | A: 전술 25% | B: 설정 5% |
| --- | ---: | ---: |
| 미니배치 일반 / 전술 행 | 384 / 128 | 486 / 26 |
| 실제 전술 행 비율 | 25.0% | 5.078% |
| 사이클 일반 추출 횟수 | 49,152 | 62,208 |
| 사이클 전술 추출 횟수 | 16,384 | 3,328 |
| 일반 정책 CE / 가치 MSE | 2.48368 / 0.75364 | 2.46338 / 0.74671 |
| teacher 정책 CE / 가치 MSE | 0.51902 / 0.04772 | 0.65203 / 0.09377 |
| 사이클 후보 평가 | 16/40 (40.0%) | 15/40 (37.5%) |
| 후보 흑 / 백 승수 (각 20판) | 10 / 6 | 11 / 4 |
| 가중치 학습 시간 | 3.174초 | 3.120초 |
| 사이클 계산 시간 | 156.461초 | 157.107초 |

사이클 계산 시간은 자가 대국·teacher·학습·평가를 포함하고 체크포인트·추론용
export 등 파일 작업과 시작 준비를 제외한다. 같은 총 65,536회 추출 중 일반
자료 추출이 26.56% 늘고 전술 추출이 줄어드는 연결을 실제로 확인했다.
표의 추출 횟수는 고유 위치 수가 아니고 손실은 학습 중 배치에 대한 관측치다.
일반 loss가 조금 낮아지고 teacher loss가 높아졌다는 사실만으로 기력 향상이나
정체 해결을 결론 내리지 않는다.

혼합 표본 수가 다르면 RNG 소비량이 달라져 **사이클 평가 seed도 달라진다**.
따라서 16승과 15승을 같은 대국 목록의 공정한 기력 A/B로 해석하지 않는다.
학습 후 두 candidate를 공통 새 seed·동일 상대·동일 탐색 예산의 충분한 외부
대국으로 다시 비교해야 한다. 각 조건은 1사이클뿐이며, C/D/E의 장기 학습
효과나 5%가 최적 비율이라는 증거는 아직 없다.

실제 지표는 [A metrics](../runs/plateau-A-25pct-128-champion/metrics.jsonl)와
[B metrics](../runs/plateau-B-05pct-128-champion/metrics.jsonl)에 있다.
각 폴더의 `latest.pt`는 자기 실험의 완료 516사이클을 재개하는 상태이고,
`candidate.pt`는 승격과 무관한 학습 후 후보다. E 폴더는 여전히 준비한
515사이클 경계다. 모든 분기는 원본 파일과 champion을 보존했다.

### 구현 검증 범위

CLI 검증 12개, actor·구간별 갱신 예산 검증 9개, 후보 추출 검증 9개,
증명 학습 검증 9개와 C++ 회귀 검증 7개가 통과했다. actor 검증에는
실제 CUDA learner 자료 생성·혼합 LOSS 사이클·행 가중 분류 MSE 검증도 포함된다.
네트워크 서버 권한에 막힌 localhost UI 검증은 권한을 갖춘 별도 실행에서
28개를 통과했다. 서로 다른 실행 조건의 결과를 합쳐 단일 전체 검증 실행이
모두 통과했다고 표시하지 않는다. 이번 작업은 `engine/`를 변경하지 않았다.
