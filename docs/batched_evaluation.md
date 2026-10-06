# C++ leaf 배치 평가와 하위 트리 재사용

`--evaluation-backend batched_cpp`는 평가 경로에만 적용한다. 기존 `engine/`의
규칙 코드는 변경하지 않고 `mcts/BatchedPUCT`와 바인딩에서 입력 준비·탐색을
처리한다. 기존 `legacy` 경로는 정확성 비교 및 이전 체크포인트 재개를 위해
그대로 남아 있으며, 기본값도 `legacy`다.

## 실행 구조

1. C++가 여러 미평가 leaf를 고르고, 상태와 루트부터의 경로를 보관한다.
2. 평가 중 표시와 임시 방문 수/virtual loss로 같은 leaf 중복 요청을 막는다.
3. 각 leaf의 합법 수를 한 번 계산해 노드 확장과 입력 마스크에 공유한다.
4. C++가 스키마 1의 연속 float32 입력 `[N,10,9,9]`와 bool 마스크 `[N,82]`를 만든다.
5. Python은 후보·기준 모델별로 이미 준비된 요청을 묶고 GPU에서 추론한다.
6. 정책·가치를 CPU로 한 번에 회수해 C++가 각 경로에 정확히 한 번 역전파한다.

`--evaluation-leaf-batch-size`는 한 탐색의 leaf 상한이다. 추론 서비스의 최대
행 수는 `min(evaluation_workers, evaluation_games) * leaf_batch_size`다.
후보·기준 모델은 별도 큐이므로 이 값이 항상 실제 추론 배치 크기는 아니다.
최초 루트 정책은 먼저 평가해야 한다. 이후에는 leaf 하나가 아니라 한 묶음의
결과를 기다린다. 무제한 비동기 발행이나 동일 대국 내 GPU 추론 중 다음 묶음
생성까지 구현한 것은 아니다. 다른 대국의 C++ 작업과 추론은 진행될 수 있다.

## 안전한 트리 재사용

실제 착수가 수락되면 후보·기준 모델의 **각자 소유한** 트리 모두에 `advance`
한다. 선택한 자식과 그 후손만 보존·압축하고 나머지는 버린다. 상대의 착수도
반영해야 다음 자기 차례에 두 수 전 상태의 트리를 잘못 쓰지 않는다.

자식이 아직 탐색되지 않았으면 그 모델의 트리를 비우고 다음 탐색을 새로
시작한다. 상태 불일치도 초기화한다. 비교에는 돌/중립돌 배치, 영구 집 소유권,
차례, 양쪽 돌 재고, 연속 패스 수, 규칙 옵션 및 종료 결과가 포함된다.
비합법 `advance` 요청은 보관된 트리를 변경하지 않고 거절한다.

기존 방문·가치 통계는 다음 선택에 쓰지만, 반환하는 `moves.visits`와 가치 평균은
**이번 호출에서 추가한 시뮬레이션**의 통계다. 따라서 방문 수 합은 여전히
`result.simulations`와 같다. 재사용이 탐색 예산을 대신하지 않으며, 시간 제한이
없을 때 매 착수에 지정한 횟수만큼 새 시뮬레이션을 수행한다.

모델마다 정책·가치가 다르므로 일반 탐색 트리를 두 모델 사이에 공유하지 않는다.
모델 가중치를 바꾸면 반드시 `clear()`한다. 평가마다 모델 사본과 탐색기를 새로
만들므로 학습 갱신·승격 후 이전 가중치의 트리가 남지 않는다. 트리 자체는
체크포인트에 저장하지 않는다. 루트 Dirichlet 잡음은 이 평가 전용 API에서 금지한다.

## 실행과 재개

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\gpu-strength-speed-pilot-2026-10-06\latest.pt --evaluation-backend batched_cpp --evaluation-workers 12 --evaluation-leaf-batch-size 8 --evaluation-reuse-tree --iterations 1 --output runs\gpu-batched-evaluation-pilot
```

기존 체크포인트/실험을 보존하려면 새 출력 폴더를 사용한다. 이 문서의 명령은
예시이며 실행 자체를 완료했다는 의미가 아니다. `--no-evaluation-reuse-tree`로
재사용 없이 leaf batching만 비교할 수 있다.

체크포인트 버전 5는 worker 수, 평가 backend, leaf 배치 크기, 트리 재사용 여부를
저장한다. 버전 1~4는 이전 `legacy` 알고리즘으로 복원한다. 새 설정은
`--reconfigure` 없이 명시적으로 덮어쓸 수 있으며 이후 저장부터 유지된다.

## 계측과 비교

`evaluation_diagnostics`에 두 모델의 신경망 평가 행 수, 추론 배치 수,
평균/최대 배치 크기와 트리 재사용 통계를 기록한다. 서비스의 시간은 합산 작업
시간이며, 여러 worker가 함께 기다린 큐 대기 합계는 벽시계 시간보다 클 수 있다.
이를 전체 평가 시간의 비율로 해석하지 않는다.

새 탐색 통계의 `network_evaluations`, `inference_batches`,
`max_observed_batch_size`, `reused_nodes`, `inherited_visits`는 각 search 범위다.
`advance_calls`, `reuse_hits`는 탐색 객체의 누적값이다. 평가 진단은 전자는
착수별 합계(최대 배치는 최대값), 후자는 대국 종료 시 최종값 합계로 집계한다.

```powershell
.\.venv\Scripts\python.exe examples\batched_evaluation_benchmark.py --model runs\gpu-strength-v2\best.pt --device cuda --workers 12 --games 20 --simulations 32 --leaf-batch-sizes 1 4 8 16 --reuse-modes off on --output docs\benchmarks\batched_evaluation.jsonl
```

추론용 모델 파일은 `--model`로 지정하고, 상대를 생략하면 같은 가중치의 사본과
대국한다. 저장된 학습 후보와 기준 모델을 그대로 비교하려면 대신 전체
체크포인트를 사용한다.

```powershell
.\.venv\Scripts\python.exe examples\batched_evaluation_benchmark.py --training-checkpoint runs\gpu-strength-speed-pilot-2026-10-06\latest.pt --device cuda --workers 12 --games 40 --simulations 128 --leaf-batch-sizes 8 16 --reuse-modes on --output docs\benchmarks\batched_evaluation_128.jsonl
```

`--training-checkpoint`는 공개된 `Trainer.load_checkpoint()`의 엄격한 검증으로
저장된 learner·champion을 읽고, Replay·optimizer를 측정 전에 해제한다.
학습·승격·체크포인트 저장은 실행하지 않는다. 원본 파일과 모델 가중치의 해시가
실행 전후 같아야 하며, 원본 반복·기준 모델 버전과 설정도 JSONL에 남긴다.
`--model`과 함께 쓰거나 `--reference-model`을 추가할 수 없다. 출력 JSONL도
기존 파일을 덮어쓰지 않으므로 실행마다 새 경로를 지정한다. 시작·로드·warm-up·
해시 계산·JSONL 쓰기는 제외하고, 서비스 생성·종료, 규칙·탐색·인코딩·추론과
기존 전술 검사는 평가 시간에 포함한다.

`leaf_batch_size=1`이고 재사용을 끄면 결정적 정책·가치 콜백에서 기존 PUCT와
일치해야 한다. 그 외 설정은 결과를 기다리기 전 선택과 기존 통계를 사용하므로
방문 분포·착수·대국 길이가 달라질 수 있다. 같은 판수·탐색 예산의 시간과
진행 수/초를 함께 비교한다. 동일 모델의 흑백 쌍 50%는 배색 대칭 검증이며
기력 동등성의 증거가 아니다. 기력에 대한 비교는 별도의 충분한 대국이 필요하다.

현재 전술 검사는 그대로 유지한다. 2~3수 강제 승패 solver, 전술 캐시·가지치기는
이번 변경에 포함하지 않는다.

## 실측과 검증: 2026-10-06

RTX 3050에서 같은 `best.pt` 두 개, 20판·수당 새 탐색 32회·worker 12를 사용해
9개 설정을 각각 한 번 측정했다. `legacy`의 37.898초가 leaf 1·재사용 없음에서는
18.717초로 줄었으며, 두 경로의 총 694수·21,942회 신경망 평가와 집계 대국 결과가
일치했다. C++ 입력 준비·상태 보관과 새 추론 서비스의 전체 효과를 비교한 결과다.
여러 leaf를 함께 평가하는 leaf 16·재사용 켬은 5.243초였지만 총 868수로 탐색
경로가 달라졌다. 7.23배는 같은 설정 예산에서의 시간 비율이지 동일 수순·동일
작업량의 가속이나 기력 보존 증명이 아니다. 이 짧은 실험에서 재사용만 추가한
시간 차이는 작았으며, 대국 길이도 바뀌므로 재사용의 독립 효과를 단정하지 않는다.

완료 244사이클의 learner와 기준 모델 106을 고정한 별도 평가에서는
40판·새 탐색 128회·worker 12 조건의 시간이 `legacy` 263.131초,
leaf 8·재사용 켬 18.958초, leaf 16·재사용 켬 10.695초였다. 총 진행 수는
각각 1,448/1,467/1,111수였다. leaf 8은 신경망 평가 행 수도 기존보다 많았지만
GPU 추론 호출을 65,670회에서 7,886회로 줄였다. leaf 16의 24.60배 시간 비율은
더 짧아진 대국·줄어든 작업량 효과도 포함하므로 동일 작업량 가속으로 표시하지 않는다.

표에 사용한 값은 다른 검증 작업 없이 실행한 재측정이다. CPU 검증이 일부 겹친
최초 기록도 예비 원본으로 보존했다. 같은 모델·시드에서도 leaf 8의 진행 수·
승수가 달라졌다. 동적 GPU 배치에 따른 작은 수치 차이도 탐색 선택을 바꿀 수
있으므로 비트 단위의 동일 수순을 보장하지 않는다. 최종 후보 승률은
`legacy` 60%, leaf 8/16 각 55%였지만
40판으로 기력 보존·향상을 결론 내리지 않는다. 이 실험은 평가만 실행했으며
새 backend로 2,048판 자가 대국·학습을 포함한 전체 사이클을 다시 완료하지는 않았다.

전체 결과와 측정 조건은 [성능 기록](performance.md), 9개 설정 원본은
[평가 배치 JSONL](benchmarks/batched_evaluation_2026-10-06.jsonl)에 있다.
128회 탐색의 최종 원본은
[재측정 JSONL](benchmarks/batched_evaluation_128_repeat_2026-10-06.jsonl), 최초 기록은
[예비 JSONL](benchmarks/batched_evaluation_128_2026-10-06.jsonl)에 있다.
기본 경로는 자동으로 변경하지 않았으므로 학습 평가에 적용할 때는
`--evaluation-backend batched_cpp`를 명시한다.

C++와 Python·CUDA 검증을 포함한 CTest 27개 묶음이 모두 통과했다
(총 289.23초). 단일 leaf와 기존 PUCT의 일치, 입력·마스크의 기존 인코딩 일치,
새 탐색 예산·역전파 횟수, 패스·포획·자충수 종료, 상대 착수를 포함한 트리 이동,
상태 불일치 초기화, 콜백 실패·재진입·동시 사용, 모델별 큐 분리와 버전 1~5
체크포인트 호환성을 검증했다. 이번 작업에서 `engine/`는 변경하지 않았다.

새 패키지를 가상환경에 재설치한 뒤, `PYTHONPATH` 없이 설치된 모듈로 새 Python
검증 4개 묶음의 40개 사례도 모두 통과했다(실제 CUDA 2개 포함).
의존성 검사 `pip check`와 설치된 `BatchedPUCT`·입력 인코더의 API도 확인했다.
