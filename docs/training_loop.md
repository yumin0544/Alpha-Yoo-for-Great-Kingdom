# 반복 강화학습과 학습 재개

`python/kingdom_ai/loop.py`는 신경망 PUCT 자가 대국, 제한된 자료 버퍼,
미니배치 학습, 기존 모델과의 대국 평가를 반복한다. 완료한 반복의 전체 상태를
저장하므로 프로그램을 중단한 뒤 이어서 실행할 수 있다. 게임의 규칙과 실제
착수는 선택한 자가 대국 경로에서 수행하며 `engine/`는 변경하지 않는다.

기본 `cpu` 경로는 기존 C++ 규칙·PUCT로 대국을 순차 생성한다. `cuda` 경로는
여러 독립 대국의 규칙·입력 인코딩·PUCT 트리·신경망 추론을 GPU에서 함께
실행하고 같은 학습 자료 버퍼에 연결한다. 미니배치 가중치 갱신은 `--device`의
장치에서 수행한다. 평가 대국은 두 경로 모두 기존 C++ 규칙·PUCT를 사용하며,
기본은 순차 실행이고 `--evaluation-workers`가 2 이상이면 독립 대국들의 모델
추론을 `--device`에서 batch로 묶는다. Bitboard와 C++/LibTorch 직접 추론은
이후 작업이다.
`--evaluation-backend batched_cpp`는 C++에서 여러 leaf와 입력을 준비하고,
모델별 GPU 배치 추론 및 실제 착수 후 하위 트리 재사용을 연결한다.
기본값과 이전 체크포인트는 `legacy`다. 자세한 내용은
[C++ leaf 배치 평가](batched_evaluation.md)를 참고한다.
반복 학습 프로그램의 구현과 수십만 판 학습으로 얻은 기력 검증은 구분한다.

## 실행 순서

한 번의 반복은 다음 단계로 구성된다.

1. 현재 기준 모델인 `champion`으로 지정한 수의 PUCT 자가 대국을 끝까지 진행한다.
2. 착수 전 상태, 탐색 방문 비율, 최종 승패 목표를 FIFO 자료 버퍼에 넣는다.
3. `online_tactics`를 켰으면 새 대국의 일부 위치를 깊게 읽고, 증명한 전술을
   별도 FIFO 버퍼에 저장한다. 기본값은 꺼짐이다.
4. 일반 replay에서 위치를 복원추출하여 `learner`를 지정한 횟수만큼 갱신한다.
   온라인 전술 버퍼가 있으면 설정한 비율로 그 위치를 미니배치에 섞는다.
5. 갱신한 후보와 `champion`이 같은 탐색 예산으로 흑백을 교대해 대국한다.
6. 후보 승률이 승격 기준 이상이면 후보를 새로운 `champion`으로 채택한다.
7. 완료한 반복의 모델, 두 버퍼, optimizer, 난수와 진행 횟수를 저장하고 지표를 기록한다.

승격되지 않아도 `learner`와 Adam 상태는 다음 반복으로 이어진다.
자가 대국에 사용하는 모델은 평가를 통과한 `champion`이다. 따라서 후보를
거부할 때마다 학습된 가중치와 optimizer를 초기화하지 않는다.

CPU 자가 대국은 매 판, GPU 자가 대국은 매 배치 새로운 시드를 사용한다.
기본적으로 루트 Dirichlet 잡음과
방문 수에 따른 착수 샘플링을 사용한다. 일반 replay의 정책 목표는 실제 방문
비율이고, 착수 샘플링 온도는 게임에서 선택하는 수에만 적용된다. 가치 목표는 저장한
착수 전 플레이어가 최종 승자이면 `+1`, 아니면 `-1`이다.

GPU 경로는 한 번에 최대 `self_play_batch_size`판을 진행하며 배치의 모든
대국이 종료된 뒤 다음 배치를 시작한다. 착수 전 입력·합법 수·방문 정책과
최종 승패를 기존 `GameData` 형식의 CPU 자료로 회수해 FIFO 버퍼에 저장한다.
방문 정책은 온도를 적용하기 전의 탐색 방문 비율이고, 온도는 실제 착수의
추출 확률에만 적용한다. 매 자료 수집 호출에서 현재 `champion`의 가중치로
GPU 탐색기를 만들어, 승격된 모델이 다음 반복의 자가 대국에 반영된다.

## 짧게 실행하기

저장소 루트에서 업데이트한 패키지가 설치된 가상 환경을 사용한다.
설치 환경은 [Python 바인딩 안내](python_bindings.md)와
[신경망 연결 안내](neural_network.md)를 참고한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --iterations 1 --games-per-iteration 2 --simulations 4 --eval-games 2 --eval-simulations 4 --train-steps 2 --batch-size 16 --channels 8 --residual-blocks 1 --output runs\smoke
```

이 설정은 실행 연결을 확인하기 위한 것이다. 평가 2판과 탐색 4회로 얻은 결과를
학습 품질이나 기력 향상의 증거로 사용하지 않는다. `--iterations`는 **이번
실행에서 추가로 수행할 반복 수**이며 총 목표 반복 수가 아니다.

출력 디렉토리에 이미 `latest.pt`, `best.pt` 또는 `metrics.jsonl`이 있으면
새 실행은 오류로 끝난다. 기존 실행을 이어가려면 `--resume`을 사용한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --resume runs\smoke\latest.pt --iterations 1 --output runs\smoke
```

재개 시 학습 설정과 모델 구성은 체크포인트에서 복원한다. `--iterations`,
`--output`, `--device`, `--threads`와 `--evaluation-*` 실행 설정을 변경할 수 있으며,
탐색 횟수·학습률·버퍼
용량 등 학습 설정을 함께 지정하면 오류를 표시한다. `--initial-model`은
기존 `save_model` 파일의 가중치로 **새 학습 실행**을 시작하는 옵션이다.
`--resume`과 함께 사용할 수 없고 optimizer, 버퍼와 진행 횟수는 이어받지 않는다.
허용된 학습 설정을 바꾸려면 `--resume --reconfigure`와 기존 학습 파일이 없는
새 `--output`을 사용한다. `--prepare-only`를 더하면 실제 대국 전에 변경된
전체 상태만 저장할 수 있다. 사이클당 판수·갱신 수·일반 replay 용량은 변경할
수 없고, 이미 채운 온라인 전술 replay의 용량 변경도 거부한다.

## GPU 자가 대국과 학습 시작하기

CUDA용 PyTorch와 호환 드라이버가 설치된 환경에서 자가 대국 경로를 `cuda`,
학습 장치를 `cuda`로 함께 지정한다. 아래는 128판을 한 배치로 생성하고
버퍼 학습·평가·저장까지 한 번 실행하는 설정 예시다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --self-play-backend cuda --self-play-batch-size 128 --evaluation-workers 12 --iterations 1 --games-per-iteration 128 --simulations 32 --train-steps 8 --batch-size 64 --eval-games 20 --eval-simulations 32 --output runs\gpu-rl
```

자가 대국 루트 잡음 비율 0.25와 착수 온도 1.0은 기본적으로 켜져 있다.
기존 잡음 없는 자가 대국 벤치마크의 36.79판/초를 이 학습의 속도로 가정하지
않는다. CLI와 `metrics.jsonl`에 실제 자료 생성 처리량, 자료 생성·학습·평가
시간과 전체 반복 처리량을 기록한다. 첫 반복의 자료 생성 시간에는 최초 CUDA
커널 준비가 포함될 수 있다.

중단 후에는 CUDA 장치를 다시 지정하고, 자가 대국 경로와 배치 크기는
체크포인트에서 복원한다. 평가 worker는 버전 4 이상에서, backend·leaf 배치·트리
재사용 여부는 버전 5 이상에서 복원된다. 이전 형식은 `legacy` 평가로 복원한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\gpu-rl\latest.pt --iterations 1 --output runs\gpu-rl
```

`--self-play-backend cuda`는 CUDA 학습 장치를 필요로 한다. CUDA 자가 대국
체크포인트를 CPU 장치로 재개하면 오류를 표시한다. `--self-play-batch-size`는
동시에 진행할 대국 수이고, `--batch-size`는 가중치 갱신에 사용하는 위치 수다.
`--threads`는 PyTorch CPU 연산 스레드 설정이며 GPU 동시 대국 수와 별개다.

### GPU 동시 대국 배치 늘리기

`--self-play-batch-size`로 동시 대국 상한을 늘린다. 한 반복의 대국 수인
`--games-per-iteration`도 그 이상이어야 실제로 그 크기의 배치를 사용한다.
예를 들어 반복당 128판을 설정한 상태에서 배치 상한만 1024로 바꾸면 실제로는
128판을 함께 실행한다. `--batch-size`는 별도의 가중치 갱신용 위치 수다.

같은 초기 가중치에서 동시 대국 배치만 비교하는 실행기를 제공한다.

```powershell
.\.venv\Scripts\python.exe examples\gpu_trainer_benchmark.py --games 1024 --batch-sizes 128 256 512 1024 --repeats 1 --initial-model runs\gpu-trainer-validation-2026-10-05\best.pt --output runs\batch-scaling.jsonl
```

각 설정은 같은 가중치에서 새 버퍼·optimizer로 한 번씩 시작한다. 수당 32회
탐색, 루트 잡음 0.25, 착수 온도 1, 학습 미니배치 64개·갱신 8회, 평가 2판을
사용한다. 준비 작업을 제외한 자료 생성·버퍼 저장·학습·평가 시간과 최대 GPU
tensor 메모리를 기록한다. 파일 저장 시간은 반복 처리량에 포함하지 않는다.
동시 대국 묶음이 달라지면 난수 소비와 수순도 달라지므로 생성 위치 수와
평균 수순을 함께 비교한다. 실제 결과는 [성능 기록](performance.md)에 있다.

현재 모델·32회 탐색 조건에서는 배치 1024개로 실제 자료 생성·CUDA 학습·평가를
완료했고, 자료 생성 23.03판/초와 학습·평가 포함 22.13판/초를 측정했다.
같은 1024판 생성 조건의 배치 128개는 각각 11.38판/초와 10.75판/초였다.
각 설정 1회 실측이며 파일 저장 시간은 제외한다.

현재 환경의 기준 모델 가중치에서 배치를 늘린 새 실행을 시작하는 예시다.
자가 대국 배치 크기와 반복당 판수는 1024개, 가중치 갱신용 미니배치는 64개다.
새 실행이므로 optimizer와 버퍼는 새로 구성한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --self-play-backend cuda --self-play-batch-size 1024 --games-per-iteration 1024 --simulations 32 --batch-size 64 --eval-games 20 --eval-simulations 32 --initial-model runs\gpu-trainer-validation-2026-10-05\best.pt --output runs\gpu-1024
```

이 시작 예시는 평가 20판을 지정한다. 위 처리량 실측은 평가 2판을 사용했다.
기존 `--resume`은 저장된 배치 크기와 반복당 판수를 복원하며 설정 변경을
허용하지 않는다.

### 자료 보존과 학습량을 함께 늘리는 설정

`runs/gpu-r1`의 1024판 실행은 27,079개 위치를 생성했지만 버퍼가 10,000개여서
17,079개가 학습 전에 밀려났다. 미니배치 64개·갱신 32회는 복원추출
2,048회에 해당한다. 평가 20판·탐색 32회는 53.28초로 반복 시간의 약 52%였다.

같은 저장 버퍼에서 32,768개 위치를 추출하는 CUDA 학습을 배치별로 두 번씩
측정했다. 배치 64/128/256/512의 중앙 시간은 각각 6.638/3.784/1.968/1.102초였다.
배치 512는 이 측정 범위에서 위치 처리 속도가 가장 높았다. 갱신 횟수가 서로
다르므로 이 결과로 기력이나 수렴 속도가 더 좋다고 판단하지 않는다.

```powershell
.\.venv\Scripts\python.exe examples\training_batch_benchmark.py --checkpoint runs\gpu-r1\latest.pt --sample-draws 32768 --repeats 2 --output runs\training-batch-comparison.jsonl
```

측정에는 버퍼 추출·GPU 전송·학습을 포함하며 준비와 파일 저장은 제외한다.
입력 체크포인트는 수정하지 않고, 비교용 모델은 저장하지 않는다.

자료 손실을 줄이고 평가 비용을 더 많은 자가 대국에 분산하는 시작 설정이다.
자가 대국은 여전히 한 묶음에 1024판이며, 두 묶음을 끝낸 뒤 학습한다.
학습은 512개 위치·128회 갱신으로 총 65,536회 복원추출한다. 모든 위치를
한 번씩 학습하는 보장은 없다. 평가 20판·탐색 32회와 승격 기준 0.55는 유지한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --self-play-backend cuda --self-play-batch-size 1024 --iterations 1 --games-per-iteration 2048 --simulations 32 --replay-capacity 131072 --train-steps 128 --batch-size 512 --eval-games 20 --eval-simulations 32 --threads 1 --initial-model runs\gpu-r1\best.pt --output runs\gpu-r1-tuned
```

버퍼는 CPU 메모리를 최대 약 457MiB 사용하고, 채워진 버퍼가 체크포인트에
저장되므로 디스크 사용과 저장 시간도 늘어난다. 131,072개는 관측한 평균
수순을 바탕으로 정한 용량이며, 모든 가능한 2048판 수순의 위치를 보존하는
상한은 아니다. 반복이 이어지면 오래된 위치부터 교체한다.

이 설정으로 실제 2048판·학습 128회·평가 20판을 완료했다. 생성한 52,377개
위치를 모두 보존했고, 자료 생성은 22.07판/초, 학습·평가 포함 반복 처리량은
12.59판/초였다. 기존 기준 모델 상대 15승 5패로 새 모델을 승격했다.
파일 저장은 처리량에서 제외하며, 이전 실행과 출발 가중치가 다르므로
설정 변경의 효과만을 분리한 비교는 아니다. [상세 실측](performance.md)에
시간·자료 보존·체크포인트 검증을 기록했다.

이 실행은 기존 기준 모델의 가중치를 가져오며 optimizer와 버퍼는 새로
시작한다. 완료된 설정을 그대로 이어갈 때는 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\gpu-r1-tuned\latest.pt --iterations 10 --output runs\gpu-r1-tuned
```

## 설정

최근 학습의 색상 편향 진단과 전술 확인·FPU·자료 증강·기존 학습 상태를
보존하는 설정 변경은 [기력 개선 안내](strength_optimization.md)를 참고한다.

| CLI 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `--iterations` | 1 | 이번 실행의 추가 반복 수 |
| `--output` | `runs/rl` | 저장 파일과 지표를 기록할 디렉토리 |
| `--games-per-iteration` | 4 | 한 반복의 완료 자가 대국 수 |
| `--self-play-backend` | `cpu` | `cpu`: 기존 C++ 순차 대국; `cuda`: GPU 규칙·배치 PUCT |
| `--self-play-batch-size` | 128 | CUDA 경로에서 동시에 진행할 최대 대국 수 |
| `--simulations` | 128 | 자가 대국의 매 수 PUCT 탐색 횟수 |
| `--c-puct` | 1.5 | PUCT 탐색 계수 |
| `--dirichlet-alpha` / `--dirichlet-epsilon` | 0.3 / 0.25 | 자가 대국 루트 잡음의 분포와 혼합 비율 |
| `--train-steps` | 8 | 한 반복의 미니배치 가중치 갱신 횟수 |
| `--batch-size` | 64 | 미니배치의 위치 수 |
| `--replay-capacity` | 10,000 | FIFO 버퍼에 보관할 최대 위치 수 |
| `--learning-rate` | 0.001 | Adam 학습률 |
| `--weight-decay` | 0.0001 | Adam 가중치 감쇠 |
| `--eval-games` | 20 | 한 반복의 평가 대국 수, 2 이상 짝수 |
| `--eval-simulations` | 128 | 평가 대국의 매 수 PUCT 탐색 횟수 |
| `--evaluation-workers` | 새 학습 1 / 재개 시 저장값 | 동시 평가 대국 수; 모델별 추론 요청을 batch로 묶음 |
| `--evaluation-backend` | 새 학습 `legacy` / 재개 시 저장값 | `batched_cpp`는 C++ leaf 배치·입력 생성 경로 |
| `--evaluation-leaf-batch-size` | 새 학습 8 / 재개 시 저장값 | `batched_cpp` 한 탐색의 미평가 leaf 상한 |
| `--evaluation-reuse-tree` | 새 학습 켜짐 / 재개 시 저장값 | `batched_cpp` 모델별 하위 트리 재사용; `--no-`로 해제 |
| `--eval-opening-moves` / `--eval-opening-temperature` | 6 / 1.0 | 평가 초반 방문 수 착수 샘플링 |
| `--promotion-threshold` | 0.55 | 후보 승률이 이 값 이상이면 승격 |
| `--temperature` | 1.0 | 자가 대국의 방문 수 착수 샘플링 온도 |
| `--seed` | 42 | 모델 초기화와 자료 생성·표본 추출의 시드 |
| `--channels` / `--residual-blocks` | 32 / 2 | 새 모델의 구성 |
| `--device` | `cpu` | PyTorch 추론과 학습 장치 |
| `--threads` | 1 | CLI 실행의 PyTorch CPU 스레드 수 |
| `--online-tactics` | 꺼짐 | 새 자가 대국의 일부 위치를 깊게 증명하고 미니배치에 혼합 |
| `--online-tactics-max-cases` | 32 | 사이클당 확인할 CPU 전술 위치 상한 |
| `--online-tactics-max-depth` | 9 | 양쪽 착수를 합친 증명 깊이(ply) |
| `--online-tactics-max-nodes` | 2,000,000 | 위치당 증명 탐색 노드 상한 |
| `--online-tactics-time-limit-ms` | 2,000 | 위치당 CPU 증명 시간 예산(ms) |
| `--online-tactics-generation-seconds` | 30 | 사이클당 전술 자료 생성 시간 예산; 전체 사이클 상한은 아님 |
| `--online-tactics-fraction` | 0.25 | 증명 버퍼가 있을 때 미니배치의 전술 표본 비율 |
| `--online-tactics-replay-capacity` | 1,024 | 일반 replay와 분리된 증명 위치 FIFO 용량 |
| `--online-tactics-min-proof-depth` | 3 | 채택할 승리 증명의 최소 수순 길이(ply) |

같은 설정을 Python의 `TrainingConfig`에서도 지정할 수 있다.
라이브러리를 가져오는 것만으로 CPU 스레드 설정을 바꾸지는 않는다.

자료 버퍼의 단위는 판이 아닌 **착수 전 위치**다. 버퍼가 가득 차면 가장
오래된 위치부터 버린다. 스키마 1의 CPU tensor 저장량은 위치당 3,655바이트로,
10,000개 용량은 약 34.9MiB다. 모델, optimizer, 배치와 저장 중 사본의 메모리는
별도로 필요하다. 배치는 복원추출하므로 버퍼에 있는 위치 수보다 크게 설정할 수 있다.

다음은 자가 대국 수를 설정상 40만 판으로 만드는 예시다. 실제 실행·속도·기력은
이 설정으로 검증하지 않았다. 자료 생성량에 대한 학습 갱신 비율과 평가 횟수는
짧은 실행에서 손실과 대국 결과를 확인하며 조절해야 한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --iterations 1000 --games-per-iteration 400 --simulations 128 --train-steps 200 --batch-size 256 --replay-capacity 100000 --eval-games 20 --eval-simulations 128 --output runs\long-run
```

이 경우 1,000회 × 400판 = 400,000판의 **자가 대국**이며, 매 반복의 평가
대국은 별도로 수행한다. 이전에 계산한 예상 시간은 이 프로그램의 전체 학습
시간 실측치가 아니다. 지표의 자료 생성·학습·평가 시간을 각각 확인한다.

## 온라인 장기 전술을 학습 루프에 연결하기

`--online-tactics`는 매 사이클 생성한 대국에서 일부 비종료 위치를 뽑고,
실제 착수 기록으로 재현하여 깊은 CPU 증명 탐색을 수행한다. 현재 플레이어의
승리가 확정된 위치 중 최소 증명 길이 조건을 만족한 것만 별도 전술 FIFO에
저장한다. 증명된 승리 수를 정책 목표로, `+1`을 가치 목표로 사용한다.
`UNKNOWN`을 안전·패배·무승부 라벨로 바꾸지 않으며 `LOSS`도 임의의 정책
정답으로 채택하지 않는다. 실제 대국에서 관찰한 승패 목표는 일반 replay에
그대로 유지한다.

일반 replay와 증명 replay는 분리되어 있으므로 전술 위치가 기존 전략 자료를
밀어내지 않는다. 증명 버퍼가 있으면 기본적으로 미니배치의 25%를 전술에서,
75%를 일반 replay에서 복원추출한다. 증명 위치가 아직 없으면 원래 미니배치로
학습한다. 새 정답을 못 찾은 사이클도 이전 증명 자료를 계속 사용할 수 있다.
설정한 `--train-steps` 안에서 혼합하므로 사이클당 Adam 갱신 수는 증가하지
않는다. 전술 자료에는 회전·반사를 적용하기 전의 원본 증명도 함께 저장한다.

이 기능은 자가 대국의 모든 수에 깊은 전술 탐색을 추가하지 않는다. 자가 대국의
PUCT 예산, CUDA의 기존 얕은 검사, 평가 예산과 champion 승격 기준은 그대로다.
새 긴 강제 승리 문제를 매 사이클 발견한다고 보장하지 않으며 추가 CPU 시간도
필요하다. 자세한 안전 조건·옵션·실행 방법은
[증명 기반 전술 학습](tactical_curriculum.md#매-사이클-새로운-전술을-계산하여-학습하기)에 있다.

기존 v6 전술 보강 체크포인트에서 설정만 준비하는 예시다. 원본을 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe examples\train.py `
  --resume runs\tactics-candidate-2026-10-07\latest.pt `
  --device cuda --reconfigure --prepare-only --online-tactics `
  --output runs\tactics-online-2026-10-07
```

기본 teacher 예산은 위 설정 표에 나온 값이며 새 상태에 저장된다. 이후 아래처럼
재개하면 온라인 전술 설정·전술 버퍼도 함께 복원한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py `
  --resume runs\tactics-online-2026-10-07\latest.pt `
  --device cuda --iterations 10 --output runs\tactics-online-2026-10-07
```

시작 옵션을 지정하지 않은 새 실행과 v1~v6의 일반 재개에서는 온라인 전술이
꺼져 있다. 온라인 전술을 꺼도 저장된 증명 버퍼는 보존하며, 다시 켤 때 사용할
수 있다. 끄거나 예산을 바꿀 때도 `--reconfigure`와 새 출력 폴더를 사용한다.

## 저장 파일과 재개 범위

| 파일 | 내용 | 읽는 API |
| --- | --- | --- |
| `latest.pt` | learner와 champion, Adam, 일반·증명 버퍼, 설정, 진행 횟수, 난수 상태 | `Trainer.load_checkpoint` / `--resume` |
| `best.pt` | 현재 champion의 스키마·모델 구성·가중치 | 기존 `load_model` |
| `metrics.jsonl` | 완료한 반복마다 한 줄의 JSON 지표 | 일반 JSON Lines 도구 |

저장한 `best.pt`와 사람이 직접 대국하려면 `examples/play_ai.py`를 사용한다.
현재 저장 자료의 위치와 범위, CPU·CUDA 대국 실행과 입력 방법은
[직접 대국 안내](play_ai.md)에 기록한다.

`latest.pt`는 모델 가중치만 저장하던 파일과 형식이 다르다. 두 종류는 서로의
복원 API로 읽지 않는다. 전체 체크포인트는 `weights_only=True`로 읽고 입력
스키마, 가중치·Adam·버퍼·진행 횟수·난수 상태의 일관성을 검증한다.

현재 전체 체크포인트는 버전 7이다. 버전 1~6도 읽는다. 버전 1~5 파일에 없는
`tactical_training_steps`는 0으로, 버전 1~6에 없는 온라인 전술 설정·증명
버퍼는 꺼짐·빈 버퍼로 복원한다. 버전 7은 `online_tactical_replay`에 증명
자료·메타데이터를 별도로 저장하고 복원 시 검증한다. 버전 6에서 도입한 완료
사이클 외 증명 기반 전술 갱신 카운터도 유지한다. 진행 횟수 검증식은 다음과 같다.

```text
self_play_games = iteration × games_per_iteration
training_steps = iteration × train_steps_per_iteration + tactical_training_steps
```

`training_steps`는 실제 전체 Adam 갱신 수이고 `tactical_training_steps`는 그중
추가 전술 갱신 수다. 전술 보강으로 자가 대국 판수·완료 사이클·champion 버전을
늘리지 않는다. `Trainer.train_tactical_batch()`는 성공한 갱신에만 두 학습
카운터를 늘리고, 실패한 부분 상태의 저장을 금지한다. 모델·optimizer·버퍼·
전용 Generator는 이어받고 champion은 자동 교체하지 않는다.
온라인 전술 혼합은 사이클 안의 기존 갱신을 사용하므로 별도 추가 갱신 카운터를
늘리지 않는다. 이미 증명 자료가 있는 전술 replay의 용량 변경은 거부하고,
온라인 기능을 끄는 경우에도 자료를 유지한다.

전술 보강 후 저장한 `last_metrics`의 과거 평가 결과는 갱신한 learner의 새
평가가 아니다. `tactical_finetuning_since_evaluation`로 이를 표시하며 실제
학습 전후 전술 지표는 별도 `tactical_report.json`에 기록한다. 버전 6의 전술
보강 출력은 기존 전체 체크포인트 재개 API로 읽을 수 있다. 명령과 정답 생성의
안전 조건은 [증명 기반 전술 학습](tactical_curriculum.md)에 설명한다.

체크포인트는 임시 파일을 쓴 뒤 교체하며, 반복 시작 전과 완료 후에 저장한다.
`Ctrl+C`로 중단하거나 반복 도중 오류가 발생하면 **마지막으로 완료한 반복**의
체크포인트에서 재개한다. 진행 중이던 대국·학습·평가는 그 반복을 처음부터
다시 실행한다. Python API에서도 불완전한 반복을 그대로 저장하거나 계속
실행하지 못하게 한다.

전용 CPU Generator가 CPU의 판별 시드 또는 GPU의 배치별 시드와 버퍼 표본
추출을 관리한다. GPU 자료 수집기는 이 시드로 전용 CUDA Generator들을
초기화하며 완료 반복 경계에서 난수 흐름을 복원한다.
전체 체크포인트에는 이 Generator와 Python·PyTorch CPU·CUDA 난수 상태를
저장한다. 장치·PyTorch 버전 등을 바꾸면 부동소수점 연산 결과가 달라질 수 있다.
온라인 증명 탐색의 wall-clock 제한이 걸린 위치는 CPU 부하에 따라 증명 성공
여부가 달라질 수 있다. 같은 시드·저장 상태만으로 시간 제한 teacher의 비트
단위 재현성까지 보장하지 않는다.

## 평가와 기록 해석

평가에서는 후보가 흑인 판과 백인 판을 짝으로 진행하며 두 판에 같은 시드를
사용한다. 루트 Dirichlet 잡음과 시간 제한은 사용하지 않는다. 기본적으로
처음 6수는 온도 1로 방문 수에서 착수를 뽑고, 이후에는 가장 많이 방문한 수를
선택한다. 동일 모델끼리의 같은 시드 흑백 쌍은 정확히 1승 1패가 된다.

기본 `legacy`의 `evaluation_workers=1`은 기존 순차 경로를 그대로 사용한다. 2 이상이면
대국마다 독립 C++ PUCT를 두고 후보·기준 모델별 추론 요청을 두 개의 batch
서비스로 묶는다. 동적 batch 구성의 부동소수점 순서가 승격 결과에 영향을 줄
가능성이 있으므로 worker 수를 버전 4 체크포인트에 저장한다. 재개 시 지정하지
않으면 저장값을 복원하고, 명시하면 그 실행부터 덮어쓴 뒤 다음 저장에 보존한다.
버전 1~3 파일에는 이 값이 없어 worker 1로 복원한다. 처음 병렬화할 때만
`--evaluation-workers 12`처럼 지정하면 된다. 지정값이 평가 판수보다 크면 실제
worker는 평가 판수로 제한된다. 동일 worker의 CPU split-resume와 짧은 CUDA
비교는 일치했다. 다만 병렬 경로는 같은 worker 수에서도 thread 실행 순서에
따라 추론 batch 구성이 달라질 수 있어 CUDA의 비트 단위 재현성을 보장하지
않는다. 장치·PyTorch 버전 변경도 부동소수점 결과에 영향을 줄 수 있다.

기본 20판이면 후보가 11승 이상 했을 때 55% 기준을 통과한다. 이 값은 현재
기준 모델과의 관측 승률이며, 절대 기력이나 모든 상대에 대한 승률은 아니다.
적은 평가 판수는 편차가 크므로 장기 학습의 기준은 추가 검증으로 정한다.

지표에는 완료 반복 수, 누적 자가 대국·가중치 갱신 수, 기준 모델 버전,
이번에 만든 위치 수, 버퍼 크기, 평균 수순, 종료 사유, 세 손실, 흑백별 후보
승수와 승격 여부, 자료 생성·학습·평가의 경과 시간이 들어간다.
온라인 경로는 `online_tactics_seconds`와 중첩
`online_tactics` 보고서도 기록한다. 이 보고서에는 후보·검사한 위치, 증명
결과와 제외 사유, 새 채택 표본 수 `added_samples`, 증명 버퍼 크기 `replay_size`,
혼합 갱신 수 `mixed_updates`, 미니배치별 `tactical_rows_per_batch`·
`replay_rows_per_batch`, 실제 teacher 시간 `seconds`가 들어간다. CLI에도
채택 수·버퍼·혼합 비율·시간을 출력한다. 증명 위치 채택 수가 곧 전체 기력
향상이나 장기 전술 일반화 성능을 의미하지 않는다.
`self_play_backend`와 `self_play_batch_size`는 실행 경로와 설정을 나타낸다.
`self_play_games_per_second`는 이번 완료 판수를 자료 생성 시간으로 나눈 값,
`iteration_games_per_second`는 같은 판수를 학습·평가까지의 전체 반복 시간으로
나눈 값이다. 전체 반복 시간에는 완료 후 체크포인트 쓰기와 콜백 시간은
포함하지 않는다.
손실 감소와 후보 승격을 함께 살피고, 평가 예산과 상대가 바뀌었는지도 확인한다.

### 성능·버퍼 지표 분석

새 지표 스키마 2는 자료 생성 시간을 GPU 계산·자료 구성 구간과 CPU replay
저장 구간으로 나누고 다음 값을 추가한다.

- `self_play_positions_per_second`: 판 길이 변화에 덜 왜곡되는 위치 처리량
- `self_play_compute_seconds`, `replay_store_seconds`: 자가 대국과 버퍼 적재 시간
- 평균·95백분위·최대 수순, 자가 대국 흑백 승수와 종료 사유
- replay 용량·회전율·이번 자료 보존 비율
- 반복당 학습 추출 수와 생성 위치/버퍼 위치 대비 추출 비율
- 이번 실행의 평가 worker 수

`Trainer.run()`이 파일을 저장할 때 반환값·`metrics.jsonl`·callback에는
`initial_checkpoint_seconds`, `checkpoint_seconds`, `checkpoint_bytes`와
`checkpoint_written`, `elapsed_with_checkpoint_seconds`도 기록한다.
`checkpoint_written`은 실제 파일을 쓴 실행과 파일 없이 API만 실행한 경우를
구분한다. 저장 시간은 저장이 끝난 뒤에만
알 수 있으므로 같은 체크포인트 내부의 `last_metrics`에는 포함하지 않는다.
`elapsed_seconds`는 자가 대국·온라인 teacher·학습·평가를 포함한 반복 계산
시간이다. 온라인 경로가 꺼져 있으면 teacher 계산은 없다.
`elapsed_with_checkpoint_seconds`도 JSONL append와 callback의 `best.pt` export
시간은 포함하지 않으므로 전체 체감 wall-clock과 동일하다고 간주하지 않는다.

기존 및 새 지표 파일은 체크포인트를 읽지 않는 분석기로 함께 요약한다.

```powershell
.\.venv\Scripts\python.exe examples\analyze_training.py runs\gpu-r1-tuned\metrics.jsonl --recent 10 --target-iterations 300
```

출력은 전체·최근 위치/초와 판/초, 평균 수순, 단계별 시간 비중, 후보의 흑백별
평가 승률, 승격률, replay가 최근 몇 사이클을 담는지와 남은 기록 시간 추정을
보여준다. 구형 행에는 체크포인트·replay 세부 시간이 없으므로 이를 0으로
간주하지 않고 기록 범위를 따로 표시한다. `--json`으로 기계 판독 결과를 얻는다.

CLI의 일반 학습 실행은 `Trainer.run()`의 시작 경계 저장만 사용한다. 이 저장이
완료된 뒤 대국을 시작하고, `best.pt`는 첫 반복 완료 후 callback에서 내보낸다.
따라서 초기 저장 실패 때 `best.pt`만 남지 않으며, 이전처럼 같은 대형
`latest.pt`를 실행 직전에 두 번 연속 저장하지 않는다. `--prepare-only`도
`latest.pt`를 먼저 저장한 뒤 `best.pt`를 내보낸다.

### 사이클당 판수와 replay 용량 조정 기준

`gpu-r1-tuned`의 최근 10사이클은 2,048판에서 평균 14.49수, 약 29,670개
위치를 만들었다. 현재 131,072 위치 버퍼는 최근 생성량 기준 약 4.41사이클을
담는다. 초기 장기 실행처럼 평균 수순이 25.6수라면 약 2.50사이클이다. 따라서
버퍼는 고정 GB보다 **최근 몇 사이클을 보존하는지**로 판단한다.

새 전술·128회 탐색 설정은 대국 길이를 바꿀 수 있으므로 첫 3~5사이클 동안은
2,048판·131,072 위치를 유지하고 실제 생성 위치를 먼저 측정한다. 이후 목표
3~5사이클에 맞춰 대략 `판수 × 평균 수순 × 목표 사이클`로 용량을 계산한다.
위치당 tensor payload는 3,655바이트이며 버퍼를 키우면 RAM뿐 아니라 매 경계의
체크포인트 크기와 저장 시간도 같은 비율로 늘어난다.

현재 512개 × 128회 갱신은 사이클당 65,536개 위치를 복원추출한다. 가득 찬
131,072 버퍼의 슬롯당 평균 0.5회이며, 균등 복원추출에서 한 사이클에 적어도
한 번 뽑힐 것으로 기대되는 고유 슬롯은 약 39%다. 버퍼만 두 배로 키우면 이
비율이 낮아지므로 학습 추출량도 함께 비교해야 한다.

평가 비용을 분산하려고 사이클당 판수를 4,096로 늘리는 실험은 가능하지만,
기준 모델 갱신이 절반 빈도로 늦어지고 같은 비율을 유지하려면 학습 step과
버퍼도 함께 조정해야 한다. 현재 체크포인트는 판수·step·용량 변경을 재개 중에
허용하지 않는다. 병렬 평가는 축소 A/B에서 12 worker가 순차보다 2.72배
빨랐으므로 먼저 2,048판 설정을 유지한 채 실제 128회 탐색 사이클의 새 처리량과
기력 지표를 측정한다. 그 뒤 4,096판을 독립 A/B 실행으로 비교한다. 단순히
1,024판으로 줄이면 고정 평가 비중이 커질 수 있어 속도 최적화의 기본값으로
권하지 않는다.

실제 새 설정의 첫 사이클은 64,952개 위치·평균 31.715수였고, 131,072 버퍼는
이 생성량 기준 약 2.02사이클을 담았다. 모든 새 위치를 보존했으며, 현재는
이 용량과 2,048판을 유지해 기존 편향 자료가 교체되는 동안 추이를 확인한다.
3~5사이클을 담으려면 이번 생성량으로 약 195,000~325,000 위치가 필요하지만,
한 번의 측정만으로 용량을 키우지 않는다. 용량 변경은 별도 실행에서 학습
추출량·자료 다양성·후보 평가와 함께 비교한다.

측정 실행은 완료 244사이클까지 별도 저장했고, 준비한 `gpu-strength-v2` 원본은
243사이클 상태 그대로다. 측정한 상태를 이어가려면 다음 명령을 사용한다.
worker 12는 체크포인트에서 복원된다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\gpu-strength-speed-pilot-2026-10-06\latest.pt --iterations 3 --output runs\gpu-strength-speed-pilot-2026-10-06
.\.venv\Scripts\python.exe examples\analyze_training.py runs\gpu-strength-speed-pilot-2026-10-06\metrics.jsonl --recent 4
```

첫 사이클의 체크포인트 포함 기록 시간은 314.40초였다. 평가 206.40초가 계산
시간의 66.3%이므로 사이클당 판수를 조정할 때 평가 비용과 champion 갱신
간격을 함께 판단한다. 상세 측정 범위는 [성능 기록](performance.md)에 있다.

## Python API

```python
from kingdom_ai import Trainer, TrainingConfig, load_model

config = TrainingConfig(
    games_per_iteration=2, simulations=4, evaluation_games=2,
    evaluation_simulations=4, train_steps_per_iteration=2, batch_size=16,
)
trainer = Trainer(config, device="cpu", evaluation_workers=4)
trainer.run(1, checkpoint_path="runs/api/latest.pt", metrics_path="runs/api/metrics.jsonl")
trainer.export_champion("runs/api/best.pt")

resumed = Trainer.load_checkpoint("runs/api/latest.pt", device="cpu")
resumed.run(1, checkpoint_path="runs/api/latest.pt", metrics_path="runs/api/metrics.jsonl")
resumed.export_champion("runs/api/best.pt")
champion = load_model("runs/api/best.pt", device="cpu")
```

`Trainer.run`은 기본적으로 완료 반복의 지표 목록을 반환한다. 장기 실행에서는
`collect_metrics=False`를 지정하여 그 목록이 메모리에 누적되지 않게 한다.
지표 파일과 `on_iteration` 콜백은 이 설정에서도 계속 기록·호출된다.
CLI는 항상 `collect_metrics=False`로 실행한다.

하위 구성은 `ReplayBuffer`와 `evaluate_models`로 따로 사용할 수 있다.
버퍼의 `sample(batch_size, generator=..., device=...)`에는 호출자가 관리하는
CPU `torch.Generator`가 필요하다. `evaluate_models(candidate, reference, ...)`는
후보 관점의 승패와 흑백별 승수를 반환하며 `workers`로 병렬 대국·batch 추론을
선택한다. 기본값 1은 이전 순차 실행과 같다.

## 검증

버퍼의 FIFO 교체·고정 메모리·사본 분리·시드 표본 추출·저장 복원,
흑백 교대 평가·실제 종료·모델 보존, 완료 반복·승격·거부·중단 재개를 검사한다.
검증 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe tests\replay_test.py
.\.venv\Scripts\python.exe tests\evaluation_test.py
.\.venv\Scripts\python.exe tests\reinforcement_test.py
.\.venv\Scripts\python.exe tests\gpu_training_test.py
.\.venv\Scripts\python.exe tests\gpu_trainer_test.py
ctest --test-dir build-python -C Release --output-on-failure
```

CTest는 `BUILD_PYTHON_BINDINGS=ON`, `BUILD_NEURAL_TESTS=ON`과 PyTorch가 설치된
Python으로 구성한 빌드를 사용한다. 테스트와 짧은 실행은 프로그램의 동작을
검증하며, 대규모 학습의 완료나 기력 달성을 의미하지 않는다.

GPU 통합 검증은 기존 CPU 엔진과 착수 기록·입력·정책·최종 가치 목표를
대조하고, FIFO 자료 저장·CUDA 가중치 갱신·모델 승격 반영·완료 반복 저장과
재개를 확인한다. CUDA가 없는 환경에서는 GPU 검사를 생략한다.

### 검증 기록: 2026-10-02

- 기존 엔진·바인딩·신경망·PUCT 검증을 포함한 CTest 11개 묶음이 모두 통과했다.
- 새 Python 검사는 버퍼 13개, 흑백 교대 평가 6개, 반복 학습·저장 재개 11개로
  총 30개다. 같은 CPU 설정에서 연속 학습과 저장 후 재개의 가중치·Adam·버퍼·
  진행 횟수·전용 Generator 결과가 일치함을 확인했다.
- 소스 패키지를 사용하는 CLI로 1회 실행하고 저장 파일에서 1회를 추가로
  재개했다. 완료 반복 2회, 누적 자가 대국 4판, 평가 총 4판, 가중치 갱신 4회,
  최대 128개로 제한된 버퍼와 JSON Lines 2줄을 확인했다.
- 최종 패키지를 빌드·설치한 뒤 소스 경로 지정 없이 CLI에서 1회를 더 재개했다.
  완료 반복 3회, 누적 자가 대국 6판, 평가 총 6판, 가중치 갱신 6회, 버퍼 128개와
  JSON Lines 3줄을 확인했다. 다른 실행의 기록이 있는 출력 폴더를 거부하고
  기존 파일을 보존하는 것도 확인했다.
- 이 짧은 실행은 8채널·잔차 블록 1개, 자가 대국과 평가 각각 수마다 4회
  탐색한 연결 검사다. 모델 기력이나 목표 처리량 달성을 검증하지 않았다.
- `engine/` 파일은 변경하지 않는다.

### GPU 연결 검증: 2026-10-05

- 완료 GPU 대국의 입력·합법 수·방문 정책·가치 관점을 기존 CPU 엔진과 비교했다.
  루트 잡음과 착수 샘플링의 재현성, 배치 마지막의 남은 대국, 종료된 대국의
  자료 제외, 원본 모델과 전역 난수 보존도 검사했다.
- GPU 자가 대국·CPU 버퍼·CUDA 가중치 갱신·기존 평가를 실제로 실행했다.
  동일한 CUDA 환경과 결정적 cuDNN 설정에서 연속 2회 학습과 1회 학습 후
  저장·재개한 결과의 생성 자료·미니배치·가중치·Adam·버퍼·난수·진행 횟수가
  정확히 일치했다. 장치·PyTorch 설정이 달라지면 같은 결과를 보장하지 않는다.
- GPU 자료 생성 테스트 6개와 Trainer 연결 테스트 5개를 추가했다.
  기존 엔진·바인딩·학습 검사를 포함한 CTest 17개 묶음이 모두 통과했다.
- 기존 버전 1 체크포인트는 CPU 경로로 복원하며, 버전 2는 새 자가 대국
  설정을 함께 저장한다. GPU 체크포인트 재개 시 CUDA 장치를 요구한다.
- 설치한 패키지에서 128판 배치·수당 32회 탐색·잡음 0.25·온도 1로 실제
  3회 반복을 완료했다. 자가 대국 384판, 평가 6판, CUDA 갱신 24회, FIFO
  버퍼 10,000개와 프로세스 종료 후 체크포인트 재개를 확인했다. 검증 파일은
  `runs/gpu-trainer-validation-2026-10-05/latest.pt`, 기준 모델은 같은 폴더의
  `best.pt`다. 단계별 측정과 범위는 [성능 기록](performance.md)에 있다.

### 병목 개선 검증: 2026-10-06

- 기존 C++ 규칙·바인딩·GPU PUCT·자료 생성·학습·재개와 새 지표 분석을 포함한
  CTest 22개 묶음이 모두 통과했다(240.84초).
- 4,096개를 넘는 벌크 적재, ring 경계와 용량 초과 입력의 FIFO 순서,
  잘못된 마지막 표본을 포함한 입력의 사전 검증 원자성을 검사했다.
- CPU 병렬 평가를 사용하는 연속 학습과 저장·재개 결과가 정확히 일치했다.
  짧은 CUDA 평가는 순차·병렬 비교와 병렬 3회 반복 결과가 일치했고, 추론
  실패 시 worker 종료와 모델 mode 복원도 확인했다.
- 체크포인트 버전 1~3을 읽고 worker 1로 복원하며, 버전 4의 worker 저장·복원과
  명시적 변경, 잘못된 runtime 설정 거부를 검사했다.
- 일반 CLI와 prepare-only 각각의 최초 저장에 OSError·KeyboardInterrupt를
  주입한 네 경우 모두 `best.pt`만 남지 않았고, 초기 중단 안내가 올바르다.
- 프로젝트 가상환경을 오프라인으로 갱신하고 설치된 패키지의 새 API와 지표
  분석기를 실행했다. `engine/` 파일은 변경하지 않았다.
