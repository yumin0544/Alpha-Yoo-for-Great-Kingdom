# GPU 규칙 판정과 PUCT

`GpuStateBatch`와 `GpuPUCT`는 대국 상태, 규칙 판정, 합법 수 마스크,
입력 인코딩, 트리 선택·확장·역전파와 신경망 추론을 CUDA에서 수행한다.
기존 CPU 엔진은 변경하지 않고 규칙 일치 검증의 기준으로 사용한다.
CPU C++ 탐색과 GPU 신경망 평가를 결합하는 기존
[배치 추론 실행기](gpu_benchmark.md)와 실행 경로가 다르다.

Python은 커널과 모델을 호출하는 반복문을 진행한다. 평가할 잎을 모으는
`torch.nonzero`의 크기 정보 동기화와 탐색 종료 후 집계 오류 코드의 CPU 회수가
있다. 보드·규칙·트리 계산을 GPU에서 수행한다는 뜻이며 GPU 사용률이 항상
100%라는 뜻은 아니다. 완료 대국 처리량과 실제 대국 길이를 함께 측정한다.

## 실행 환경과 초기 준비

CUDA용 PyTorch, 호환되는 NVIDIA 드라이버와 NVRTC 라이브러리가 필요하다.
현재 머신에서는 PyTorch 2.9.1+cu126에 포함된 NVRTC 12.6을 사용해
RTX 3050의 `compute_86`용 커널 컴파일과 실행을 확인했다.
이 GPU 경로를 실행하기 위해 추가 C++ 컴파일러나 CUDA Toolkit을
다운로드·설치하지 않았다. 기존 Python 확장 모듈을 새로 빌드하는 데 필요한
C++ 환경은 [설치 안내](gpu_benchmark.md)의 별도 요구사항이다.

`gpu_runtime.py`는 CUDA C++ 소스 문자열을 NVRTC로 PTX로 컴파일하고,
CUDA Driver API로 PyTorch의 CUDA 컨텍스트와 현재 스트림에서 실행한다.
PyTorch가 텐서 저장 공간을 관리하며 외부 커널에서 사용한 텐서도 스트림에
등록한다. NVRTC의 런타임 컴파일과 PTX 생성 방식은
[NVIDIA NVRTC 문서](https://docs.nvidia.com/cuda/archive/12.6.3/nvrtc/index.html)에 있다.

패키지를 가져올 때에는 NVRTC를 로드하지 않아 CUDA가 없는 CPU 환경에서도
기존 CPU API를 가져올 수 있다. 처음 GPU 객체나 커널을 사용할 때 CUDA
라이브러리를 로드하고 컴파일한다. 초기 CUDA 준비·NVRTC 컴파일·모델 전송과
준비 탐색은 벤치마크 시간에서 제외한다.

## 기본 사용법

```python
from kingdom_ai import GpuStateBatch, GpuPUCT, GpuPUCTOptions, PolicyValueNet

games = GpuStateBatch.initial(12)
searcher = GpuPUCT(
    PolicyValueNet(),
    GpuPUCTOptions(simulations=32, dirichlet_epsilon=0.0),
)
result = searcher.search(games)
accepted = games.play(result.actions)
```

각 배치 행은 독립된 대국과 탐색 트리다. `search`는 입력 대국을 변경하지
않으며 `play`가 추천 행동을 실제로 적용한다. 초기 모델의 가중치는 무작위이므로
이 예제는 기력 검증이 아니다. 학습된 체크포인트를 사용할 때에도 그 모델의
기력은 별도 평가해야 한다.

`GpuPUCT`는 원본 모델을 복사해 GPU의 FP32 평가 모드로 고정하고
`torch.inference_mode()`에서 추론한다. 원본의 가중치·장치·학습 모드와
gradient를 변경하지 않는다. 새로운 가중치를 사용하려면 탐색기를 다시 만든다.
정책 사전 확률과 역전파 값의 누적은 `float64`를 사용한다.

## 상태와 규칙

`GpuStateBatch.states`는 CUDA `int32` 텐서 `[N, 170]`이다.
각 행에 보드, 영구 집 소유권, 차례, 돌 재고, 패스 횟수와 종료 결과를 보존한다.

| 위치 | 내용 |
| --- | --- |
| `0:81` | 보드: 빈칸 0, 흑 1, 백 2, 중립 돌 3 |
| `81:162` | 집 소유권: 미확정 0, 흑 1, 백 2 |
| `162` | 현재 플레이어 |
| `163`, `164` | 흑·백의 남은 돌 |
| `165` | 연속 패스 횟수 |
| `166` | 종료 사유: 진행 중 0, 포획 1, 자충수 2, 연속 패스 3 |
| `167`, `168` | 승자와 포획한 돌 수 |
| `169` | 예약 필드 |

GPU 종료 사유 번호는 CPU enum의 순서와 다르며 `from_engine`에서 이름을
기준으로 명시적으로 변환한다. `GpuStateBatch.from_engine(states)`는 CPU
상태 목록을 가져오는 검증용 경계이며, `snapshot()`은 GPU 결과를 CPU로
회수하는 디버그 기능이다. GPU 탐색은 CPU 엔진의 규칙 함수를 호출하지 않는다.

기본 확정 규칙만 지원한다. 각자 돌 41개, 자기 집과 상대 집 안 착수 금지,
한 가장자리 집 인정, 상대 포획 우선, 포획 없는 자충수 수락 후 즉시 패배,
연속 패스 시 흑의 집이 최소 3칸 많아야 흑 승리를 적용한다.
분석용 비기본 `GameRules`는 CPU 상태를 가져올 때 거부한다.
중립 돌 위치 변경과 중립 돌 없는 변형은 지원한다.

```python
games = GpuStateBatch.initial(12, neutral=(0, 0))
games_without_neutral = GpuStateBatch.initial(12, neutral=None)
features, legal_masks = games.encode()
copied_games = games.clone()
```

`encode()`는 기존 스키마 1과 같은 CUDA FP32 `[N, 10, 9, 9]` 입력과
CUDA bool `[N, 82]` 합법 수 마스크를 반환한다. 행동 번호는 행 우선으로
`row * 9 + col`이며 패스는 81이다. `play`는 `[N]` 정수 행동을 받고
CUDA bool `[N]` 수락 여부를 반환한다. 거부된 착수는 상태를 보존한다.

## 탐색 옵션과 결과

`GpuPUCTOptions`는 고정 시뮬레이션 수, 탐색 계수, 루트 잡음의 alpha·비율과
시드를 지정한다. 기본값은 수당 32회, `c_puct=1.5`, alpha 0.3, 잡음 비율
0.25와 시드 42다. 시간 제한과 트리 재사용은 지원하지 않으며 호출마다
새 트리를 만든다. 루트 평가는 시뮬레이션 예산에서 제외한다.
포획·자충수·연속 패스로 종료된 잎은 신경망 대신 확정 승패를 역전파한다.

반환값 `GpuSearchResult`의 모든 텐서는 CUDA에 있다.

| 필드 | 의미 |
| --- | --- |
| `actions` | `[N]` 추천 행동; 이미 종료된 대국은 -1 |
| `visits`, `policy` | `[N, 82]` 루트 방문 수와 방문 비율 |
| `priors` | `[N, 82]` 합법 수의 정책 사전 확률 |
| `values`, `best_values` | 루트 현재 플레이어 관점의 평균값과 추천 행동 값 |
| `simulations` | 각 대국에서 완료한 시뮬레이션 수 |
| `network_evaluations` | 각 대국의 실제 비종료 상태 평가 수; 루트 포함 |
| `nodes` | 각 대국의 생성 노드 수; 이미 종료된 대국은 0 |

추천은 방문 수, 행동 값, 사전 확률, 낮은 행동 번호 순으로 결정한다.
종료 상태는 방문·평가·시뮬레이션 수와 정책이 0이고 추천 행동이 없다.
종료 값의 부호는 현재 차례의 플레이어 관점이다.

같은 시드를 줘도 CPU는 대국별 `mt19937_64`, GPU는 배치용 CUDA 난수
스트림을 사용하므로 양수 루트 잡음의 표본은 동일하지 않다.
GPU 배치 크기와 호출 순서도 난수 스트림의 소비 순서를 바꿀 수 있다.
`reset_seed()`는 전용 GPU 스트림을 옵션의 시드부터 다시 시작하며,
전역 PyTorch 난수 상태는 바꾸지 않는다.

## 완료 대국 비교

```powershell
.\.venv\Scripts\python.exe examples\gpu_puct_benchmark.py --games 24 --simulations 32 --workers 12 --gpu-batch-sizes 12 24 --repeats 3
```

CPU 기준은 C++ 신경망 PUCT와 CPU 배치 추론이다. `--workers`는 CPU의
대국 작업자 수이며 `--gpu-batch-sizes`는 GPU에서 함께 탐색할 독립 대국 수다.
`--cpu-threads`는 PyTorch CPU 연산 스레드 수다. 같은 모델 가중치·탐색 예산을
사용하고 준비 시간을 제외한 실제 완료 대국 처리량을 측정한다.
모델 추론은 FP32, 트리 사전 확률·값 누적은 FP64이며 TF32를 끈다.

벤치마크의 기본 루트 잡음은 0으로 CPU/GPU 난수 표본 차이를 제거한다.
신경망 연산의 작은 수치 차이로 경로가 달라질 가능성은 남아 있으므로
완료 판수·평균 수순·평가 수/초, 첫 설정 대비 달라진 대국 수와 결과 체크섬을
함께 출력한다. `--root-noise 0.25`로 잡음을 켜면 난수 생성기 차이도 함께
적용되므로 동일한 대국 경로를 전제하지 않는다.

GPU 측정 시간에는 상태 생성, 탐색, 실제 착수, 종료 상태 확인과 CUDA 동기화,
최종 상태·착수 기록·탐색 및 평가 카운터의 CPU 회수와 결과 집계를 포함한다.
시간 측정이 끝난 뒤 GPU가 기록한 착수를 기존 CPU 엔진에서 다시 실행해
합법 착수·종료 결과·전체 상태 일치를 검증한다. 이 CPU 기준 리플레이 시간은
별도로 출력하며 GPU 처리량 계산에서 제외한다.

GPU 배치를 넓힌 설정도 별도로 측정할 수 있다. 모델과 대국 길이에 따라
효율이 달라지므로 아래 값의 속도 향상을 미리 가정하지 않는다.

```powershell
.\.venv\Scripts\python.exe examples\gpu_puct_benchmark.py --backends cuda --gpu-batch-sizes 64 --games 128 --simulations 32 --repeats 3
```

실측값은 [성능 기록](performance.md)에 기록한다. 기존 순수 MCTS와는
시뮬레이션의 내용·모델 호출 비용이 달라 같은 탐색 횟수로 직접 환산하지 않는다.

## Trainer에서 GPU 자가 대국 사용하기

`Trainer`에 `self_play_backend="cuda"`와 `device="cuda"`를 지정하면 GPU 규칙과
배치 PUCT로 자료를 생성한 뒤 CPU FIFO 버퍼에 저장하고 CUDA에서 미니배치
가중치 갱신을 수행한다. 루트 잡음과 방문 수 착수 샘플링을 적용하며 기본값은
잡음 비율 0.25, alpha 0.3, 온도 1.0이다. 정책 목표는 실제 방문 비율, 가치
목표는 착수 전 플레이어 관점의 최종 승패다. 승격된 기준 모델은 다음 자료
수집 호출부터 새 GPU 탐색기에 반영한다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --device cuda --self-play-backend cuda --self-play-batch-size 128 --iterations 1 --games-per-iteration 128 --simulations 32 --train-steps 8 --batch-size 64 --eval-games 20 --eval-simulations 32 --output runs\gpu-rl
.\.venv\Scripts\python.exe examples\train.py --device cuda --resume runs\gpu-rl\latest.pt --iterations 1 --output runs\gpu-rl
```

자가 대국 배치 크기·탐색·학습 설정은 완료 반복 체크포인트에 저장한다.
재개할 때는 이 설정을 복원하고 완료 반복 경계의 난수 상태로 다음 배치를
생성한다. 진행 중이던 반복은 마지막 체크포인트부터 다시 수행한다.
평가 대국은 기존 C++ 규칙·PUCT를 순차 실행하고 신경망만 CUDA에서 평가한다.
Python의 커널 호출·자료 CPU 회수·일부 동기화도 학습 흐름에 포함된다.

위 명령은 학습 시작 설정 예시다. 잡음 없이 측정했던 GPU 자가 대국
36.79판/초와 실제 학습 처리량은 별도로 측정한다. CLI는 자료 생성 처리량과
자료 생성·학습·평가 단계 시간, 전체 반복 처리량을 출력한다. 저장·재개 범위와
지표 정의는 [강화학습 루프 안내](training_loop.md)에 있다.

## 검증과 구현 위치

```powershell
.\.venv\Scripts\python.exe tests\gpu_rules_test.py
.\.venv\Scripts\python.exe tests\gpu_puct_test.py
.\.venv\Scripts\python.exe tests\gpu_training_test.py
.\.venv\Scripts\python.exe tests\gpu_trainer_test.py
```

규칙 검증은 GPU 상태 전이, 합법 수, 입력 평면과 실제 종료 대국을 CPU 엔진과
비교한다. 탐색 검증은 선택·확장·값의 관점·종료 평가 생략·노드 용량·잘못된
모델 출력과 원본 상태·모델 보존을 확인한다. 두 테스트는
`BUILD_NEURAL_TESTS=ON`인 CTest에 등록하며 CUDA가 없는 환경에서는 생략한다.

2026-10-05 RTX 3050에서 GPU 규칙 테스트 14개와 GPU PUCT 테스트 11개가
통과했다. 기존 엔진·바인딩·학습·평가·재개를 포함한 CTest 15개 묶음도 모두
통과했다. 샌드박스 임시 경로의 접근 권한 문제로 실패한 기존 체크포인트
테스트는 프로젝트의 `build-parallel/test-temp`를 TEMP·TMP·TMPDIR로 지정해
재실행했다. 기존 엔진 파일의 목록과 SHA-256은 작업 전후 동일하다.

- `python/kingdom_ai/gpu_runtime.py`: NVRTC 컴파일과 CUDA Driver API 호출.
- `python/kingdom_ai/gpu_rules.py`: GPU 상태·규칙·합법 수·입력 인코딩.
- `python/kingdom_ai/gpu_puct.py`: GPU 트리와 정책·가치 추론 연결.
- `python/kingdom_ai/gpu_training.py`: GPU 학습 자료 생성과 CPU 버퍼 형식 변환.
- `python/kingdom_ai/loop.py`: GPU 자료 수집 선택과 학습·평가·저장 재개.
- `examples/gpu_puct_benchmark.py`: 완료 대국 측정과 CPU 기준 리플레이 검증.

위 구현은 모두 `engine/` 밖에 추가한다. 기존 엔진은 읽기 전용 검증 기준으로
유지한다.

Trainer 연결 후 GPU 학습 자료 테스트 6개와 연결·재개 테스트 5개를 추가했다.
기존 검사를 포함한 CTest 17개 묶음이 통과했고, 같은 결정적 CUDA 환경에서
연속 학습과 저장 후 재개의 자료·가중치·Adam·버퍼·난수 상태가 일치했다.
