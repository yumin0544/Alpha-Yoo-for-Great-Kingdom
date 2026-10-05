# CUDA 배치 추론과 신경망 자가 대국 측정

`python/kingdom_ai/batching.py`의 `BatchedEvaluator`는 여러 대국의 C++ PUCT
평가 요청을 하나의 신경망 추론 작업자에 모은다. CPU는 규칙 판정과 트리 탐색을
수행하고 CPU 또는 CUDA 장치가 정책·가치를 계산한다. `engine/` 소스는 수정하지 않는다.
기존 순수 MCTS의 무작위 플레이아웃과 신경망 PUCT는 다른 탐색이므로,
두 실행기의 초당 대국 수로 GPU의 향상률을 계산하지 않는다.

## 설치 환경: 2026-10-05

- 실제 장치: NVIDIA GeForce RTX 3050, VRAM 8GB, 드라이버 566.14.
- Windows 11, Python 3.12.14, PyTorch 2.9.1+cu126, CUDA 런타임 12.6.
- `torch.cuda.is_available()`가 참이고 CUDA 텐서의 행렬 곱을 실제 실행했다.
- 설치 버전은 [PyTorch 공식 배포 안내](https://pytorch.org/get-started/previous-versions/#v291)의
  CUDA 12.6 Windows wheel을 사용했다. 별도 시스템 CUDA Toolkit 설치 없이 실행했다.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install "numpy>=2.2,<3"
# 프로젝트 소스가 변경되면 C++ 빌드 환경에서 패키지를 다시 설치한다.
.\.venv\Scripts\python.exe -m pip install . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

현재 PC의 MSYS2 GCC를 사용할 때에는 패키지 설치 명령 전에 빌드 프로세스의
환경을 다음과 같이 설정한다. 전역 PATH는 변경하지 않는다.

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$PWD\.venv\Scripts;$env:PATH"
$env:CMAKE_GENERATOR = "Ninja"
$env:CXX = "C:/msys64/ucrt64/bin/g++.exe"
.\.venv\Scripts\python.exe -m pip install . --no-deps --no-build-isolation
Copy-Item -LiteralPath "C:\msys64\ucrt64\bin\libstdc++-6.dll","C:\msys64\ucrt64\bin\libgcc_s_seh-1.dll","C:\msys64\ucrt64\bin\libwinpthread-1.dll" -Destination .venv\Lib\site-packages
```

## 배치 평가 API

```python
from concurrent.futures import ThreadPoolExecutor
import my_board_engine as engine
from kingdom_ai import BatchedEvaluator, PolicyValueNet

model = PolicyValueNet(channels=32, residual_blocks=2)

def recommend(index, evaluator):
    game = engine.State()
    searcher = engine.PUCT(engine.PUCTOptions(simulations=64, seed=42 + index))
    return searcher.search(game, evaluator)

with BatchedEvaluator(model, device="cuda", max_batch_size=12) as evaluator:
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda index: recommend(index, evaluator), range(12)))
    print(evaluator.stats)
```

각 대국은 독립된 `State`와 `engine.PUCT` 객체를 사용한다. C++ 탐색은 GIL을
해제하고, Python 평가 콜백은 `Future.result()`에서 대기한다. 추론 작업자는
보드를 CPU에서 인코딩하고 입력을 쌓아 장치에 전달한다. 정책·가치 출력은
한 배치로 CPU에 회수하며 기존 합법 수 마스크와 입력 스키마를 유지한다.

추론 서비스는 원본 모델을 복사해 FP32 평가 모드로 고정한다. 원본의 장치,
가중치·gradient와 각 모듈의 학습 모드를 변경하지 않는다. 학습 중 가중치가
변해도 이 서비스의 사본에는 반영되지 않으므로, 새 가중치 사용 시 새 서비스를 만든다.
종료 상태는 신경망을 호출하지 않고 확정 승패를 사용한다.

`max_batch_size`는 실제 배치 크기의 상한이다. 기본 `max_wait_ms=0`은 이미
대기 중인 요청을 모아서 바로 처리한다. 양수로 설정하면 추가 요청을 잠시 기다릴 수
있지만 Windows의 타이머 대기 비용으로 처리량이 낮아질 수 있다. 대기 기한이
끝나면 작은 배치도 처리하므로, 남은 대국 수가 줄어들 때 전체 배치를 기다리며
멈추지 않는다. 종료 시 이미 수락한 요청을 처리하고 신규 요청을 거부하며,
추론 오류가 발생하면 진행 중이거나 대기 중인 모든 요청에 오류를 전달한다.

`stats`는 신경망 평가 수, 추론 배치 수, 평균·최대 실제 배치 크기와
`batch_seconds`를 반환한다. 이 시간은 전송·forward·동기화된 CPU 출력·정책
마스킹을 포함하고 CPU 상태 인코딩·요청 대기를 제외한다. 준비 추론 후 요청이
없을 때 `reset_stats()`로 통계를 초기화할 수 있다.

## 완료 대국 비교하기

```powershell
.\.venv\Scripts\python.exe examples\gpu_benchmark.py --games 24 --simulations 32 --workers 12 --batch-sizes 1 12 --cpu-threads 1 --repeats 3
# CUDA 배치 실행만 측정하기
.\.venv\Scripts\python.exe examples\gpu_benchmark.py --devices cuda --games 24 --simulations 32 --workers 12 --batch-sizes 12
# 저장된 같은 모델로 CPU/GPU를 비교하기
.\.venv\Scripts\python.exe examples\gpu_benchmark.py --checkpoint models\policy_value.pt --games 24 --simulations 32
```

`--games`는 설정마다 실제 종료하는 총 판수다. `--workers`는 동시 대국 수,
`--batch-sizes`는 신경망 평가 요청의 묶음 상한이다. `--cpu-threads`는 PyTorch
CPU 연산 스레드 수이며 여러 값을 주면 각각 비교한다. 대국 수보다 작업자가
많으면 작업자 수를 대국 수로 제한한다.

체크포인트를 지정하지 않으면 시드 42로 기본 모델을 초기화한다. 이 가중치는
성능 측정용이며 학습된 기력의 증거가 아니다. 체크포인트를 지정하는 경우에도
그 모델의 학습 여부와 기력은 별도로 확인해야 한다.

측정은 같은 모델 가중치·판별 시드·탐색 예산·FP32 정밀도를 사용하며 TF32와
혼합 정밀도를 사용하지 않는다. 루트 잡음은 0.25, 실제 착수는 추천 수를 사용한다.
모듈 로드·모델 복사·준비 추론을 제외하고, 스레드 풀 생성·대국 제출·인코딩·
CPU/GPU 왕복·탐색·실제 착수·스레드 풀 종료까지 측정한다. CUDA 작업 완료를
확인한 뒤 시간을 끝낸다. 반복 시작 설정을 순환하며 처리량의 중앙값을 출력한다.

CPU/GPU의 작은 부동소수점 차이가 탐색 경로를 바꿀 수 있어 최종 대국 결과가
다르면 판수와 별도로 차이 수를 출력한다. 동일한 초기 보드를 인코딩한 추론도
따로 비교해, CPU/GPU 출력 오차와 평가 처리량을 확인한다. 이 고정 보드 측정은
인코딩을 제외하고 전송·추론·CPU 결과 회수·마스킹을 포함한다.

`neural_scaling_summary`는 단일 CPU 평가 대비 향상률뿐 아니라 같은 배치 크기의
CPU 대비 향상률, 측정 설정 중 가장 빠른 CPU 대비 향상률도 출력한다.
완료 판수, 평균 수순, 신경망 평가 수/초, 실제 평균 배치 크기를 함께 해석한다.

## CPU 신경망 연산 12스레드 비교

동시 대국 작업자를 12개로 유지하고 PyTorch CPU 신경망 연산 스레드를
1개와 12개로 바꾸어 CPU 및 CUDA 추론의 완료 대국 처리량을 비교한다.

```powershell
.\.venv\Scripts\python.exe examples\gpu_benchmark.py --devices cpu cuda --cpu-threads 1 12 --workers 12 --batch-sizes 12 --games 24 --simulations 32 --repeats 3 --inference-iterations 200
```

`--workers 12`의 대국 작업자와 `--cpu-threads 12`의 PyTorch CPU intra-op
연산 스레드는 별개 설정이다. 함께 지정해도 전체 OS 스레드 수가 정확히 12개가
되거나 CPU 사용률이 항상 100%가 되는 것은 아니다.

## 전체 PUCT의 GPU 실행 여부

현재 CUDA 경로는 CPU의 트리 탐색·규칙 판정과 GPU 신경망 추론을 결합한다.
전체 PUCT를 GPU에서 실행하는 구현은 없어 그 처리량을 아직 측정할 수 없다.
전체 GPU 측정에는 GPU용 규칙 판정·트리 탐색·입력 인코딩의 별도 구현과
기존 CPU 엔진을 기준으로 상태 전이·합법 수·승패가 일치하는지 검증하는 작업이
필요하다. `engine/` 밖에 신규 구현을 추가하면 기존 엔진 파일을 유지할 수 있다.
LibTorch로 추론 콜백을 바꾸는 작업만으로 CPU PUCT가 GPU로 이전되지는 않는다.

## 검증

```powershell
.\.venv\Scripts\python.exe tests\batching_test.py
```

테스트는 단일 평가와 배치 결과 일치, 실제 병렬 배치, 합법 수 마스크,
종료 상태 생략, 원본 모델·상태 보존, 독립 PUCT 포획, 예외 전파와 종료 처리를
확인한다. CUDA가 있으면 GPU 텐서 실행과 CPU/GPU 출력 비교도 실행한다.
Python 바인딩과 `BUILD_NEURAL_TESTS=ON`을 설정한 CTest에 `batching_test`로 등록했다.
2026-10-05에 CUDA 테스트를 포함한 배치 테스트 10개가 통과했고, 기존 엔진·
바인딩·신경망·강화학습을 포함한 CTest 13개 묶음이 모두 통과했다.
새 패키지를 가상환경에 재설치한 뒤 소스 경로 지정 없이 CUDA 대국 실행도 확인했다.
완료 대국과 고정 입력의 실측값은 [성능 기록](performance.md)의 RTX 3050 절에 있다.
기존 `Trainer`의 자료 생성·학습 루프는 이번 실행기와 별도로 계속 순차 실행한다.
