# 반복 강화학습과 학습 재개

`python/kingdom_ai/loop.py`는 신경망 PUCT 자가 대국, 제한된 자료 버퍼,
미니배치 학습, 기존 모델과의 대국 평가를 반복한다. 완료한 반복의 전체 상태를
저장하므로 프로그램을 중단한 뒤 이어서 실행할 수 있다. 게임의 규칙과 실제
착수는 기존 C++ 엔진을 사용하며 `engine/`는 변경하지 않는다.

현재 구현은 대국과 모델 추론을 순차적으로 실행한다. 여러 대국의 병렬 생성,
추론 배치, Bitboard 최적화와 C++/LibTorch 직접 추론은 이후 작업이다.
반복 학습 프로그램의 구현과 수십만 판 학습으로 얻은 기력 검증은 구분한다.

## 실행 순서

한 번의 반복은 다음 단계로 구성된다.

1. 현재 기준 모델인 `champion`으로 지정한 수의 PUCT 자가 대국을 끝까지 진행한다.
2. 착수 전 상태, 탐색 방문 비율, 최종 승패 목표를 FIFO 자료 버퍼에 넣는다.
3. 버퍼에서 위치를 균등하게 복원추출하여 `learner`를 지정한 횟수만큼 갱신한다.
4. 갱신한 후보와 `champion`이 같은 탐색 예산으로 흑백을 교대해 대국한다.
5. 후보 승률이 승격 기준 이상이면 후보를 새로운 `champion`으로 채택한다.
6. 완료한 반복의 모델, 버퍼, optimizer, 난수와 진행 횟수를 저장하고 지표를 기록한다.

승격되지 않아도 `learner`와 Adam 상태는 다음 반복으로 이어진다.
자가 대국에 사용하는 모델은 평가를 통과한 `champion`이다. 따라서 후보를
거부할 때마다 학습된 가중치와 optimizer를 초기화하지 않는다.

자가 대국은 매 판 다른 시드를 사용하며 기본적으로 루트 Dirichlet 잡음과
방문 수에 따른 착수 샘플링을 사용한다. 정책 목표는 항상 실제 방문 비율이고,
착수 샘플링 온도는 게임에서 선택하는 수에만 적용된다. 가치 목표는 저장한
착수 전 플레이어가 최종 승자이면 `+1`, 아니면 `-1`이다.

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
`--output`, `--device`, `--threads`만 변경할 수 있으며, 탐색 횟수·학습률·버퍼
용량 등 학습 설정을 함께 지정하면 오류를 표시한다. `--initial-model`은
기존 `save_model` 파일의 가중치로 **새 학습 실행**을 시작하는 옵션이다.
`--resume`과 함께 사용할 수 없고 optimizer, 버퍼와 진행 횟수는 이어받지 않는다.

## 설정

| CLI 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `--iterations` | 1 | 이번 실행의 추가 반복 수 |
| `--output` | `runs/rl` | 저장 파일과 지표를 기록할 디렉토리 |
| `--games-per-iteration` | 4 | 한 반복의 완료 자가 대국 수 |
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
| `--eval-opening-moves` / `--eval-opening-temperature` | 6 / 1.0 | 평가 초반 방문 수 착수 샘플링 |
| `--promotion-threshold` | 0.55 | 후보 승률이 이 값 이상이면 승격 |
| `--temperature` | 1.0 | 자가 대국의 방문 수 착수 샘플링 온도 |
| `--seed` | 42 | 모델 초기화와 자료 생성·표본 추출의 시드 |
| `--channels` / `--residual-blocks` | 32 / 2 | 새 모델의 구성 |
| `--device` | `cpu` | PyTorch 추론과 학습 장치 |
| `--threads` | 1 | CLI 실행의 PyTorch CPU 스레드 수 |

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

## 저장 파일과 재개 범위

| 파일 | 내용 | 읽는 API |
| --- | --- | --- |
| `latest.pt` | learner와 champion, Adam, 버퍼, 설정, 진행 횟수, 난수 상태 | `Trainer.load_checkpoint` / `--resume` |
| `best.pt` | 현재 champion의 스키마·모델 구성·가중치 | 기존 `load_model` |
| `metrics.jsonl` | 완료한 반복마다 한 줄의 JSON 지표 | 일반 JSON Lines 도구 |

`latest.pt`는 모델 가중치만 저장하던 파일과 형식이 다르다. 두 종류는 서로의
복원 API로 읽지 않는다. 전체 체크포인트는 `weights_only=True`로 읽고 입력
스키마, 가중치·Adam·버퍼·진행 횟수·난수 상태의 일관성을 검증한다.

체크포인트는 임시 파일을 쓴 뒤 교체하며, 반복 시작 전과 완료 후에 저장한다.
`Ctrl+C`로 중단하거나 반복 도중 오류가 발생하면 **마지막으로 완료한 반복**의
체크포인트에서 재개한다. 진행 중이던 대국·학습·평가는 그 반복을 처음부터
다시 실행한다. Python API에서도 불완전한 반복을 그대로 저장하거나 계속
실행하지 못하게 한다.

전용 CPU Generator가 판마다 사용할 시드와 버퍼 표본 추출을 관리한다.
전체 체크포인트에는 이 Generator와 Python·PyTorch CPU·CUDA 난수 상태를
저장한다. 장치·PyTorch 버전 등을 바꾸면 부동소수점 연산 결과가 달라질 수 있다.

## 평가와 기록 해석

평가에서는 후보가 흑인 판과 백인 판을 짝으로 진행하며 두 판에 같은 시드를
사용한다. 루트 Dirichlet 잡음과 시간 제한은 사용하지 않는다. 기본적으로
처음 6수는 온도 1로 방문 수에서 착수를 뽑고, 이후에는 가장 많이 방문한 수를
선택한다. 동일 모델끼리의 같은 시드 흑백 쌍은 정확히 1승 1패가 된다.

기본 20판이면 후보가 11승 이상 했을 때 55% 기준을 통과한다. 이 값은 현재
기준 모델과의 관측 승률이며, 절대 기력이나 모든 상대에 대한 승률은 아니다.
적은 평가 판수는 편차가 크므로 장기 학습의 기준은 추가 검증으로 정한다.

지표에는 완료 반복 수, 누적 자가 대국·가중치 갱신 수, 기준 모델 버전,
이번에 만든 위치 수, 버퍼 크기, 평균 수순, 종료 사유, 세 손실, 흑백별 후보
승수와 승격 여부, 자료 생성·학습·평가의 경과 시간이 들어간다.
손실 감소와 후보 승격을 함께 살피고, 평가 예산과 상대가 바뀌었는지도 확인한다.

## Python API

```python
from kingdom_ai import Trainer, TrainingConfig, load_model

config = TrainingConfig(
    games_per_iteration=2, simulations=4, evaluation_games=2,
    evaluation_simulations=4, train_steps_per_iteration=2, batch_size=16,
)
trainer = Trainer(config, device="cpu")
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
후보 관점의 승패와 흑백별 승수를 반환한다.

## 검증

버퍼의 FIFO 교체·고정 메모리·사본 분리·시드 표본 추출·저장 복원,
흑백 교대 평가·실제 종료·모델 보존, 완료 반복·승격·거부·중단 재개를 검사한다.
검증 명령은 다음과 같다.

```powershell
.\.venv\Scripts\python.exe tests\replay_test.py
.\.venv\Scripts\python.exe tests\evaluation_test.py
.\.venv\Scripts\python.exe tests\reinforcement_test.py
ctest --test-dir build-python -C Release --output-on-failure
```

CTest는 `BUILD_PYTHON_BINDINGS=ON`, `BUILD_NEURAL_TESTS=ON`과 PyTorch가 설치된
Python으로 구성한 빌드를 사용한다. 테스트와 짧은 실행은 프로그램의 동작을
검증하며, 대규모 학습의 완료나 기력 달성을 의미하지 않는다.

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
