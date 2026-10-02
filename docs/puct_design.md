# 신경망 PUCT 탐색

신경망의 정책·가치를 C++ 트리 탐색에 연결한다. 기존 순수 MCTS는 계속
`engine.MCTS`로 사용하며, 신경망 탐색은 `kingdom_ai.PUCT`로 호출한다.
규칙과 실제 착수 판정은 기존 `engine/`의 `State`를 그대로 사용한다.

트리 선택·확장·역전파는 `mcts/include/PUCT.h`와 `mcts/src/PUCT.cpp`에 있다.
새로운 비종료 상태를 평가할 때 Python 콜백이 PyTorch 모델을 호출한다.
이 단계의 추론은 Python/PyTorch에서 수행하며, C++/LibTorch 직접 추론은
개발 계획의 다음 단계다. 입력 `10×9×9`와 출력 82개 행동·가치 1개인
[스키마 1](neural_network.md)은 바뀌지 않는다.

## 설치와 실행

기존 가상 환경에 PyTorch가 설치되어 있다면 변경한 확장 모듈과 Python
패키지를 다시 설치하고 실행한다. 소스 빌드 환경은
[Python 바인딩 안내](python_bindings.md)를 따른다.

```powershell
.\.venv\Scripts\python.exe -m pip install . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe examples\puct_demo.py --simulations 128
.\.venv\Scripts\python.exe examples\puct_demo.py --self-play --simulations 64
.\.venv\Scripts\python.exe examples\puct_demo.py --checkpoint models\policy_value.pt --simulations 128
```

`--checkpoint`를 생략하면 무작위 초기 모델로 연결을 확인한다. 저장된
체크포인트를 읽을 때도 해당 모델의 기력은 별도 대국 평가로 확인해야 한다.
화면 좌표는 1부터 시작하고 API 좌표는 0부터 시작한다. 추천 실행은 루트
잡음을 끄며, `--self-play`는 루트 잡음 비율 0.25와 온도 1을 사용한다.
대국 자료를 생성하지만 이 예제가 모델의 가중치를 갱신하지는 않는다.

## Python API

```python
import my_board_engine as engine
from kingdom_ai import PUCT, PUCTOptions, PolicyValueNet

model = PolicyValueNet(channels=32, residual_blocks=2)
options = PUCTOptions(simulations=128, c_puct=1.5, seed=42)
searcher = PUCT(model, options=options, device="cpu")
game = engine.State()
result = searcher.search(game)

if result.best_move is not None:
    assert game.play(result.best_move).accepted()
```

`PUCT`는 `PolicyValueNet` 또는 기존 `NeuralAgent`를 받는다. 생략한 옵션은
아래 기본값을 사용한다. 모델을 학습한 뒤 같은 탐색기로 다시 호출하면
현재 가중치를 사용하며, 각 `search`는 새 트리를 만든다. 탐색 결과는 입력
대국을 바꾸지 않는다. 진행 중인 대국의 복사는 `State.copy()`로 수행한다.

| 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `simulations` | 128 | 완료할 트리 시뮬레이션 수; 1 이상 |
| `c_puct` | 1.5 | 정책 사전 확률을 사용하는 탐색 항의 계수; 유한한 양수 |
| `seed` | 42 | 선택적 루트 잡음의 난수 시드 |
| `time_limit_ms` | 0 | 0이면 시간 제한 없음; 지정하면 완료 시뮬레이션 이후 확인 |
| `dirichlet_alpha` | 0.3 | 루트 잡음 분포의 매개변수; 유한한 양수 |
| `dirichlet_epsilon` | 0 | 루트 사전 확률에 섞는 잡음 비율; 0부터 1까지 |

생성 시 옵션을 복사하므로 이후 원래 옵션 객체를 변경해도 기존 탐색기의
설정은 바뀌지 않는다. 시간 제한은 루트 신경망 추론도 포함하며, 진행 중인
상태에서는 적어도 1회 시뮬레이션을 마친다. 진행 중인 추론을 중단하지
않으므로 실제 시간은 지정한 한도를 넘을 수 있다.

## 선택과 가치의 관점

현재 트리 상태 `s`에서 행동 `a`의 다음 점수를 최대화한다.

```text
Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
```

`P`는 합법 수로 정규화한 신경망 정책, `N(s,a)`는 행동 방문 수,
`N(s)`는 해당 상태에서 완료한 시뮬레이션 수다. `Q`는 그 상태에서 수를
둘 플레이어 관점의 평균 평가값이며, 방문하지 않은 행동의 `Q`는 0이다.
첫 시뮬레이션처럼 점수가 같은 경우에는 사전 확률이 높은 행동을 우선하고,
그것도 같으면 행동 번호가 작은 쪽을 선택한다.

비종료 잎에서는 모델의 현재 플레이어 평가값을 사용한다. 종료 잎에서는
신경망을 호출하지 않고 C++ 엔진이 확정한 승패를 `+1` 또는 `-1`로 사용한다.
역전파하면서 차례가 바뀔 때 부호를 반전한다. 따라서 상대 차례에서도 상대가
자기에게 유리한 수를 찾는다. 포획 우선순위, 자충수 즉시 패배와 연속 패스
판정은 엔진의 최종 승자를 따른다.

완성된 자기 집·상대 집과 중립 돌 등 불법 행동은 후보에서 제외한다.
확정 규칙에서 자충수는 합법이며 그 수의 패배가 탐색 결과에 반영된다.
정책은 적어도 한 합법 행동에 양수 가중치를 주어야 하며, 합법 행동에
전부 0을 반환하면 오류다. 후보는 C++ 합법 착수 판정을 기준으로 하며 모델의
높은 확률이 규칙을 우회할 수 없다. 정책과 가치의 형상·범위·유한성을 검사한다.

정책·가치로 트리를 탐색하고 루트 방문 수로 행동 분포를 만드는 접근은
[AlphaZero 논문](https://arxiv.org/abs/1712.01815)을 참고했다.
이 프로젝트의 9×9 규칙, 예산과 구현 선택은 이 문서에 명시한 값을 사용한다.

## 탐색 결과

`engine.PUCTSearchResult`는 다음 정보를 반환한다.

| 필드 | 의미 |
| --- | --- |
| `best_move` | 루트 방문 수가 가장 많은 추천 수; 종료 상태에서는 `None` |
| `simulations` | 완료한 시뮬레이션 수 |
| `nodes` | 실제 생성한 트리 상태 수; 진행 중인 검색은 루트 포함 |
| `network_evaluations` | 비종료 상태의 정책·가치 콜백 호출 수; 루트 포함 |
| `root_value` | 완료한 시뮬레이션의 평균 평가값; 루트 현재 플레이어 관점 |
| `best_value` | 추천 행동의 평균 평가값; 루트 현재 플레이어 관점 |
| `elapsed_seconds` | 해당 검색에서 측정한 경과 시간 |
| `moves` | 루트의 모든 합법 수에 대한 `move`, `prior`, `visits`, `value` |

방문 수가 같으면 행동 평가값, 사전 확률, 행동 번호 순으로 추천 수를
정한다. 방문하지 않은 합법 수도 `visits=0`, `value=0`으로 결과에 들어간다.
평가값은 `-1`부터 `+1` 범위이며, 무작위 완료 대국에서 측정한 순수 MCTS의
`win_rate`와 다른 값이다. 모델이 학습되지 않았다면 이 값을 기력이나
실제 승률로 해석하지 않는다.

루트 평가 뒤 한 번의 시뮬레이션마다 트리 경로를 내려가 종료 상태 또는
새 잎을 평가한다. 루트 평가 자체는 시뮬레이션 예산에서 제외한다.
종료 상태를 재방문하면 신경망을 다시 호출하지 않으므로 콜백 호출 수는
`simulations + 1` 이하이며, 만들어진 상태 수도 같은 상한을 갖는다.
처음부터 종료된 대국은 신경망 호출과 시뮬레이션이 0회이고 추천 수가 없다.
이 경우 `root_value`는 엔진이 확정한 현재 플레이어의 승패 값이다.

## 자가 대국 자료와 학습

```python
from kingdom_ai import collect_puct_game, make_batch, train_step
import torch

data = collect_puct_game(model, seed=42)
batch = make_batch(data.samples, device="cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = train_step(model, optimizer, batch)
```

`collect_puct_game`은 양쪽 모두 같은 모델과 PUCT로 실제 종료까지 대국한다.
`options=None`이면 수마다 128회 탐색, 루트 잡음 비율 0.25를 사용한다.
옵션을 전달하면 그 옵션의 잡음 설정을 따른다. `temperature` 기본값은 1이다.

학습 자료는 기존 `GameData`와 `TrainingSample` 형식을 사용한다. 착수 전의
입력·합법 수 마스크와 **루트 방문 비율**을 저장하고, 대국 종료 후 그 착수
전 플레이어가 승자면 `+1`, 아니면 `-1`을 가치 목표로 정한다.
따라서 `make_batch`와 `train_step`, 체크포인트 저장·복원을 그대로 사용한다.

`sample_visits(result, temperature=1.0, generator=None)`는 방문 수로 다음 수를
뽑는다. 양수 온도 `τ`에서는 `visits^(1/τ)`를 정규화하고, 온도 0에서는
`best_move`를 고른다. 방문하지 않은 행동은 뽑지 않는다. 착수 선택의 온도를
바꾸어도 저장하는 정책 학습 목표는 원래의 방문 비율이다. 종료된 결과에는
샘플링할 수가 없으므로 오류를 반환한다.

이 자료 생성과 한 번의 학습을 바탕으로 이후 대국 축적, 재학습, 이전 모델과의
기력 평가를 반복하는 강화학습 루프를 구성한다. 반복 학습과 대규모 대국 생성이
이번 PUCT 연결만으로 완료되었다고 표시하지 않는다.

## 메모리와 동시 실행

C++ 트리는 노드를 연속 저장하고 자식은 인덱스로 참조한다. 새로운 경로의
상태는 필요할 때만 만들며 모든 합법 행동마다 `State` 사본을 미리 저장하지
않는다. 호출 중 입력 상태는 사본으로 보존하고 콜백에도 별도 사본을 전달한다.

바인딩은 C++ 탐색 동안 GIL을 해제하고 Python 모델 콜백을 실행할 때 다시
획득한다. 같은 탐색기의 검색은 직렬화하며, 같은 스레드에서 직접 다시 검색하는
재귀 호출은 오류로 거부한다. 평가 콜백 안에서 동일 탐색기를 다른 스레드로
호출하고 그 검색의 완료를 기다리면 교착될 수 있으므로, 콜백은 그런 검색을
기다리지 않고 평가값을 반환해야 한다. 다른 탐색기에서 같은 모델을 동시에
추론·학습하는 작업은 호출자가 동기화해야 한다.
현재 구현은 매 잎을 개별 추론하며, 추론 배치 처리·트리 재사용·병렬 탐색은
향후 최적화 대상이다. 처리량과 기력은 실제 측정으로 별도 확인한다.

## 검증 방법

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p puct_test.py
ctest --test-dir build-python -C Release --output-on-failure
```

C++ PUCT 검증은 신경망을 대신하는 고정 평가 콜백으로 선택과 역전파를
확인하고, Python 검증은 실제 PyTorch 모델·엔진·탐색·학습 자료 연결을
확인한다. Python 테스트는 새 모듈을 먼저 빌드·설치해야 한다. 신경망을 포함한
CTest는 기존 `BUILD_NEURAL_TESTS=ON` 빌드 설정을 사용한다.

## 검증 기록: 2026-10-02

- Windows x64에서 MSVC Release 빌드 후 전체 CTest 8개 묶음이 모두 통과했다.
  PUCT 24개 사례와 기존 신경망 24개 사례를 포함한다.
- PUCT 검증은 정책 기반 선택, 상대 관점 역전파, 포획·자충수·패스 종료,
  합법 수 마스크, 루트 잡음, 잘못된 출력, 트리 확장, 동시 호출과 예외 복구를
  확인한다. 큰 유한 탐색 계수에서도 방문 분포가 유지되는 회귀 검증을 포함한다.
- 설치한 wheel에서 `PYTHONPATH` 없이 기본 추천, 체크포인트 추천과 자가 대국
  예제를 실행했다. 수마다 64회 탐색한 시드 42의 무작위 초기 모델 대국은
  11수에 흑의 포획 승리로 끝나고 학습 표본 11개를 생성했다.
- 신경망 자가 대국 자료로 역전파·가중치 갱신과 체크포인트 저장·복원을
  검증했다. 이는 연결의 정확성 검증이며 기력이나 목표 처리량 달성 기록은 아니다.
