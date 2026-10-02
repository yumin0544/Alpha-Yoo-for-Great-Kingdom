# PyTorch 신경망 입력·출력과 엔진 연결

`python/kingdom_ai/`에서 C++ 대국 상태를 신경망 입력으로 변환하고,
PyTorch 모델의 출력을 합법 수로 제한하여 실제 엔진에 전달한다.
기존 `engine/`의 규칙 판정은 그대로 사용한다.

이 단계에서는 입력·출력 계약, 모델 추론, 순수 MCTS 대국의 학습 자료 변환,
한 번의 학습과 모델 저장·복원을 연결한다. 기존 C++ MCTS는 무작위 완료
대국을 사용하는 순수 MCTS로 유지한다. 신경망을 이용하는 PUCT 탐색과
대규모 강화학습은 후속 단계다.

## 설치와 실행

Python 바인딩과 `kingdom_ai`는 같은 프로젝트 패키지에 포함된다.
PyTorch는 선택 의존성이므로 기존 C++ 엔진과 Python MCTS만 사용하면
설치할 필요가 없다. 아래 명령은 저장소 루트의 기존 가상 환경을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ".[ai]"
.\.venv\Scripts\python.exe examples\neural_demo.py --simulations 64
```

소스 패키지를 설치할 때에는 C++20 컴파일러가 필요하다. Windows에서
Visual Studio 빌드 환경을 사용하는 방법은 [바인딩 설치 안내](python_bindings.md)를
참고한다. GPU를 사용하려면 해당 환경에 맞는 PyTorch를 별도로 설치한다.
Linux와 macOS에서는 Python 경로를 `.venv/bin/python`으로 바꾼다.

일반 패키지 저장소의 PyTorch를 사용하려면 `pip install ".[ai]"`로 선택
의존성을 함께 설치할 수도 있다. 위의 CPU 전용 설치 예제는 GPU 없이 연결을
확인하기 위한 구성이다.

`neural_demo.py`는 다음을 실행한다.

1. 초기 모델로 추천 수를 만들고 C++ 상태 사본에 실제로 적용한다.
2. 순수 MCTS로 종료까지 대국 1판을 생성한다.
3. 방문 수와 최종 승자를 학습 목표로 삼아 모델을 한 번 갱신한다.
4. `models/policy_value.pt`에 저장하고 다시 읽어 추론 결과의 일치를 확인한다.

```powershell
.\.venv\Scripts\python.exe examples\neural_demo.py --simulations 128 --seed 42 --checkpoint models\demo.pt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p neural_test.py
```

예제의 초기 가중치는 무작위다. 한 판, 한 번의 갱신은 데이터와 연산의 연결을
확인하기 위한 것으로, 모델의 대국 실력을 검증하지 않는다. 실행 예제에서만
CPU 추론 스레드를 1개로 설정하며, 라이브러리를 가져올 때 전역 설정을 바꾸지 않는다.

## 입력 계약: 스키마 1

입력은 `float32` 텐서이며 형상은 **`[N, 10, 9, 9]`**다. `N`은 한 번에 처리하는
상태 수다. 각 상태는 항상 **현재 차례인 플레이어** 관점으로 표현한다.
돌과 집을 나타내는 평면은 해당 칸이 조건에 맞으면 1, 그 외에는 0이다.

| 평면 | 내용 | 값 |
| ---: | --- | --- |
| 0 | 내 돌 | 0 또는 1 |
| 1 | 상대 돌 | 0 또는 1 |
| 2 | 중립 돌 | 0 또는 1 |
| 3 | 이미 완성된 내 집 | 0 또는 1 |
| 4 | 이미 완성된 상대 집 | 0 또는 1 |
| 5 | 현재 차례가 선공 흑인가 | 모든 칸에 0 또는 1 |
| 6 | 내 남은 돌 | 모든 칸에 `남은 돌 / 41` |
| 7 | 상대의 남은 돌 | 모든 칸에 `남은 돌 / 41` |
| 8 | 연속 패스 횟수 | 모든 칸에 `횟수 / 2` |
| 9 | 현재 합법인 보드 착수 지점 | 0 또는 1 |

집은 보드에서 새로 추정하지 않고 **C++ 상태에 보존된 소유권**을 사용한다.
자기 집과 상대 집 모두 착수가 금지되므로 빈칸과 집을 구분해야 한다.
선공 여부 평면은 선공만 3칸 차이 이상을 확보해야 이기는 점수 규칙을 표현한다.
남은 돌과 패스 횟수도 보드 배치만으로는 알 수 없는 정보다.

별도로 **`[N, 82]`의 `bool` 합법 수 마스크**를 만든다. 기준은 C++의
`state.legal_moves()`다. 이미 종료된 상태는 모든 행동이 거짓이다.
진행 중인 상태에서는 패스가 합법이며, 자충수도 확정 규칙에 따라 수락되므로
마스크에서 제거하지 않는다. 자충수의 즉시 패배를 평가하는 것은 모델과 학습의 역할이다.

스키마 1은 확정 기본 규칙인 자충수 허용 후 패배, 자기 집 착수 금지,
한 가장자리 집 허용, 각자 돌 41개를 사용한다. 중립 돌 위치 변경과 중립 돌 없는
변형은 입력에 직접 표현할 수 있다. `GameRules`의 분석용 옵션을 바꾼 상태는
규칙 정보가 없는 같은 입력으로 혼동되지 않도록 거부한다.

## 출력 계약

모델은 `(policy_logits, value)`를 반환한다.

| 출력 | 형상 | 의미 |
| --- | --- | --- |
| `policy_logits` | `[N, 82]` | 보드 81곳과 패스에 대한 정규화 전 점수 |
| `value` | `[N]` | 현재 플레이어 관점의 평가값, `-1`부터 `+1`까지 |

보드 행동은 행 우선 순서로 **`action = row * 9 + col`**이며, 행·열 좌표는
0부터 센다. `0`은 (0,0), `40`은 (4,4), `80`은 (8,8), **`81`은 패스**다.
중앙 중립 돌이 있는 기본 시작 상태에서는 행동 40이 불법이다.

`NeuralAgent.predict(state)`가 마스크를 적용하고 합법 행동에 대해서만
확률을 정규화한다. 반환 `Prediction`의 `policy`는 CPU `float32` 텐서 82개,
`value`는 Python 실수, `best_action`은 가장 확률이 높은 행동 번호,
`best_move`는 C++에 전달할 수다. 종료된 대국에는 추천 수가 없고
`best_action`과 `best_move`가 `None`이다.

평가값 `+1`은 현재 플레이어 승리, `-1`은 현재 플레이어 패배를 의미한다.
이는 학습 목표의 해석이며, 초기 모델의 평가값을 실제 승률로 보지 않는다.
`policy` 역시 신경망의 행동 분포이며 MCTS의 방문 횟수나 측정된 승률이 아니다.

## 모델 구조와 기본 호출

`PolicyValueNet(channels=32, residual_blocks=2)`는 작은 합성곱 잔차 모델이다.
입력 합성곱과 GroupNorm, 잔차 블록 뒤에 정책과 가치 출력을 나누어 둔다.
GroupNorm을 사용하므로 단일 상태 추론과 작은 학습 배치도 사용할 수 있다.
정책 출력은 82개 점수, 가치 출력은 마지막 `tanh`로 범위를 제한한다.
합법 수 정규화는 모델의 `forward` 외부에서 수행한다.

```python
import my_board_engine as engine
from kingdom_ai import NeuralAgent, PolicyValueNet

game = engine.State()
model = PolicyValueNet(channels=32, residual_blocks=2)
agent = NeuralAgent(model, device="cpu")
prediction = agent.predict(game)

if prediction.best_move is not None:
    outcome = game.play(prediction.best_move)
    assert outcome.accepted()
```

신경망이 수를 선택해도 실제 착수, 포획, 자충수와 종료 판정은 C++ 엔진이
수행한다. `predict()`는 입력 대국에 착수하지 않으며, 호출자가 수를 적용한다.

## 학습 자료와 한 번의 갱신

`collect_mcts_game(simulations=64, seed=42)`는 양쪽 모두 기존 순수 MCTS로
대국하고 매 수 **착수 전** 상태를 저장한다. 반환 `GameData`에는 `samples`,
최종 `winner`, 종료 `reason`이 있다. 각 `TrainingSample`에는 입력 특징,
합법 마스크, 방문 수에서 만든 정책 목표와 가치 목표가 들어간다.

정책 목표는 루트 후보 방문 수의 합으로 정규화한 82개 확률이다. 탐색에서
확장하지 않은 행동은 0이 된다. 적은 탐색 횟수는 합법 수 전체를 방문하지
못할 수 있으므로, 예제의 작은 탐색 예산을 학습 품질의 기준으로 삼지 않는다.

가치 목표는 **저장한 착수 전 플레이어**가 최종 승자이면 `+1`, 아니면 `-1`이다.
마지막 착수가 자충수일 때도 최종 승자를 기준으로 같은 식을 사용한다.
종료 시의 `state.to_play`만 보고 모든 표본의 부호를 정하면 안 된다.

```python
import torch
from kingdom_ai import collect_mcts_game, make_batch, train_step

data = collect_mcts_game(simulations=64, seed=42)
batch = make_batch(data.samples, device="cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = train_step(model, optimizer, batch)
print(losses)
```

학습 손실은 합법 수에 대한 정책 교차 엔트로피와 가치 평균 제곱 오차의 합이다.
`train_step`은 역전파와 가중치 갱신을 수행하고 `loss`, `policy_loss`,
`value_loss`를 반환한다. 정책 목표와 추론 모두 같은 행동 번호와 합법 마스크를
사용한다. 현재의 자료 생성은 순수 MCTS를 교사로 사용하는 첫 연결 단계다.

## 모델 저장과 복원

```python
from kingdom_ai import save_model, load_model

save_model(model, "models/policy_value.pt")
restored = load_model("models/policy_value.pt", device="cpu")
```

체크포인트에 입력·출력 스키마, 모델 구성과 `state_dict`를 함께 저장한다.
복원 시 스키마와 가중치 구조를 검사하며 `weights_only=True`로 읽는다.
입력 평면이나 행동 정의를 바꿀 때에는 스키마도 바꾸어 다른 형식의 모델을
잘못 연결하지 않도록 한다. 이 파일은 PyTorch에서 저장·복원하는 모델이며,
LibTorch용 내보내기 형식은 다음 단계에서 결정한다. 스키마 1 체크포인트의
가중치 자료형은 `float32`다. 옵티마이저 상태와 전체 학습 진행 이력은 이번
모델 파일에 저장하지 않는다.

## 검증 기록: 2026-10-02

- Windows x64, CPython 3.14.8, CPU용 PyTorch 2.14.1, NumPy 2.5.3에서 확인했다.
- 기본 모델은 32채널, 잔차 블록 2개, 학습 가능한 매개변수 56,189개다.
- `neural_test.py` 24개 사례가 통과했다. 색과 차례 관점, 집·중립 돌·재고·패스,
  자충수와 종료, 합법 수 분포, 실제 MCTS 방문 수와 승패 목표, 역전파와
  가중치 갱신, 저장·복원 및 스키마 불일치를 검증한다.
- 설치한 wheel의 `kingdom_ai`에서도 테스트와 예제 실행을 확인했다.
- 예제의 시드 42, 수마다 64회 탐색한 교사 대국은 4수에 백의 포획 승리로
  종료했다. 그 4개 상태로 학습 한 단계를 수행하고 저장·복원 후 정책·가치가
  정확히 일치함을 확인했다. 이 짧은 대국은 실행 연결 검증용이다.
- `BUILD_NEURAL_TESTS=ON`으로 기존 검증을 포함한 CTest 7개 묶음이 모두 통과했다.
  이 옵션은 `BUILD_PYTHON_BINDINGS=ON`과 PyTorch·NumPy가 준비된 Python을 요구한다.
  기본값은 꺼져 있어 일반 C++ 빌드가 신경망 의존성을 요구하지 않는다.

```powershell
cmake -S . -B build-python -DBUILD_NEURAL_TESTS=ON
ctest --test-dir build-python -C Release --output-on-failure
```

위 명령은 바인딩용 `build-python`을 이미 구성한 상태에서 사용한다.
처음 구성한다면 [Python 바인딩 안내](python_bindings.md)의 설정도 전달한다.
`engine/` 파일 목록과 SHA-256이 작업 전후 동일함을 확인했다.

PyTorch 동작과 저장 방식의 참고 문서는
[신경망 모듈](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html),
[모델 저장·복원](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)이다.

## 이후 작업

이 연결을 바탕으로 신경망 정책을 탐색의 사전 확률에 사용하고, 가치 평가로
탐색 잎을 평가하는 PUCT를 구현한다. 이후 자가 대국 자료 축적, 모델 갱신,
기존 모델과의 대국 평가, 실제 처리량 측정을 묶어 강화학습 루프를 만든다.
성능은 실제 측정 조건과 함께 기록하며, 이 모델 연결만으로 처리량이나
기력을 달성한 것으로 보고하지 않는다.
