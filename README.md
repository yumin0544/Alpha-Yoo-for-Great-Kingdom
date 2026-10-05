# 9x9 Board Game AI Engine

A C++20 engine for the custom 9x9 territory game described in
[the game rules](docs/game_rules.md). The first player uses black/blue stones;
the second uses white/orange stones. The default board has one indestructible
neutral stone at its center.

## Current features

- Fixed 9x9 array board, orthogonal neighbors, and stone groups.
- Validated placement, alternating turns, passes, and 41 stones per player.
- Instant victory when an opposing group has no remaining liberties, with
  opponent capture taking priority over simultaneous self-capture.
- Suicide moves are accepted and immediately lose to the opponent when no
  opposing group is captured.
- Territory detection using player stones, board edges, and the neutral stone;
  completed ownership is retained and neither player can enter a completed house.
- Territory using one, two, or three board edges is accepted; four edges are excluded.
- End after two consecutive passes. Black wins only when its territory exceeds
  White's territory by at least 3 empty points.
- Optional neutral position, including a game without a neutral stone.
- Configurable analysis variants, including a suicide prohibition, documented in
  [the engine design](docs/engine_design.md).
- A playable console game for one user to control both players, an automated
  demo, and standalone tests without external dependencies.
- Pure C++ MCTS with UCT selection, uniform random complete playouts,
  configurable simulation/time budgets, and an AI-versus-AI demo.
- Python access to the same C++ engine and MCTS through `my_board_engine`,
  built with pybind11 and installable with `pip`.
- A PyTorch policy/value model with a documented 10-plane input and 82-action
  output, legal-move masking, MCTS teacher data, one-step training, and checkpoints.
- Neural PUCT with C++ tree search, PyTorch leaf evaluation, optional root noise,
  and visit-based self-play data compatible with the existing training API.
- Repeated self-play learning with bounded FIFO replay, paired-color strength
  evaluation, champion promotion, metrics, and complete training resume.

Bitboard optimization, large-scale learning experiments, and C++/LibTorch
inference remain the next development stages. Training speed and model strength
must be measured with the chosen settings.

## Build and run

Requires CMake 3.20 or newer and a C++20 compiler.

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The playable target is `great_kingdom`, built as `GreatKingdom.exe` on Windows.
For the Windows GCC build, standard runtimes are linked statically, so the
executable can run without separately installing them.

### 직접 대국하기

`build/GreatKingdom.exe`를 더블클릭하면 대국 창이 열린다. MSVC처럼 여러
빌드 구성을 지원하는 환경에서는 `build/Release/GreatKingdom.exe`를 실행한다.
PowerShell에서는 다음과 같이 실행할 수 있다.

```powershell
.\build\GreatKingdom.exe
```

사용자가 선공과 후공을 모두 입력한다. AI는 아직 사용하지 않는다. 좌표는
**1부터 시작하는 행과 열**이며, `3 4`를 입력하면 현재 플레이어가 3행 4열에
돌을 놓는다.

| 입력 | 동작 |
| --- | --- |
| `행 열` (예: `3 4`) | 현재 차례의 돌 놓기 |
| `pass` / `패스` | 차례 넘기기 |
| `help` / `도움말` | 입력 방법 보기 |
| `quit` / `q` / `종료` | 프로그램 종료 |

보드의 `x`는 선공 흑/파랑, `o`는 후공 백/주황, `#`는 중립 돌이다.
`B`와 `W`는 각각 선공과 후공의 완성된 집인 빈칸이며, `.`은 집으로 확정되지
않은 빈칸이다. 착수와 패스가 수락되면 현재 보드, 다음 차례, 남은 돌과 집
점수가 표시된다. 자기 집과 상대 집 모두 착수할 수 없다.

대국이 끝나면 포획, 자충수 또는 연속 패스의 종료 사유와 승자를 표시한다.
`r`을 입력하면 새 대국을 시작하고 `q`를 입력하면 종료한다. 입력이 끝나면
프로그램도 종료한다.

The interactive game source is `examples/play.cpp`. The existing `demo` target
still runs an automated board and pass example: use `build/Release/demo.exe`
for a multi-configuration Windows build, or `build/demo` for a
single-configuration build.

### MCTS로 추천 수 확인하기

순수 MCTS는 기존 엔진의 규칙 판정을 호출한다. `engine/` 파일은 변경하지
않았으며, 탐색 구현은 `mcts/`에 있다. 초기 보드의 선공 추천 수와 실제 측정한
탐색 통계를 보려면 다음을 실행한다.

```powershell
.\build\MCTSDemo.exe
.\build\MCTSDemo.exe 5000
```

기본값은 1000회 시뮬레이션이다. 양쪽 모두 AI로 종료까지 대국하려면
`--self-play`를 사용한다. 이 모드의 기본값은 수마다 64회 시뮬레이션이다.

```powershell
.\build\MCTSDemo.exe --self-play
.\build\MCTSDemo.exe --self-play 200
```

MSVC 등 다중 구성 빌드는 `build/Release/MCTSDemo.exe`에 실행파일을 생성한다.
추천 좌표는 1부터 시작하는 행·열이며 패스는 `pass`다. 출력 승률은 무작위
대국으로 얻은 추정치다. API와 알고리즘은 [MCTS 설계](docs/mcts_design.md)를
참조한다. 기존 `GreatKingdom.exe`는 계속 사용자가 양쪽을 조작하는 대국이다.

## Basic API

Public headers live in `engine/include/board`, in the `kingdom` namespace.
API row and column coordinates are **zero-based**; the rules document uses
one-based coordinates. The default neutral position is therefore `{4, 4}`.

```cpp
#include "board/State.h"

kingdom::State game;
auto outcome = game.play(kingdom::Move::place(0, 0));
if (outcome.accepted()) {
    auto pass_outcome = game.play(kingdom::Move::pass());
}
```

`State::score()` returns current empty territory counts. Check
`State::result().finished()` for game completion; the result contains the winner
and reason after the game ends.
`Board::to_string()` uses `x`, `o`, `v`, and `.` for Black, White, neutral,
and empty cells respectively.

Link the `mcts` target and include `MCTS.h` to request a search without changing
the supplied game state:

```cpp
#include "MCTS.h"

kingdom::State game;
kingdom::MCTSOptions options;
options.simulations = 1000;
kingdom::MCTS searcher(options);
const auto recommendation = searcher.search(game);
if (recommendation.best_move) {
    const auto outcome = game.play(*recommendation.best_move);
}
```

### Python에서 엔진과 MCTS 사용하기

저장소 루트에서 현재 Python에 맞는 확장 모듈을 설치한다. Python과 C++20
컴파일러가 필요하며, Windows에서는 Visual Studio Build Tools의 C++ 도구를
사용할 수 있다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe examples\python_demo.py --simulations 1000
.\.venv\Scripts\python.exe examples\python_demo.py --self-play --simulations 64
```

이미 `.venv`가 있으면 첫 줄은 생략한다. Python API의 좌표는 **0부터 시작**하며
예제의 화면 출력은 1부터 시작한다.

```python
import my_board_engine as engine

game = engine.State()
assert game.place(0, 0).accepted()
searcher = engine.MCTS(engine.MCTSOptions(simulations=1000, seed=42))
recommendation = searcher.search(game)
if recommendation.best_move is not None:
    assert game.play(recommendation.best_move).accepted()
print(game.board.to_string())
```

바인딩은 `engine/`의 기존 규칙 판정을 호출한다. `game.board` 등 상태 조회값은
사본이며, 대국은 `place`, `play`, `pass_turn`으로 진행한다. 종료 여부는
`game.result.finished()`로 확인한다. 설치, 전체 API와 별도 CMake 빌드 방법은
[Python 바인딩 안내](docs/python_bindings.md)에 있다. 기본 C++ 빌드는 Python
패키지 설치 없이 계속 사용할 수 있다.

### 여러 코어로 순수 MCTS 자가 대국 실행하기

`--workers`로 동시에 진행할 대국 수를 지정한다. 각 대국은 별도 상태와
MCTS 객체를 사용하며, C++ 탐색 중에는 Python GIL을 해제한다.

```powershell
.\.venv\Scripts\python.exe examples\python_benchmark.py --games 1000 --simulations 64 --workers 12
# 같은 대국을 작업자 수별로 비교하고, 세 번 측정한 처리량의 중앙값을 출력한다.
.\.venv\Scripts\python.exe examples\python_benchmark.py --games 240 --simulations 64 --workers 1 2 4 6 12 --repeats 3
```

`--games`는 전체 완료 대국 수이며, `--workers` 기본값은 1이다.
각 판의 시드는 작업자 수와 관계없이 `seed + 판 번호`로 정한다. 결과 출력에는
완료 판수, 평균 수순, 내부 탐색 횟수와 결과 검증값이 포함된다.
이 명령은 신경망 없는 순수 MCTS를 실행한다. 실측 조건과 결과는
[처리 속도 기록](docs/performance.md)에 있다.

### PyTorch 모델 연결 확인하기

입력은 현재 플레이어 관점의 `[N, 10, 9, 9]` 텐서다. 모델은 보드 81칸과
패스의 정책 점수 `[N, 82]`, 현재 플레이어의 평가값 `[N]`을 출력한다.
C++의 합법 수를 적용하여 모델의 추천 수를 실제 대국에 전달한다.

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ".[ai]"
.\.venv\Scripts\python.exe examples\neural_demo.py --simulations 64
```

예제는 모델의 합법 수 추천, 순수 MCTS 대국 1판에서 만든 자료로 학습 1회,
모델 저장·복원과 결과 일치를 확인한다. 초기 가중치는 무작위이며, 이 한 번의
학습으로 기력이 검증된 것은 아니다. 기존 C++ MCTS는 순수 무작위 탐색을
유지하며, 신경망 탐색은 별도 PUCT로 사용한다. 입력 평면, 행동 번호와 학습 목표는
[신경망 연결 안내](docs/neural_network.md)에 있다.

### 신경망 PUCT로 추천 수와 자가 대국 만들기

CPU와 CUDA의 신경망 배치 추론을 같은 모델·탐색 예산으로 비교하려면 다음을 실행한다.
CUDA용 PyTorch가 필요하며, 이 명령은 종료까지의 실제 대국 처리량을 측정한다.

```powershell
.\.venv\Scripts\python.exe examples\gpu_benchmark.py --games 24 --simulations 32 --workers 12 --batch-sizes 1 12 --repeats 3
```

설치 환경, 배치 평가 API와 측정 기준은 [GPU 배치 추론 안내](docs/gpu_benchmark.md)에 있다.
체크포인트를 지정하지 않은 모델은 성능 측정용 초기 가중치다.

C++ PUCT가 신경망 정책을 탐색의 사전 확률로 사용하고, 새로운 잎의 평가값은
PyTorch 모델에서 받는다. 종료 상태는 모델 대신 C++ 엔진의 확정 승패를
사용한다. 업데이트한 패키지를 설치한 뒤 다음 예제를 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pip install . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe examples\puct_demo.py --simulations 128
.\.venv\Scripts\python.exe examples\puct_demo.py --self-play --simulations 64
```

기본 실행은 추천 수와 후보별 방문 수·정책 사전 확률·평가값을 표시한다.
`--self-play`는 종료까지 대국하고 학습 자료를 만든다. `--checkpoint`에 기존
모델 파일을 주면 그 가중치를 사용하고, 생략하면 무작위 초기 모델로 연결을
확인한다. 입력 형식은 기존 `10×9×9`, 정책 행동은 패스를 포함한 82개다.

```python
from kingdom_ai import PUCT, PUCTOptions, PolicyValueNet

searcher = PUCT(PolicyValueNet(), PUCTOptions(simulations=128), device="cpu")
recommendation = searcher.search(game)
if recommendation.best_move is not None:
    assert game.play(recommendation.best_move).accepted()
```

평가 검색은 루트 잡음을 기본적으로 끄며, 학습용 자료 생성은 선택적 루트
잡음과 방문 수에 따른 행동 샘플링을 제공한다. 탐색은 입력 대국을 바꾸지
않는다. API, 값의 관점과 학습 자료 계약은 [PUCT 설계](docs/puct_design.md)에 있다.

### 반복 학습과 중단 후 재개하기

현재 기준 모델의 자가 대국을 FIFO 버퍼에 축적하고, 후보를 미니배치로 학습한
뒤 흑백 교대 대국으로 평가한다. 후보 승률이 기준 이상이면 기준 모델을
교체한다. 아래는 작은 탐색 예산으로 실행 연결을 확인하는 예시다.

```powershell
.\.venv\Scripts\python.exe examples\train.py --iterations 1 --games-per-iteration 2 --simulations 4 --eval-games 2 --eval-simulations 4 --train-steps 2 --batch-size 16 --channels 8 --residual-blocks 1 --output runs\smoke
.\.venv\Scripts\python.exe examples\train.py --resume runs\smoke\latest.pt --iterations 1 --output runs\smoke
```

`latest.pt`에는 모델·optimizer·버퍼·설정·진행 횟수·난수 상태를 저장한다.
`best.pt`는 기존 `load_model`로 읽을 수 있는 기준 모델이고, `metrics.jsonl`은
완료 반복별 손실·평가·시간 기록이다. `--iterations`는 이번 실행의 추가 반복
수이며, 재개할 때 학습 설정은 체크포인트에서 복원한다. 실행 중인 반복은
중단 후 처음부터 다시 수행한다. 설정과 API는
[강화학습 루프 안내](docs/training_loop.md)에 있다.

## Roadmap

- [x] Record game rules and development plan
- [x] Basic Engine
- [ ] Bitboard
- [x] Pure MCTS
- [x] pybind11
- [x] PyTorch input/output contract and model connection
- [x] Neural PUCT and self-play data generation
- [x] Repeated learning, replay buffer, strength evaluation, and full resume
- [ ] Large-scale learning experiments and validated playing strength
- [ ] C++ / LibTorch Self Play

## Project references

- [Game rules and original examples (한국어)](docs/game_rules.md)
- [Development plan and implementation choices (한국어)](docs/engine_design.md)
- [Pure MCTS API and implementation (한국어)](docs/mcts_design.md)
- [Python bindings and examples (한국어)](docs/python_bindings.md)
- [PyTorch input/output and model connection (한국어)](docs/neural_network.md)
- [Neural PUCT search and self-play data (한국어)](docs/puct_design.md)
- [Repeated learning, evaluation, and resume (한국어)](docs/training_loop.md)
- [Online game](https://worldsstone.com)
