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

Bitboard optimization, Python bindings, and neural network training are
the next development stages.

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

## Roadmap

- [x] Record game rules and development plan
- [x] Basic Engine
- [ ] Bitboard
- [x] Pure MCTS
- [ ] pybind11
- [ ] PyTorch Neural Network and Self Play
- [ ] C++ / LibTorch Self Play

## Project references

- [Game rules and original examples (한국어)](docs/game_rules.md)
- [Development plan and implementation choices (한국어)](docs/engine_design.md)
- [Pure MCTS API and implementation (한국어)](docs/mcts_design.md)
- [Online game](https://worldsstone.com)
