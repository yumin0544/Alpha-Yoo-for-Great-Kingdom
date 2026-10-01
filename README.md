# 9x9 Board Game AI Engine

A C++20 engine for the custom 9x9 territory game described in
[the game rules](docs/game_rules.md). The first player uses black/blue stones;
the second uses white/orange stones. The default board has one indestructible
neutral stone at its center.

## Current features

- Fixed 9x9 array board, orthogonal neighbors, and stone groups.
- Validated placement, alternating turns, passes, and stone inventory.
- Instant victory when an opposing group has no remaining liberties, with
  opponent capture taking priority over simultaneous self-capture.
- Territory detection using player stones, board edges, and the neutral stone;
  completed ownership is retained and opponents cannot enter it.
- End after two consecutive passes. Black wins only when its territory exceeds
  White's territory by at least 3 empty points.
- Optional neutral position, including a game without a neutral stone.
- Configurable handling of unresolved rule details, documented in
  [the engine design](docs/engine_design.md).
- A command-line demo and standalone tests without external dependencies.

MCTS, Bitboard optimization, Python bindings, and neural network training are
the next development stages.

## Build and run

Requires CMake 3.20 or newer and a C++20 compiler.

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run `build/Release/demo.exe` for a multi-configuration Windows build, or
`build/demo` for a single-configuration build.

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

## Roadmap

- [x] Record game rules and development plan
- [x] Basic Engine
- [ ] Bitboard
- [ ] Pure MCTS
- [ ] pybind11
- [ ] PyTorch Neural Network and Self Play
- [ ] C++ / LibTorch Self Play

## Project references

- [Game rules and original examples (한국어)](docs/game_rules.md)
- [Development plan and implementation choices (한국어)](docs/engine_design.md)
- [Online game](https://worldsstone.com)
