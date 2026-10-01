# Python 바인딩

`bindings/module.cpp`에서 기존 C++ 엔진과 순수 MCTS를 pybind11로 감싼다.
Python 모듈 이름은 `my_board_engine`이며, Windows에서는 `.pyd`, Linux와
macOS에서는 Python 확장 모듈인 `.so`로 빌드한다. 바인딩은 `engine/` 파일을
수정하지 않고 공개 API를 호출한다. 규칙 판정과 탐색은 계속 C++에서 실행한다.

## 설치와 예제

Python과 C++20 컴파일러가 필요하다. Windows에서는 Visual Studio Build Tools의
C++ 빌드 도구를 사용할 수 있다. 저장소 루트에서 가상 환경을 만들고 설치한다.
`pip`는 `pyproject.toml`에 지정한 빌드 의존성인 scikit-build-core와 pybind11을
준비한 뒤 현재 Python에 맞는 확장 모듈을 빌드한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe examples\python_demo.py --simulations 1000
.\.venv\Scripts\python.exe examples\python_demo.py --self-play --simulations 64
```

이미 `.venv`가 있다면 첫 줄을 생략한다. Linux와 macOS에서는 Python 경로를
`.venv/bin/python`으로 바꾼다. 소스 변경 후에는 `pip install .`을 다시 실행한다.
Python 버전과 플랫폼이 다른 환경에는 해당 환경에서 새로 빌드해야 한다.

`python_demo.py`는 초기 보드에서 추천 수 하나를 찾고 적용한다.
`--self-play`를 지정하면 양쪽 모두 MCTS로 끝까지 대국한다. `--simulations`는
수마다 실행할 탐색 횟수이며 기본값은 1000이다. 출력 승률은 무작위 완료
대국에 근거한 추정치다. PyTorch는 이 예제와 바인딩의 의존성이 아니다.

## 좌표와 기본 호출

Python API는 C++ API처럼 **0부터 8까지**의 행·열 좌표를 사용한다. 규칙 문서의
중앙 (5,5)은 `Position(4, 4)`다. 콘솔 예제의 출력 좌표는 사람이 읽기 쉽도록
1부터 표시한다.

```python
import my_board_engine as engine

game = engine.State()
outcome = game.place(0, 0)
assert outcome.accepted()
assert game.to_play == engine.Cell.White

outcome = game.pass_turn()
assert outcome.accepted()
print(game.board.to_string())

options = engine.MCTSOptions(simulations=1000, seed=42)
recommendation = engine.MCTS(options).search(game)
if recommendation.best_move is not None:
    outcome = game.play(recommendation.best_move)
    assert outcome.accepted()

if game.result.finished():
    print(game.result.winner, game.result.reason)
```

`Move.place(row, col)`은 착수, `Move.pass_turn()`은 패스를 나타낸다.
`outcome.accepted()`가 거짓이면 `outcome.error`에서 거부 사유를 읽는다.
거부된 수는 상태를 바꾸지 않는다. 기본 규칙에서는 자충수를 수락하고 상대
승리로 종료하므로, 수락 여부와 종료 여부를 별도로 확인한다.

## 상태 API

| 호출 또는 속성 | 의미 |
| --- | --- |
| `State()` | 중앙 중립 돌을 둔 기본 대국 시작 |
| `State(neutral=None)` | 중립 돌 없이 시작 |
| `State(neutral=Position(row, col))` | 중립 돌 위치를 지정하여 시작 |
| `state.place(row, col)` | 현재 플레이어가 착수 |
| `state.play(move)` / `state.pass_turn()` | 수 적용 / 패스 |
| `state.is_legal(move)` / `state.legal_moves()` | 합법 여부 / 패스를 포함한 합법 수 목록 |
| `state.to_play` / `state.remaining_stones(color)` | 현재 차례 / 해당 플레이어의 남은 돌 |
| `state.board` / `state.ownership` | 보드 / 81칸의 집 소유권 사본 |
| `state.territory_owner(Position(row, col))` | 해당 칸의 집 소유자 |
| `state.score()` | `black`, `white` 필드가 있는 현재 집 점수 |
| `state.consecutive_passes` | 연속 패스 횟수 |
| `state.result` | 승자, 종료 사유, 종료 점수와 포획 수의 사본 |
| `state.copy()` | 집 소유권, 차례, 재고와 종료 상태를 포함한 완전한 복사 |

`Cell`은 `Empty`, `Black`, `White`, `Neutral`이다. 대국 중 `result.finished()`는
거짓이며 `result.reason`은 `EndReason.None_`이다. 종료 사유는 `Capture`,
`Suicide`, `TwoPasses`다. `TwoPasses`일 때 흑 집이 백 집보다 3칸 이상 많아야
흑 승리이며, 그 외에는 백 승리다.

`state.board`, `state.rules`, `state.ownership`, `state.result`는 사본이다.
사본을 바꿔도 대국 상태는 바뀌지 않는다. 실제 대국은 `play`, `place`,
`pass_turn`으로 진행한다. 따라서 `state.board.place(...)`는 게임 착수에 사용할
수 없다.

별도의 `Board`는 분석용 배치를 만들 수 있다. `Board(cells)`에는 행 우선 순서의
`Cell` 81개를 전달하고, `board.at(Position(...))`로 칸을 읽는다.
`State(board, to_play, rules)`는 그 배치에서 분석을 시작하는 생성자다. 보드만으로
진행 중인 대국의 연속 패스나 과거에 확정된 집 등 모든 이력을 복원할 수
없으므로, 진행 중인 대국을 복제하려면 `state.copy()`를 사용한다.

`GameRules()`의 기본값은 사용자 확정 규칙과 같다. 분석용 옵션은
`suicide_rule=SuicideRule.Loses`, `allow_own_territory_moves=False`,
`allow_single_edge_territory=True`, `stones_per_player=41`이다. 이를 바꾸면
기본 규칙과 다른 변형 대국이 된다.

## MCTS API

```python
options = engine.MCTSOptions(
    simulations=5000,
    exploration=2 ** 0.5,
    seed=42,
    time_limit_ms=1000,
)
searcher = engine.MCTS(options)
result = searcher.search(game)
```

`search()`는 전달한 상태의 사본을 탐색하므로 원래 대국은 바뀌지 않는다.
결과의 `best_move`는 추천 수이며 종료된 대국에서는 `None`이다.
`simulations`, `nodes`, `total_rollout_plies`, `win_rate`, `elapsed_seconds`는
실제 실행의 탐색 통계다. `moves`의 각 항목에는 후보 `move`, `visits`,
`win_rate`가 있다. 후보 목록은 탐색에서 확장한 수만 포함하므로 전체 합법 수
목록과 같다고 가정하지 않는다.
`win_rate`는 탐색을 요청한 상태에서 현재 차례인 플레이어 관점으로, 추천 수의
승률을 추정한 값이다. 후보별 승률도 같은 플레이어 관점을 사용한다.

기본 탐색 횟수는 1000, 탐색 계수는 √2, 시드는 42다. `time_limit_ms=0`이면
시간 제한이 없다. 시간 제한을 지정해도 현재 시뮬레이션을 완료한 뒤 확인하며,
진행 중인 대국에서는 최소 한 번 탐색한다. `simulations`와 시간 제한 중 먼저
충족하는 조건에서 멈춘다. 시드를 고정한 재현 비교는 시간 제한을 끄고 같은
빌드에서 새 `MCTS` 객체를 만들어 수행한다. 한 객체를 반복 사용하면 난수
상태가 이어진다.

C++ 탐색을 수행하는 동안 Python의 GIL을 해제한다. 같은 `MCTS` 객체에 대한
동시 탐색은 차례로 처리한다. 호출 시점에 복사한 게임 상태로 탐색하며,
`searcher.options`도 옵션의 사본이다. Python 스레드를 늘렸다는 이유만으로
탐색 속도가 비례해서 증가한다고 보장하지 않는다.

## CMake로 직접 빌드하기

기본 C++ 빌드는 Python 의존성을 요구하지 않는다. CMake에서 바인딩을 직접
빌드할 때에는 `BUILD_PYTHON_BINDINGS=ON`을 지정하고, 사용할 Python과
pybind11의 CMake 설정 위치를 제공한다.

```powershell
.\.venv\Scripts\python.exe -m pip install pybind11
$pythonPath = (Resolve-Path .venv\Scripts\python.exe).Path
$pybindConfig = & $pythonPath -m pybind11 --cmakedir
cmake -S . -B build-python -G "Visual Studio 17 2022" -A x64 -DBUILD_PYTHON_BINDINGS=ON "-DPython_EXECUTABLE=$pythonPath" "-Dpybind11_DIR=$pybindConfig"
cmake --build build-python --config Release
ctest --test-dir build-python -C Release --output-on-failure
```

위 예시는 Windows의 Visual Studio 2022 빌드 구성을 사용한다. 다른 컴파일러를
사용할 때는 `-G`와 `-A`를 해당 환경에 맞게 변경한다.
일반 사용은 위의 `pip install .`이 설정 경로를 자동으로 전달하므로 간편하다.
직접 빌드한 확장 모듈은 해당 출력 디렉토리를 Python의 모듈 검색 경로에 넣거나
설치한 뒤 가져온다.

빌드 구성은 [pybind11의 빌드 안내](https://pybind11.readthedocs.io/en/stable/compiling.html),
[scikit-build-core 안내](https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html)를
참고한다. 탐색 중 Python 잠금 처리에 관한 배경은
[pybind11의 GIL 안내](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)에 있다.

## 검증 기록: 2026-10-02

- Windows x64, CPython 3.14.8, pybind11 3.1.0, scikit-build-core 1.1.0,
  MSVC 19.41, Ninja, Release 구성에서 `.pyd` 빌드와 가상환경 설치를 확인했다.
- 이 PC의 기본 Visual Studio 구성은 `ucrtd.lib` 검색에 실패했다. 설치된 MSVC와
  Windows SDK 10.0.22621.0의 헤더·라이브러리 경로를 **빌드 프로세스에만** 지정해
  Ninja로 빌드했다. 전역 설정은 바꾸지 않았다. Windows에서 소스를 빌드할 때에는
  C++ 도구와 SDK가 준비된 개발자용 터미널을 사용한다.
- 빌드 의존성을 `.venv`에 준비한 뒤 `pip install . --no-build-isolation`으로
  wheel을 빌드하고 설치했다. 설치된 모듈도 `import my_board_engine`으로 로드했다.
- CTest 6개 묶음이 모두 통과했다. `python_binding_test`에는 규칙, 상태 복사,
  예외 변환, GIL 해제와 같은 탐색 객체의 동시 호출을 확인하는 15개 사례가 있다.
  같은 컴파일러로 빌드한 C++ 참조 프로그램과 Python의 10개 게임 상황 및
  고정 시드 MCTS의 추천 수·후보 통계가 정확히 일치했다.
- 설치된 모듈에서 Python 15개 사례를 다시 확인했고, 예제의 1000회 추천 수와
  수마다 64회 탐색하는 자가 대국이 정상 종료했다. Python을 사용하지 않는
  기존 GCC C++ 구성의 CTest 5개도 통과했다.
- `engine/` 파일 목록과 SHA-256이 작업 전후 동일함을 확인했다. Linux/macOS
  빌드와 PyTorch 학습은 이번 검증 범위에 포함하지 않는다.

설치한 모듈의 규칙과 바인딩을 다시 검사하려면 다음을 실행한다.

```powershell
.\.venv\Scripts\python.exe tests\python_binding_test.py --reference build-python\binding_reference.exe
```

참조 실행파일은 바인딩과 같은 컴파일러로 빌드해야 한다. `--reference`를
생략하면 C++ 결과 비교 한 항목은 건너뛰고 나머지 사례를 검사한다.
