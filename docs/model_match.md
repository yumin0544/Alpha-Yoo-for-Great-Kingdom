# 두 모델 대결과 결과 확인

`examples/model_match.py`는 저장된 정책·가치 모델 두 개를 기존 C++ 규칙 엔진과
PUCT로 대결시키는 콘솔 프로그램이다. 각 판의 흑백 배정, 승자, 종료 사유,
수순 수와 최종 승패·흑백별 승률을 표시한다. `engine/` 수정 없이 기존 평가
경로를 사용한다.

저장소 루트에서 기존 `.venv`의 Python으로 실행한다. `my_board_engine` 바인딩과
PyTorch가 설치되어 있어야 한다. 이 예제는 저장소의 `python/`을 우선 읽으므로
Python 계층만 바뀐 경우 패키지를 다시 설치하지 않아도 된다.

```powershell
.\.venv\Scripts\python.exe examples\model_match.py --model-a runs\model-a\best.pt --model-b runs\model-b\best.pt --games 20 --simulations 128 --output runs\match-a-b.jsonl
```

파일은 `save_model`로 저장한 추론용 스키마 1 체크포인트여야 한다. 학습 실행의
`best.pt`를 그대로 사용할 수 있다. 모델·optimizer·버퍼를 함께 저장하는
`latest.pt`는 지원하지 않으므로 해당 학습 실행에서 추론용 `best.pt`를 내보낸 뒤
지정한다. `--name-a`와 `--name-b`로 화면의 참가자 이름을 정할 수 있다.

모델 파일 없이 실행 연결과 보드 표시를 확인하려면 다음을 사용한다.

```powershell
.\.venv\Scripts\python.exe examples\model_match.py --demo --games 2 --simulations 4 --show-board
```

`--demo`는 고정 시드 11과 29로 작은 초기 모델 두 개를 만들며 가중치를 파일에
저장하지 않는다. 학습되지 않은 모델이므로 결과를 학습 기력의 근거로 삼지 않는다.
`--demo`와 모델 경로 옵션은 함께 사용할 수 없다.

| 옵션 | 기본값과 동작 |
| --- | --- |
| `--model-a`, `--model-b` | 두 추론용 모델 파일; 일반 대결에서는 모두 필요 |
| `--name-a`, `--name-b` | 선택적 참가자 표시 이름 |
| `--games` | 20; 2 이상인 짝수 판수 |
| `--simulations` | 128; 두 모델에 똑같이 적용하는 매 수 탐색 횟수 |
| `--device` | `cpu`; CUDA용 PyTorch와 장치가 있으면 `cuda` 사용 가능 |
| `--seed` | 42; 0 이상 `2**64` 미만 |
| `--tactical-checks` | 기본 꺼짐; 두 모델 모두 즉시 승리와 한 수 안전 검사 적용 |
| `--show-board` | 완료된 판마다 최종 보드 표시 |
| `--watch` | 수마다 착수와 보드 표시 |
| `--output` | 선택적 새 JSONL 파일 경로; 기존 파일 덮어쓰기 거부 |
| `--rating-a`, `--rating-b` | 각 1500; 이번 대결 시작 시 내부 레이팅 |
| `--k` | 32; 완료된 전체 대결을 한 번 갱신하는 계수 |

`--watch`를 사용하면 모든 수를 확인할 수 있다. 표시 좌표는 1부터 시작하며
보드 기호는 `x`(흑/파랑), `o`(백/주황), `#`(중립 돌), `B`와 `W`(각자의
완성된 빈 집), `.`(미확정 빈칸)이다. 화면의 집 점수는 포획·자충수 승리를
뒤집지 않으며 승자는 엔진의 실제 종료 결과를 따른다.

```powershell
.\.venv\Scripts\python.exe examples\model_match.py --model-a runs\model-a\best.pt --model-b runs\model-b\best.pt --games 2 --device cuda --watch
```

두 판마다 한 흑백 쌍을 구성한다. 0부터 세는 쌍 번호 `i`의 시드는
`(seed + i) % 2**64`이고, A 흑/B 백과 B 흑/A 백이 같은 시드를 사용한다.
두 모델 모두 처음 6수는 방문 수를 온도 1로 샘플링하고 이후에는 최다 방문 수를
선택한다. 루트 Dirichlet 잡음과 시간 제한은 끄며 `legacy`, `workers=1` 평가를
사용한다. `--device cuda`는 신경망 추론 장치를 바꾸며 실제 규칙 판정과 PUCT
트리는 기존 CPU C++ 경로에서 진행한다. 기본 중앙 중립 돌과 확정 게임 규칙을
그대로 사용한다.

같은 시드는 서로 다른 모델에 같은 오프닝 수순을 강제하지 않는다. 모델·설정·
환경을 동일하게 유지하면 재검증할 수 있지만, 장치와 PyTorch 버전의 차이에 따른
부동소수점 계산까지 같은 결과를 보장하지 않는다.

`--output`의 JSONL은 한 줄당 하나의 JSON 객체다. `session`은 참가자·모델 경로·
파일 SHA-256·실행 환경·대결 설정, `game`은 완료된 판의 결과와 수순·최종 상태,
`summary`는 전체 완료 결과와 레이팅 갱신을 기록한다. 중단·실패하면 `aborted`를
기록하고 전체 대결 레이팅을 갱신하지 않는다. 이미 완료한 `game` 기록은 남지만
이 파일로 대결을 재개하는 기능은 없다.

수순의 행동 번호 `0..80`은 `행 * 9 + 열`이며 여기서 행·열은 0부터 시작한다.
`81`은 패스다. 따라서 기록의 행동 번호 `0`은 화면의 (1행, 1열)에 해당한다.

레이팅은 `internal_series_elo`로 표시하는 내부 시리즈 평가다. A의 관측 점수
`S_A = A 승리 수 / 전체 판수`와 시작 레이팅으로 다음을 한 번 계산한다.

```text
E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
change = K * (S_A - E_A)
R_A' = R_A + change
R_B' = R_B - change
```

고정 K에서는 2판과 20판 대결의 갱신 가중치가 같다. 이 결과는 이번 상대와
설정에서 얻은 내부 평가이며 절대 기력이나 공인 레이팅을 뜻하지 않는다.
지속적인 레이팅 데이터베이스는 제공하지 않는다. 다음 대결에 이전 값을
사용하려면 `--rating-a`와 `--rating-b`로 직접 지정한다.

이전 구상 중 흑백 균형, 모델 해시와 결과 보존은 적용한다. 첫 프로그램은 이미
있는 모델 평가를 재사용하여 두 참가자의 결과를 보여주는 범위로 구현한다.
범용 `Player` 계층, 라운드 로빈과 레이팅 저장소는 별도 후속 기능이다.

자동 검증은 `tests/model_match_test.py`와 `tests/model_match_cli_test.py`다.
동일 모델의 흑백 균형, 기존 평가와 결과 일치, 수순 재생, 오류 시 모델 상태·
난수 복원, 실제 저장 모델 로드, 콘솔 표시와 결과 파일 보존을 확인한다.
`BUILD_NEURAL_TESTS=ON`인 CMake 구성에서는 두 테스트도 CTest에 등록한다.
