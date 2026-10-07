# 학습 실행 파일과 로그 정리

## 2026-10-08 정리 결과

실제 디렉터리 이름은 `runs`다. `past-version`의 16개 파일은 삭제·수정하지
않았으며 정리 전후 내용 SHA-256과 수정 시각이 모두 같음을 확인했다.
`engine/`도 변경하지 않았다.

- 전체 파일: 139개 → 51개.
- 전체 용량: 8,784,675,214바이트 → 3,032,806,301바이트, 약 5.36GiB 회수.
- 이전 실험 21개 폴더와 종료된 대국의 보조 파일 30개를 활성 경로에서 삭제했다.
- 원본 91개 파일을 먼저 ZIP에 압축하고 각 항목을 다시 읽어 SHA-256을 검증했다.
- 원본 복구 ZIP은 `runs/archive/cleanup-2026-10-08.zip`이며 약 131MiB다.
- 파일별 출처·크기·해시·보호 경로 검증 기록은 같은 이름의 `.zip.json` 파일에 있다.

현재 남긴 경로는 다음과 같다.

| 경로 | 용도 |
| --- | --- |
| `runs/past-version/` | 사용자가 보존한 이전 챔피언·로그·대국 요약, 변경 금지 |
| `runs/tactics-depth20-2026-10-08-0138i/` | 최신 학습: 완료 575사이클·champion 178 |
| `runs/tactics-depth20-2026-10-08/` | 깊이 20 변경 시 보존한 565사이클 출발점 |
| `runs/matches/` | 기존 대국 원시 수순·승패·요약과 실행 요청 |
| `runs/archive/` | 이전 실험과 삭제한 보조 로그의 압축 원본·검증 보고서 |

두 활성 폴더의 `latest.pt`는 정리 전후 전체 파일 SHA-256까지 동일하다.
대국의 `results.jsonl`과 `request.json`도 보존했다. 완료·중단된 대국의
`worker.log`, `progress.json`, `stop.request`만 정리하며 실패의 원인을 담는
worker 로그는 남기는 정책이다.

## 최신 학습 재개

```powershell
.\.venv\Scripts\python.exe examples\train.py `
  --resume runs\tactics-depth20-2026-10-08-0138i\latest.pt `
  --device cuda --iterations 1 --output runs\tactics-depth20-2026-10-08-0138i
```

과거 문서에 나온 `gpu-*`, `model-*`, `plateau-*`, `tactics-online-*` 등의
실제 실행 경로는 이번 압축 보관 대상이다. 과거 파일을 사용하는 명령을 실행하려면
아래처럼 먼저 복원하고 복원된 경로를 지정한다. 기록상의 학습 횟수·설정은
역사적 사실이므로 문서의 과거 결과를 새 실행 결과로 바꾸지는 않는다.

## 원본 복원

현재 파일을 덮어쓰지 않도록 **새 폴더**에 압축을 푼다.

```powershell
Expand-Archive -LiteralPath .\runs\archive\cleanup-2026-10-08.zip `
  -DestinationPath .\runs\recovered-2026-10-08
```

예를 들어 E 원본은 `runs/recovered-2026-10-08/plateau-E-05pct-256-learner/latest.pt`다.
정리 전 최신 로그의 상세 전술 기록도 ZIP의
`tactics-depth20-2026-10-08-0138i/metrics.jsonl`에 있다. 복구 ZIP을 삭제하면
이번에 정리한 파일의 복구 수단을 잃으므로 별도 보관 없이 지우지 않는다.

## 앞으로의 로그

`Trainer.run()`이 JSONL을 기록할 때 긴 전술 `records`의 보드·수순을 매 사이클
반복 저장하지 않고 `online_tactics.record_summary`로 바꾼다. 결과별 탐색 수,
채택 WIN/LOSS 수, 증명 깊이 분포, 예산 소진 수, 최소 깊이 등 제외 사유,
채택 motif·선후공 분포를 유지한다. 손실·일반/전술 표본 추출·평가·성능·actor
해시와 탐색 예산은 그대로 남긴다.

최근 로그는 338,443 → 58,612바이트로 약 82.7% 줄었다. 10사이클의 성능
요약이 원본과 같음을 검증했다. 체크포인트의 증명 replay·증명 메타데이터와
마지막 반복 상세 기록, API 반환·callback의 상세 기록은 변경하지 않는다.
`train.py`는 오래된 설치본 대신 현재 저장소의 Python 코드를 읽으므로,
기존 네이티브 확장 모듈을 재빌드하지 않아도 축약 로그가 적용된다.

## 정리 도구의 안전 조건

`examples/cleanup_runs.ps1`은 기본적으로 계획만 출력한다. 삭제하려는 실험
폴더를 `-ArchiveRun`으로 **명시**하고 `-Apply`를 지정해야 압축·검증 후 정리한다.
상위 경로·저장소 외부·심볼릭 링크와 junction·모든 `past-version` 경로를 거부한다.
원본이 백업 중 변경되거나 항목 해시가 맞지 않으면 삭제하지 않는다.
학습·대국 쓰기가 끝난 뒤 사용한다. 기존 복구 ZIP은 덮어쓰지 않는다.

관련 회귀 검증 56개와 실제 로그 분석 도구의 축약 로그 읽기가 통과했다.
