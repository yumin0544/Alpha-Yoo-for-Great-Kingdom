"""Launch the local Great Kingdom board and saved-model controls."""

import argparse
from pathlib import Path
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from kingdom_ai.play_ui import PlaySession, PlaySettings, make_server


def main():
    parser = argparse.ArgumentParser(description="클릭으로 두는 Great Kingdom")
    parser.add_argument("--port", type=int, default=8765, help="로컬 포트; 0은 빈 포트 자동 선택")
    parser.add_argument("--open", action="store_true", help="브라우저에서 바로 열기")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("포트는 0~65535 사이여야 합니다.")
    import torch
    torch.set_num_threads(1)
    models = {path.relative_to(ROOT).as_posix(): path
              for path in (ROOT / "runs").rglob("best.pt") if path.is_file()}
    session = PlaySession(models)
    try:
        server = make_server(session, ROOT / "python" / "kingdom_ai" / "web", args.port)
    except OSError as error:
        session.close()
        parser.error(f"게임을 열지 못했습니다: {error}. --port 0으로 빈 포트를 사용하세요.")
    session.new_game(PlaySettings(), session.revision)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Great Kingdom: {url}", flush=True)
    print("종료: 이 창에서 Ctrl+C. 창을 켜 둔 채 브라우저에서 플레이하세요.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
