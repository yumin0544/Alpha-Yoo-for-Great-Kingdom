@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. See docs/match_ui.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "examples\play_ui.py" --open --page matches --port 0
if errorlevel 1 pause
