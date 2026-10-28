@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First follow the setup commands in README.md to create .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000 --workers 1
pause

