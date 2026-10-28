@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First follow the setup commands in README.md to create .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
