@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=.venv_cpu\Scripts\python.exe"

if not exist "%PY%" (
  echo Run install_cpu.bat first.
  pause
  exit /b 1
)

"%PY%" check_environment.py --device cpu
if errorlevel 1 (
  echo Run repair_cpu_environment.bat.
  pause
  exit /b 1
)

"%PY%" geocomb_gui.py
if errorlevel 1 pause
