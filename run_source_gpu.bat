@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_gpu\Scripts\python.exe"

if not exist "%PY%" (
    echo ERROR: GPU environment not found.
    echo Run install_gpu_v1_9.bat first.
    pause
    exit /b 1
)

"%PY%" check_gpu_environment_v1_9.py
if errorlevel 1 (
    pause
    exit /b 1
)

"%PY%" geocomb_gpu_entry.py
