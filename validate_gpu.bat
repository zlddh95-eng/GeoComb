@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_gpu\Scripts\python.exe"
set "OUT=%CD%\regression_output_gpu"

echo ============================================================
echo GeoComb NVIDIA GPU regression validation v1.9
echo ============================================================
echo.

if not exist "%PY%" (
    echo ERROR: GPU environment not found.
    echo Run install_gpu_v1_9.bat first.
    pause
    exit /b 1
)

"%PY%" check_gpu_environment_v1_9.py
if errorlevel 1 (
    echo GPU environment validation FAILED.
    pause
    exit /b 1
)

if exist "%OUT%" rmdir /s /q "%OUT%"

"%PY%" geocomb_cli.py validate ^
  --device cuda ^
  --output-folder "%OUT%"

if errorlevel 1 (
    echo.
    echo GPU regression validation FAILED.
    echo Review:
    echo   %OUT%\regression_validation_summary.txt
    echo   %OUT%\regression_validation.json
    echo   %OUT%\project
    pause
    exit /b 1
)

echo.
echo ============================================================
echo GPU REGRESSION VALIDATION PASSED
echo Results:
echo   %OUT%
echo ============================================================
pause
exit /b 0
