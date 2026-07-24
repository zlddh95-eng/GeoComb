@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "OUT=%CD%\regression_output_cpu"

if not exist "%PY%" (
    echo ERROR: CPU environment not found.
    pause
    exit /b 1
)

if exist "%OUT%" rmdir /s /q "%OUT%"

"%PY%" check_environment.py
if errorlevel 1 (
    echo Environment validation FAILED.
    pause
    exit /b 1
)

"%PY%" geocomb_cli.py validate ^
  --device cpu ^
  --output-folder "%OUT%"

if errorlevel 1 (
    echo.
    echo Regression validation FAILED.
    echo Review:
    echo   %OUT%\regression_validation_summary.txt
    echo   %OUT%\regression_validation.json
    echo   %OUT%\project
    pause
    exit /b 1
)

echo.
echo Regression validation PASSED.
echo Results were preserved in:
echo   %OUT%
pause
