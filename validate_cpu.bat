@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "OUT=%CD%\regression_output_cpu"

if not exist "%PY%" (
    echo ERROR: CPU environment not found.
    echo Run install_cpu.bat first.
    pause
    exit /b 1
)

"%PY%" check_environment.py
if errorlevel 1 (
    echo Environment validation FAILED.
    pause
    exit /b 1
)

if not exist "regression_test\input\1.JPG" goto :missing_assets
if not exist "regression_test\expected\cell_predictions_3stage.csv" goto :missing_assets

if exist "%OUT%" rmdir /s /q "%OUT%"

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
exit /b 0

:missing_assets
echo.
echo OPTIONAL REGRESSION CHECK NOT RUN.
echo The authorized local regression_test assets were not found.
echo They are not redistributed in the public source repository.
echo The CPU Python environment check passed successfully.
pause
exit /b 2
