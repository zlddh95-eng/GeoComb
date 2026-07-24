@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo GeoComb NVIDIA GPU environment installer v1.9
echo Project: %CD%
echo ============================================================
echo.

for %%I in ("%CD%") do set "PROJECT_PATH=%%~fI"
if /I not "%PROJECT_PATH%"=="C:\GeoComb" (
    echo ERROR: Run this installer from C:\GeoComb.
    echo Current path:
    echo   %PROJECT_PATH%
    pause
    exit /b 2
)

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher was not found.
    echo Install 64-bit Python 3.11 first.
    pause
    exit /b 3
)

echo Running NVIDIA GPU and disk-space preflight...
py -3.11 gpu_preflight_v1_9.py
if errorlevel 1 (
    echo.
    echo GPU preflight FAILED.
    pause
    exit /b 4
)

echo.
echo Removing any incomplete GPU environment...
if exist ".venv_gpu" rmdir /s /q ".venv_gpu"

echo Creating Python 3.11 GPU environment...
py -3.11 -m venv ".venv_gpu"
if errorlevel 1 goto :failed

set "PY=.venv_gpu\Scripts\python.exe"

echo Updating pip tools...
"%PY%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :failed

echo.
echo Installing PyTorch 2.11 with CUDA 12.8...
"%PY%" -m pip install --no-cache-dir ^
  "torch==2.11.0" ^
  "torchvision==0.26.0" ^
  --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed

echo.
echo Installing GeoComb GPU dependencies...
"%PY%" -m pip install --no-cache-dir ^
  -r requirements-gpu-v1_9.txt
if errorlevel 1 goto :failed

echo.
echo Checking package consistency...
"%PY%" -m pip check
if errorlevel 1 goto :failed

echo.
echo Running CUDA computation and module validation...
"%PY%" check_gpu_environment_v1_9.py
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo GPU ENVIRONMENT INSTALLATION PASSED
echo.
echo Next:
echo   1. validate_gpu.bat
echo   2. run_source_gpu.bat
echo   3. build_gpu.bat
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo GPU ENVIRONMENT INSTALLATION FAILED
echo Review the full output above.
echo ============================================================
pause
exit /b 1
