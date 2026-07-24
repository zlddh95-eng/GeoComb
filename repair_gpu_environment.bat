@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found.
  echo Install 64-bit Python 3.11 from python.org and select Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv_gpu\Scripts\python.exe" (
  echo Creating GPU virtual environment...
  py -3.11 -m venv .venv_gpu
  if errorlevel 1 goto :fail
)

set "PY=.venv_gpu\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 goto :fail

"%PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
  echo Installing CUDA PyTorch...
  "%PY%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 goto :fail
)

"%PY%" -m pip install --upgrade -r requirements-common.txt
if errorlevel 1 goto :fail
"%PY%" -m pip check
if errorlevel 1 goto :fail
"%PY%" check_environment.py --device cuda
if errorlevel 1 goto :fail

echo.
echo GPU environment repair PASSED.
echo Next run: validate_gpu.bat
pause
exit /b 0

:fail
echo.
echo GPU environment repair FAILED.
echo Copy the complete output from this window for diagnosis.
pause
exit /b 1
