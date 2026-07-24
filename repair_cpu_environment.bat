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

if not exist ".venv_cpu\Scripts\python.exe" (
  echo Creating CPU virtual environment...
  py -3.11 -m venv .venv_cpu
  if errorlevel 1 goto :fail
)

set "PY=.venv_cpu\Scripts\python.exe"

echo Updating pip tools...
"%PY%" -m pip install --upgrade pip wheel setuptools
if errorlevel 1 goto :fail

"%PY%" -c "import torch, torchvision" >nul 2>nul
if errorlevel 1 (
  echo Installing CPU PyTorch...
  "%PY%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto :fail
) else (
  echo PyTorch is already installed.
)

echo Installing or repairing GeoComb dependencies...
"%PY%" -m pip install --upgrade -r requirements-common.txt
if errorlevel 1 goto :fail

echo Checking dependency consistency...
"%PY%" -m pip check
if errorlevel 1 goto :fail

echo Validating imports...
"%PY%" check_environment.py --device cpu
if errorlevel 1 goto :fail

echo.
echo CPU environment repair PASSED.
echo Next run: validate_cpu.bat
pause
exit /b 0

:fail
echo.
echo CPU environment repair FAILED.
echo Copy the complete output from this window for diagnosis.
pause
exit /b 1
