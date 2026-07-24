@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo GeoComb reproducible CPU environment v1.4
echo This recreates the environment using the Linux-reference
echo package versions where Windows wheels are available.
echo ============================================================
echo.

for %%I in ("%CD%") do set "PROJECT_PATH=%%~fI"
echo %PROJECT_PATH% | findstr /I /B "C:\GeoComb" >nul
if errorlevel 1 (
    echo ERROR: Run this from C:\GeoComb.
    pause
    exit /b 2
)

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher was not found.
    pause
    exit /b 3
)

if exist ".venv_cpu" rmdir /s /q ".venv_cpu"

py -3.11 -m venv ".venv_cpu"
if errorlevel 1 goto :failed

set "PY=.venv_cpu\Scripts\python.exe"

"%PY%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :failed

"%PY%" -m pip install ^
  --index-url https://download.pytorch.org/whl/cpu ^
  "torch==2.11.0+cpu" ^
  "torchvision==0.26.0+cpu"
if errorlevel 1 goto :failed

"%PY%" -m pip install --no-cache-dir -r requirements-reference-v1_4.txt
if errorlevel 1 goto :failed

"%PY%" -m pip check
if errorlevel 1 goto :failed

"%PY%" check_environment.py
if errorlevel 1 goto :failed

echo.
echo Reproducible CPU environment installation PASSED.
echo Next run validate_cpu.bat.
pause
exit /b 0

:failed
echo.
echo Reproducible CPU environment installation FAILED.
pause
exit /b 1
