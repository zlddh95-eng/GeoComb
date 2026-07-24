@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo GeoComb CPU environment repair v1.2
echo Project folder: %CD%
echo ============================================================
echo.

for %%I in ("%CD%") do set "PROJECT_PATH=%%~fI"

echo %PROJECT_PATH% | findstr /I /B "C:\GeoComb" >nul
if errorlevel 1 (
    echo ERROR: GeoComb must be run from a short path.
    echo.
    echo Move or copy this entire project folder to:
    echo     C:\GeoComb
    echo.
    echo Then run this BAT file from C:\GeoComb.
    echo The current Downloads path is too long for some Qt files.
    echo.
    pause
    exit /b 2
)

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python Launcher ^(py.exe^) was not found.
    echo Install 64-bit Python 3.11 and enable the Python Launcher.
    pause
    exit /b 3
)

echo Removing the incomplete CPU environment...
if exist ".venv_cpu" rmdir /s /q ".venv_cpu"

echo Creating a new Python 3.11 environment...
py -3.11 -m venv ".venv_cpu"
if errorlevel 1 goto :failed

set "PY=.venv_cpu\Scripts\python.exe"

echo Updating pip tools with a PyTorch-compatible setuptools version...
"%PY%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 goto :failed

echo Installing CPU PyTorch...
"%PY%" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
if errorlevel 1 goto :failed

echo Installing GeoComb dependencies...
"%PY%" -m pip install --no-cache-dir -r requirements-common.txt
if errorlevel 1 goto :failed

echo Checking installed package consistency...
"%PY%" -m pip check
if errorlevel 1 goto :failed

echo Running import validation...
"%PY%" check_environment.py
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo CPU environment repair PASSED.
echo Next, run:
echo     validate_cpu.bat
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo CPU environment repair FAILED.
echo Copy the complete output from this window for diagnosis.
echo ============================================================
pause
exit /b 1
