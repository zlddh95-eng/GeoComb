@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "SOURCE=%CD%"
set "TARGET=C:\GeoComb"

echo ============================================================
echo Copy GeoComb to a short Windows path
echo Source: %SOURCE%
echo Target: %TARGET%
echo ============================================================
echo.

if /I "%SOURCE%"=="%TARGET%" (
    echo GeoComb is already located at C:\GeoComb.
    call repair_cpu_environment_v1_2.bat
    exit /b %errorlevel%
)

if not exist "%TARGET%" mkdir "%TARGET%"

echo Copying project files...
robocopy "%SOURCE%" "%TARGET%" /E /R:2 /W:2 ^
  /XD ".venv_cpu" ".venv_gpu" "build" "dist" "release" ^
      "regression_output_cpu" "regression_output_gpu" "__pycache__" ^
  /XF "*.pyc"

set "ROBOCOPY_CODE=%ERRORLEVEL%"
if %ROBOCOPY_CODE% GEQ 8 (
    echo.
    echo ERROR: Robocopy failed with code %ROBOCOPY_CODE%.
    pause
    exit /b %ROBOCOPY_CODE%
)

echo.
echo Copy completed.
echo Repairing the CPU environment under C:\GeoComb...
cd /d "%TARGET%"
call repair_cpu_environment_v1_2.bat
exit /b %errorlevel%
