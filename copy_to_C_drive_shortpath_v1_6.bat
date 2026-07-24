@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "SOURCE=%CD%"
set "TARGET=C:\GeoComb"

echo Copying GeoComb to %TARGET%...
if not exist "%TARGET%" mkdir "%TARGET%"

robocopy "%SOURCE%" "%TARGET%" /E /R:2 /W:2 ^
  /XD ".venv_cpu" ".venv_gpu" "build_output" "dist" "release" ^
      "regression_output_cpu" "regression_output_gpu" "__pycache__" ^
  /XF "*.pyc"

set "ROBOCOPY_CODE=%ERRORLEVEL%"

if %ROBOCOPY_CODE% GEQ 8 (
    echo Robocopy failed with code %ROBOCOPY_CODE%.
    pause
    exit /b %ROBOCOPY_CODE%
)

echo Copy complete.
echo Note: the source-code build folder is included.
pause
