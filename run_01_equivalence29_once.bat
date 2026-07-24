@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "SCRIPT=benchmark_geocomb_cpu_gpu_v1.py"
set "INPUT=C:\GeoComb_Benchmark_29\input"
set "CPU_EXE=C:\GeoComb\release\GeoComb_Windows_CPU\GeoComb_CPU\GeoComb_CPU.exe"
set "GPU_EXE=C:\GeoComb\release\GeoComb_Windows_GPU\GeoComb_GPU\GeoComb_GPU.exe"
set "OUT=C:\GeoComb_Benchmark_29\equivalence_once"

echo ============================================================
echo GeoComb CPU/GPU 29-image equivalence check
echo One measured run per device; no warm-up
echo ============================================================
echo.

for %%F in ("%PY%" "%SCRIPT%" "%CPU_EXE%" "%GPU_EXE%") do (
    if not exist "%%~F" (
        echo ERROR: Missing file:
        echo   %%~F
        pause
        exit /b 1
    )
)

if not exist "%INPUT%" (
    echo ERROR: Benchmark input folder was not found:
    echo   %INPUT%
    echo Run prepare_benchmark29_wsl_v1.sh in WSL first.
    pause
    exit /b 1
)

if exist "%OUT%" rmdir /s /q "%OUT%"

"%PY%" "%SCRIPT%" ^
  --input-folder "%INPUT%" ^
  --cpu-exe "%CPU_EXE%" ^
  --gpu-exe "%GPU_EXE%" ^
  --output-root "%OUT%" ^
  --warmup-runs 0 ^
  --measured-runs 1

if errorlevel 1 (
    echo.
    echo EQUIVALENCE CHECK FAILED.
    pause
    exit /b 1
)

echo.
echo Results:
echo   %OUT%\benchmark_summary.txt
echo   %OUT%\GeoComb_CPU_GPU_benchmark.xlsx
echo.
pause
