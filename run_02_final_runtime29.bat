@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "SCRIPT=benchmark_geocomb_cpu_gpu_v1.py"
set "INPUT=C:\GeoComb_Benchmark_29\input"
set "CPU_EXE=C:\GeoComb\release\GeoComb_Windows_CPU\GeoComb_CPU\GeoComb_CPU.exe"
set "GPU_EXE=C:\GeoComb\release\GeoComb_Windows_GPU\GeoComb_GPU\GeoComb_GPU.exe"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "OUT=C:\GeoComb_Benchmark_29\final_benchmark_%STAMP%"

echo ============================================================
echo GeoComb final CPU/GPU 29-image benchmark
echo 1 full warm-up + 3 measured runs per device
echo Temporary project folders are deleted after metric extraction.
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
    pause
    exit /b 1
)

"%PY%" "%SCRIPT%" ^
  --input-folder "%INPUT%" ^
  --cpu-exe "%CPU_EXE%" ^
  --gpu-exe "%GPU_EXE%" ^
  --output-root "%OUT%" ^
  --warmup-runs 1 ^
  --measured-runs 3

if errorlevel 1 (
    echo.
    echo FINAL BENCHMARK FAILED.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo FINAL BENCHMARK PASSED
echo.
echo Results:
echo   %OUT%
echo.
echo Main files:
echo   %OUT%\benchmark_summary.txt
echo   %OUT%\GeoComb_CPU_GPU_benchmark.xlsx
echo   %OUT%\manuscript_benchmark_text.txt
echo ============================================================
pause
