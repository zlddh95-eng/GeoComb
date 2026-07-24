@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "SCRIPT=benchmark_geocomb_cpu_gpu_v2.py"
set "INPUT=C:\GeoComb_Benchmark_29\input"
set "CPU_EXE=C:\GeoComb\release\GeoComb_Windows_CPU\GeoComb_CPU\GeoComb_CPU.exe"
set "GPU_EXE=C:\GeoComb\release\GeoComb_Windows_GPU\GeoComb_GPU\GeoComb_GPU.exe"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "OUT=C:\GeoComb_Benchmark_29\final_benchmark_v2_%STAMP%"

echo ============================================================
echo GeoComb final CPU/GPU 29-image benchmark v2
echo 1 warm-up + 3 measured runs per device
echo ============================================================
echo.

"%PY%" "%SCRIPT%" ^
  --input-folder "%INPUT%" ^
  --cpu-exe "%CPU_EXE%" ^
  --gpu-exe "%GPU_EXE%" ^
  --output-root "%OUT%" ^
  --warmup-runs 1 ^
  --measured-runs 3

set "CODE=%ERRORLEVEL%"

if "%CODE%"=="0" (
    echo.
    echo FINAL BENCHMARK PASSED:
    echo   %OUT%
) else (
    echo.
    echo FINAL BENCHMARK FAILED WITH CODE %CODE%.
    echo Partial timing and prediction snapshots are retained in:
    echo   %OUT%
)

pause
exit /b %CODE%
