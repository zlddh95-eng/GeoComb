@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "OUT=C:\GeoComb_Benchmark_29\equivalence_once"
set "CPU_SNAPSHOT=%OUT%\prediction_snapshots\CPU_measured_01.csv.gz"
set "GPU_EXE=C:\GeoComb\release\GeoComb_Windows_GPU\GeoComb_GPU\GeoComb_GPU.exe"

echo ============================================================
echo Resume GeoComb CPU/GPU equivalence after v1 cleanup failure
echo The completed 99.5-minute CPU run will be reused.
echo ============================================================
echo.

for %%F in (
    "%PY%"
    "benchmark_geocomb_cpu_gpu_v2.py"
    "resume_equivalence_after_cpu_v2.py"
    "%CPU_SNAPSHOT%"
    "%GPU_EXE%"
) do (
    if not exist "%%~F" (
        echo ERROR: Missing required file:
        echo   %%~F
        echo.
        echo Do not delete C:\GeoComb_Benchmark_29\equivalence_once.
        pause
        exit /b 1
    )
)

"%PY%" resume_equivalence_after_cpu_v2.py

set "CODE=%ERRORLEVEL%"

if "%CODE%"=="0" (
    echo.
    echo ============================================================
    echo RESUMED EQUIVALENCE CHECK PASSED
    echo.
    echo Summary:
    echo   %OUT%\benchmark_summary_resumed.txt
    echo.
    echo Excel:
    echo   %OUT%\GeoComb_CPU_GPU_equivalence_resumed.xlsx
    echo ============================================================
) else (
    echo.
    echo Resume finished with code %CODE%.
    echo Review:
    echo   %OUT%\benchmark_summary_resumed.txt
    echo   %OUT%\logs\GPU_measured_01_resume.log
)

pause
exit /b %CODE%
