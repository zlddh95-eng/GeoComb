@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "SCRIPT=benchmark_geocomb_cpu_gpu_v2.py"
set "INPUT=C:\GeoComb_Benchmark_29\input"
set "CPU_EXE=C:\GeoComb\release\GeoComb_Windows_CPU\GeoComb_CPU\GeoComb_CPU.exe"
set "GPU_EXE=C:\GeoComb\release\GeoComb_Windows_GPU\GeoComb_GPU\GeoComb_GPU.exe"
set "OUT=C:\GeoComb_Benchmark_29\equivalence_once_v2"

if exist "%OUT%" rmdir /s /q "%OUT%"

"%PY%" "%SCRIPT%" ^
  --input-folder "%INPUT%" ^
  --cpu-exe "%CPU_EXE%" ^
  --gpu-exe "%GPU_EXE%" ^
  --output-root "%OUT%" ^
  --warmup-runs 0 ^
  --measured-runs 1

set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%
