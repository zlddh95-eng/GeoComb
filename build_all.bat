@echo off
setlocal
cd /d "%~dp0"
echo Building CPU package...
call build_cpu.bat
if errorlevel 1 exit /b 1
echo Building GPU package...
call build_gpu.bat
if errorlevel 1 exit /b 1
echo Both packages were built successfully.
pause
