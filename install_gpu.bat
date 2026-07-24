@echo off
setlocal
cd /d "%~dp0"
call repair_gpu_environment.bat
exit /b %ERRORLEVEL%
