@echo off
setlocal
cd /d "%~dp0"
call repair_cpu_environment.bat
exit /b %ERRORLEVEL%
