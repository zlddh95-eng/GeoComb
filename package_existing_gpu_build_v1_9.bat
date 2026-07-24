@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_gpu\Scripts\python.exe"
set "DIST=build_output\gpu\dist\GeoComb_GPU"
set "RELEASEDIR=release\GeoComb_Windows_GPU"
set "RELEASEZIP=release\GeoComb_Windows_GPU.zip"

if not exist "%PY%" (
    echo ERROR: GPU environment was not found.
    pause
    exit /b 1
)

if not exist "%DIST%\GeoComb_GPU.exe" (
    echo ERROR: Existing GPU executable was not found:
    echo   C:\GeoComb\%DIST%\GeoComb_GPU.exe
    pause
    exit /b 1
)

if exist "%RELEASEDIR%" rmdir /s /q "%RELEASEDIR%"
if not exist "release" mkdir "release"
mkdir "%RELEASEDIR%" >nul 2>&1

robocopy "%DIST%" "%RELEASEDIR%\GeoComb_GPU" ^
  /E /R:2 /W:2 >nul

set "ROBOCOPY_CODE=%ERRORLEVEL%"

if %ROBOCOPY_CODE% GEQ 8 (
    echo ERROR: Robocopy failed with code %ROBOCOPY_CODE%.
    pause
    exit /b %ROBOCOPY_CODE%
)

for %%F in (
    "docs\USER_GUIDE.md"
    "docs\THIRD_PARTY_NOTICES.md"
    "docs\BUILD_AND_VALIDATION.md"
    "docs\KNOWN_LIMITATIONS.md"
) do (
    if exist "%%~F" copy /y "%%~F" "%RELEASEDIR%\" >nul
)

"%PY%" package_release_zip_v1_9.py ^
  --source "%RELEASEDIR%" ^
  --destination "%RELEASEZIP%"

if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo GPU packaging PASSED:
echo   C:\GeoComb\%RELEASEZIP%
pause
