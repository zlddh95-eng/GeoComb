@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_cpu\Scripts\python.exe"
set "SPEC=build\GeoComb.spec"
set "BUILDROOT=build_output\cpu"
set "DIST=%BUILDROOT%\dist"
set "WORK=%BUILDROOT%\work"
set "RELEASEDIR=release\GeoComb_Windows_CPU"
set "RELEASEZIP=release\GeoComb_Windows_CPU.zip"
set "LOG=%BUILDROOT%\build_cpu.log"

echo ============================================================
echo GeoComb Windows CPU build v1.8
echo Project: %CD%
echo ============================================================
echo.

if not exist "%PY%" (
    echo ERROR: %PY% was not found.
    exit /b 1
)

if not exist "%SPEC%" (
    echo ERROR: %SPEC% was not found.
    exit /b 1
)

for %%F in (
    "geocomb_entry.py"
    "models\frame_roi_yolo26m_1024_best.pt"
    "models\stage1_best.pt"
    "models\stage2_best.pt"
    "models\stage3_best.pt"
    "pipeline_scripts\alternating_regularize_localhex_from_gray_step1_v1.py"
) do (
    if not exist "%%~F" (
        echo ERROR: Required build asset is missing:
        echo   %%~F
        exit /b 1
    )
)

echo Validating Python environment...
"%PY%" check_environment.py
if errorlevel 1 (
    echo Environment validation FAILED.
    exit /b 1
)

if exist "%BUILDROOT%" rmdir /s /q "%BUILDROOT%"
mkdir "%BUILDROOT%" >nul 2>&1
mkdir "%DIST%" >nul 2>&1
mkdir "%WORK%" >nul 2>&1

set "GEOCOMB_VARIANT=CPU"

echo.
echo Building GeoComb_CPU.exe...
echo Full log:
echo   C:\GeoComb\%LOG%
echo.

"%PY%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --distpath "%DIST%" ^
    --workpath "%WORK%" ^
    "%SPEC%" > "%LOG%" 2>&1

set "BUILD_CODE=%ERRORLEVEL%"
type "%LOG%"

if not "%BUILD_CODE%"=="0" (
    echo.
    echo PyInstaller build FAILED with code %BUILD_CODE%.
    echo Review:
    echo   C:\GeoComb\%LOG%
    exit /b %BUILD_CODE%
)

if not exist "%DIST%\GeoComb_CPU\GeoComb_CPU.exe" (
    echo.
    echo ERROR: PyInstaller returned success, but the executable was not found:
    echo   %DIST%\GeoComb_CPU\GeoComb_CPU.exe
    exit /b 1
)

call :package_existing_build
exit /b %ERRORLEVEL%

:package_existing_build
if exist "%RELEASEDIR%" rmdir /s /q "%RELEASEDIR%"
if not exist "release" mkdir "release"
mkdir "%RELEASEDIR%" >nul 2>&1

echo.
echo Copying distribution files...
robocopy "%DIST%\GeoComb_CPU" "%RELEASEDIR%\GeoComb_CPU" /E /R:2 /W:2 >nul
set "ROBOCOPY_CODE=%ERRORLEVEL%"

if %ROBOCOPY_CODE% GEQ 8 (
    echo ERROR: Robocopy failed with code %ROBOCOPY_CODE%.
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

if exist "%RELEASEZIP%" del /f /q "%RELEASEZIP%"

echo Creating ZIP archive...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$source=(Resolve-Path '%RELEASEDIR%').Path;" ^
  "$destination=Join-Path (Get-Location) '%RELEASEZIP%';" ^
  "Compress-Archive -Path (Join-Path $source '*') -DestinationPath $destination -CompressionLevel Optimal -Force;" ^
  "if (-not (Test-Path -LiteralPath $destination)) { throw 'ZIP was not created.' }"

if errorlevel 1 (
    echo ZIP creation FAILED.
    exit /b 1
)

if not exist "%RELEASEZIP%" (
    echo ERROR: ZIP was not created.
    exit /b 1
)

for %%I in ("%RELEASEZIP%") do set "ZIP_SIZE=%%~zI"

echo.
echo ============================================================
echo BUILD PASSED
echo.
echo Executable:
echo   C:\GeoComb\%RELEASEDIR%\GeoComb_CPU\GeoComb_CPU.exe
echo.
echo ZIP:
echo   C:\GeoComb\%RELEASEZIP%
echo.
echo ZIP size:
echo   %ZIP_SIZE% bytes
echo ============================================================
exit /b 0
