@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "DIST=build_output\cpu\dist\GeoComb_CPU"
set "RELEASEDIR=release\GeoComb_Windows_CPU"
set "RELEASEZIP=release\GeoComb_Windows_CPU.zip"

echo ============================================================
echo GeoComb CPU package existing build v1.8
echo ============================================================
echo.

if not exist "%DIST%\GeoComb_CPU.exe" (
    echo ERROR: The existing executable was not found:
    echo   C:\GeoComb\%DIST%\GeoComb_CPU.exe
    echo Run build_cpu.bat first.
    pause
    exit /b 1
)

if exist "%RELEASEDIR%" rmdir /s /q "%RELEASEDIR%"
if not exist "release" mkdir "release"
mkdir "%RELEASEDIR%" >nul 2>&1

echo Copying existing PyInstaller distribution...
robocopy "%DIST%" "%RELEASEDIR%\GeoComb_CPU" /E /R:2 /W:2 >nul
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
    pause
    exit /b 1
)

if not exist "%RELEASEZIP%" (
    echo ERROR: ZIP was not created.
    pause
    exit /b 1
)

for %%I in ("%RELEASEZIP%") do set "ZIP_SIZE=%%~zI"

echo.
echo ============================================================
echo PACKAGING PASSED
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
pause
