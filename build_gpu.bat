@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv_gpu\Scripts\python.exe"
set "SPEC=build\GeoComb.spec"
set "BUILDROOT=build_output\gpu"
set "DIST=%BUILDROOT%\dist"
set "WORK=%BUILDROOT%\work"
set "RELEASEDIR=release\GeoComb_Windows_GPU"
set "RELEASEZIP=release\GeoComb_Windows_GPU.zip"
set "LOG=%BUILDROOT%\build_gpu.log"
set "LICENSESRC=licenses\windows_cuda"
set "SOURCE_URL=https://github.com/zlddh95-eng/GeoComb/releases/tag/v1.0.0"

echo ============================================================
echo GeoComb Windows NVIDIA GPU platform build
echo Project: %CD%
echo ============================================================
echo.

if not exist "%PY%" (
    echo ERROR: GPU environment not found.
    echo Run install_gpu.bat first.
    exit /b 1
)

if not exist "%SPEC%" (
    echo ERROR: %SPEC% was not found.
    exit /b 1
)

for %%F in (
    "geocomb_gpu_entry.py"
    "models\frame_roi_yolo26m_1024_best.pt"
    "models\stage1_best.pt"
    "models\stage2_best.pt"
    "models\stage3_best.pt"
    "pipeline_scripts\alternating_regularize_localhex_from_gray_step1_v1.py"
    "LICENSE"
    "MODEL_CARD.md"
    "ASSET_MANIFEST.json"
    "README.md"
    "README_KOREAN.md"
    "docs\USER_GUIDE.md"
    "docs\THIRD_PARTY_NOTICES.md"
    "docs\BUILD_AND_VALIDATION.md"
    "docs\KNOWN_LIMITATIONS.md"
    "%LICENSESRC%\requirements_frozen.txt"
    "%LICENSESRC%\third_party_packages.csv"
) do (
    if not exist "%%~F" (
        echo ERROR: Required build or release asset is missing:
        echo   %%~F
        exit /b 1
    )
)

dir /b /s "%LICENSESRC%\license_files\*" >nul 2>&1
if errorlevel 1 (
    echo ERROR: No collected license files were found under:
    echo   %LICENSESRC%\license_files
    exit /b 1
)

echo Validating GPU environment...
"%PY%" check_gpu_environment_v1_9.py
if errorlevel 1 (
    echo GPU environment validation FAILED.
    exit /b 1
)

if exist "%BUILDROOT%" rmdir /s /q "%BUILDROOT%"
mkdir "%BUILDROOT%" >nul 2>&1
mkdir "%DIST%" >nul 2>&1
mkdir "%WORK%" >nul 2>&1

set "GEOCOMB_VARIANT=GPU"

echo.
echo Building GeoComb_GPU.exe...
echo This can take several minutes and use substantial disk space.
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
    echo PyInstaller GPU build FAILED with code %BUILD_CODE%.
    echo Review:
    echo   C:\GeoComb\%LOG%
    exit /b %BUILD_CODE%
)

if not exist "%DIST%\GeoComb_GPU\GeoComb_GPU.exe" (
    echo ERROR: GPU executable was not created.
    exit /b 1
)

if exist "%RELEASEDIR%" rmdir /s /q "%RELEASEDIR%"
if not exist "release" mkdir "release"
mkdir "%RELEASEDIR%" >nul 2>&1

echo.
echo Copying GPU distribution files...
robocopy "%DIST%\GeoComb_GPU" ^
  "%RELEASEDIR%\GeoComb_GPU" ^
  /E /R:2 /W:2 >nul

set "ROBOCOPY_CODE=%ERRORLEVEL%"

if %ROBOCOPY_CODE% GEQ 8 (
    echo ERROR: Robocopy failed with code %ROBOCOPY_CODE%.
    exit /b %ROBOCOPY_CODE%
)

for %%F in (
    "README.md"
    "README_KOREAN.md"
    "docs\USER_GUIDE.md"
    "docs\THIRD_PARTY_NOTICES.md"
    "docs\BUILD_AND_VALIDATION.md"
    "docs\KNOWN_LIMITATIONS.md"
    "LICENSE"
    "MODEL_CARD.md"
    "ASSET_MANIFEST.json"
) do (
    copy /y "%%~F" "%RELEASEDIR%\" >nul
    if errorlevel 1 (
        echo ERROR: Failed to copy release asset:
        echo   %%~F
        exit /b 1
    )
)

robocopy "%LICENSESRC%" "%RELEASEDIR%\%LICENSESRC%" /E /R:2 /W:2 >nul
set "ROBOCOPY_CODE=%ERRORLEVEL%"

if %ROBOCOPY_CODE% GEQ 8 (
    echo ERROR: License inventory copy failed with code %ROBOCOPY_CODE%.
    exit /b %ROBOCOPY_CODE%
)

(
    echo GeoComb 1.0.0 corresponding source release:
    echo %SOURCE_URL%
) > "%RELEASEDIR%\SOURCE_RELEASE.txt"

if not exist "%RELEASEDIR%\SOURCE_RELEASE.txt" (
    echo ERROR: SOURCE_RELEASE.txt was not created.
    exit /b 1
)

echo.
echo Creating large-file-safe ZIP64 archive...
"%PY%" package_release_zip_v1_9.py ^
  --source "%RELEASEDIR%" ^
  --destination "%RELEASEZIP%"

if errorlevel 1 (
    echo ZIP creation FAILED.
    exit /b 1
)

if not exist "%RELEASEZIP%" (
    echo ERROR: GPU ZIP was not created.
    exit /b 1
)

for %%I in ("%RELEASEZIP%") do set "ZIP_SIZE=%%~zI"

echo.
echo ============================================================
echo GPU BUILD PASSED
echo.
echo Executable:
echo   C:\GeoComb\%RELEASEDIR%\GeoComb_GPU\GeoComb_GPU.exe
echo.
echo ZIP:
echo   C:\GeoComb\%RELEASEZIP%
echo.
echo ZIP size:
echo   %ZIP_SIZE% bytes
echo ============================================================
exit /b 0
