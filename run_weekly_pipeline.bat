@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%backend"
set "PYTHON_EXE=C:\Users\kumar\AppData\Local\Programs\Python\Python313\python.exe"
set "INIT_MARKER=%PROJECT_DIR%.pipeline_initialized"
set "FORCE_INIT=0"

if /I "%~1"=="--init" set "FORCE_INIT=1"

if not exist "%BACKEND_DIR%\requirements.txt" (
  echo ERROR: backend\requirements.txt not found
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo WARNING: Configured Python not found. Falling back to PATH python.
  set "PYTHON_EXE=python"
)

if "%FORCE_INIT%"=="1" (
  del /f /q "%INIT_MARKER%" >nul 2>nul
)

if not exist "%INIT_MARKER%" (
  echo [Init] Installing backend dependencies and creating data directories...
  pushd "%BACKEND_DIR%"
  "%PYTHON_EXE%" -m pip install -r requirements.txt
  if errorlevel 1 (
    popd
    echo ERROR: Backend dependency installation failed.
    exit /b 1
  )
  popd

  if not exist "%PROJECT_DIR%artifacts" mkdir "%PROJECT_DIR%artifacts"
  if not exist "%PROJECT_DIR%data" mkdir "%PROJECT_DIR%data"
  if not exist "%PROJECT_DIR%data\raw" mkdir "%PROJECT_DIR%data\raw"
  if not exist "%PROJECT_DIR%data\processed" mkdir "%PROJECT_DIR%data\processed"
  if not exist "%PROJECT_DIR%data\metadata" mkdir "%PROJECT_DIR%data\metadata"

  > "%INIT_MARKER%" echo initialized %date% %time%
  echo [Init] Complete.
) else (
  echo [Init] Already initialized. Use --init to force reinstall.
)

echo [Run] Executing weekly pipeline...
pushd "%BACKEND_DIR%"
"%PYTHON_EXE%" -m scripts.weekly_update
if errorlevel 1 (
  popd
  echo ERROR: Weekly update failed.
  exit /b 1
)

echo [Run] Executing validation report...
"%PYTHON_EXE%" -m scripts.validate_vlr_ground_truth
set "RUN_CODE=%ERRORLEVEL%"
popd

if %RUN_CODE% neq 0 (
  echo WARNING: Validation script failed with exit code %RUN_CODE%.
  exit /b %RUN_CODE%
)

echo [Done] Weekly pipeline and validation completed successfully.
exit /b 0