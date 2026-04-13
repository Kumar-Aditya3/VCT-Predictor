@echo off
REM Setup script for VCT Tier 1 Predictor
REM Runs on Windows to initialize project and register weekly scheduler job

echo.
echo ====== VCT Tier 1 Predictor Setup ======
echo.

REM Configure paths
set PROJECT_DIR=%~dp0
set BACKEND_DIR=%PROJECT_DIR%backend
set FRONTEND_DIR=%PROJECT_DIR%frontend
set OPS_DIR=%PROJECT_DIR%ops
set PYTHON_EXE=C:\Users\kumar\AppData\Local\Programs\Python\Python313\python.exe

echo [1/4] Installing Python backend dependencies...
cd %BACKEND_DIR%
%PYTHON_EXE% -m pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies
    exit /b 1
)
echo OK: Backend dependencies installed

echo.
echo [2/4] Installing frontend dependencies...
cd %FRONTEND_DIR%
call npm install --silent
if %errorlevel% neq 0 (
    echo WARNING: Failed to install frontend dependencies
)
echo OK: Frontend dependencies installed

echo.
echo [3/4] Registering weekly scheduler task...
REM Create a scheduled task to run weekly pipeline
powershell -Command ^
    $TaskName = 'VCT-Predictor-Weekly'; ^
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 6am; ^
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Bypass -File ""%OPS_DIR%\tasks\weekly_refresh.ps1""'; ^
    $settings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew; ^
    Register-ScheduledTask -TaskName $TaskName -Trigger $trigger -Action $action -Settings $settings -Force | Out-Null; ^
    Write-Host 'OK: Weekly task registered (Monday 6am local time)'

echo.
echo [4/4] Initialization complete!
echo.
echo Next steps:
echo   - Backend API: cd backend && python -m uvicorn app.main:app --reload
echo   - Frontend: cd frontend && npm run dev
echo   - Dashboard: http://localhost:3000
echo.
pause
