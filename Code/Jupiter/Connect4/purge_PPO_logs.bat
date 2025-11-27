@echo off
echo === Purging PPQ logs, models, and plots ===

set LOG_DIR=Logs\PPO
set MODEL_DIR=Models\PPO
set PLOTS_DIR=Plots\PPO

:: Delete contents of Logs/PPQ
if exist "%LOG_DIR%" (
    echo Cleaning %LOG_DIR%...
    del /q "%LOG_DIR%\*"
    for /d %%d in ("%LOG_DIR%\*") do rd /s /q "%%d"
)

:: Delete contents of Models/PPQ
if exist "%MODEL_DIR%" (
    echo Cleaning %MODEL_DIR%...
    del /q "%MODEL_DIR%\*"
    for /d %%d in ("%MODEL_DIR%\*") do rd /s /q "%%d"
)

:: Delete contents of Plots/PPQ
if exist "%PLOTS_DIR%" (
    echo Cleaning %PLOTS_DIR%...
    del /q "%PLOTS_DIR%\*"
    for /d %%d in ("%PLOTS_DIR%\*") do rd /s /q "%%d"
)

echo === Cleanup complete! ===
pause
