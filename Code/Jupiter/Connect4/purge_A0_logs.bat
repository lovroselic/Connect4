@echo off
echo === Purging A0 logs, models, and plots ===

set LOG_DIR=Logs\A0
set MODEL_DIR=Models\A0
set PLOTS_DIR=Plots\A0

:: Delete contents of Logs/A0
if exist "%LOG_DIR%" (
    echo Cleaning %LOG_DIR%...
    del /q "%LOG_DIR%\*"
    for /d %%d in ("%LOG_DIR%\*") do rd /s /q "%%d"
)

:: Delete contents of Models/A0
if exist "%MODEL_DIR%" (
    echo Cleaning %MODEL_DIR%...
    del /q "%MODEL_DIR%\*"
    for /d %%d in ("%MODEL_DIR%\*") do rd /s /q "%%d"
)

:: Delete contents of Plots/A0
if exist "%PLOTS_DIR%" (
    echo Cleaning %PLOTS_DIR%...
    del /q "%PLOTS_DIR%\*"
    for /d %%d in ("%PLOTS_DIR%\*") do rd /s /q "%%d"
)

echo === Cleanup complete! ===
pause
