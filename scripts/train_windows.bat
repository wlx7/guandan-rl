@echo off
rem Guandan AI - headless self-play training on this PC (GPU if available).
rem Usage: train_windows.bat [run-name] [preset] [hours]
rem Stop: create an empty file named STOP in guandan_data\runs\<run-name>\ , or press Ctrl+C.
chcp 65001 >nul
cd /d "%~dp0"
set "GUANDAN_HOME=%~dp0guandan_data"
set "NAME=%~1"
if "%NAME%"=="" set "NAME=main"
set "PRESET=%~2"
if "%PRESET%"=="" set "PRESET=medium"
set "HOURS=%~3"
if "%HOURS%"=="" set "HOURS=0"
set /a ACTORS=%NUMBER_OF_PROCESSORS%-4
if %ACTORS% LSS 1 set ACTORS=1
if %ACTORS% GTR 16 set ACTORS=16
"%~dp0.venv\Scripts\python.exe" -m guandan.train --name %NAME% --preset %PRESET% --device auto --actors %ACTORS% --envs 64 --eval-interval 900 --max-hours %HOURS%
pause
