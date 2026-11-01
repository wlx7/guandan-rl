@echo off
rem Guandan AI - start the web UI (training center + LAN battle)
chcp 65001 >nul
cd /d "%~dp0"
set "GUANDAN_HOME=%~dp0guandan_data"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Environment not found, running installer first...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" -Quiet
)
"%~dp0.venv\Scripts\python.exe" -m guandan serve %*
pause
