@echo off
rem Guandan AI - one-click installer (uv + Python 3.12 + CUDA PyTorch + guandan-rl)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
