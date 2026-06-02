@echo off
setlocal
REM RT365 AI Clip Studio — always launch via PowerShell (Python 3.11 .venv311 only)
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_ai_clip_studio.ps1"
exit /b %ERRORLEVEL%
