@echo off
setlocal EnableExtensions
REM ============================================================================
REM AI Clip Studio — delegates to launch_ai_clip_studio.ps1 (visible console).
REM Keeps this .bat for double-click compatibility; shortcuts use PowerShell.
REM ============================================================================

cd /d "C:\dev\rt365-ai-editor"
if errorlevel 1 (
    echo ERROR: Could not cd to C:\dev\rt365-ai-editor
    echo Edit the path in launch_ai_clip_studio.bat if your clone is elsewhere.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File "C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1"
set "CLIP_STUDIO_EXIT=%ERRORLEVEL%"
exit /b %CLIP_STUDIO_EXIT%
