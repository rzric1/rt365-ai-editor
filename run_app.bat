@echo off
setlocal EnableExtensions
REM ============================================================================
REM AI Clip Studio launcher (Streamlit + .venv + FFmpeg on PATH)
REM From PowerShell:  cmd /c "%CD%\run_app.bat"
REM Double-click: OK in Explorer
REM ============================================================================

echo.
echo [%TIME%] AI Clip Studio - starting launcher...
echo [%TIME%] Project directory: %~dp0

cd /d "%~dp0"
if errorlevel 1 (
    echo [%TIME%] ERROR: Could not change to project folder.
    pause
    exit /b 1
)

if exist "%~dp0scripts\set_ffmpeg_path_for_session.bat" (
    echo [%TIME%] Prepending FFmpeg to PATH (WinGet helper)...
    call "%~dp0scripts\set_ffmpeg_path_for_session.bat"
) else (
    echo [%TIME%] WARNING: scripts\set_ffmpeg_path_for_session.bat not found.
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [%TIME%] WARNING: ffmpeg not found on PATH. Install FFmpeg or add it to PATH.
) else (
    echo [%TIME%] ffmpeg on PATH:
    where ffmpeg
    echo [%TIME%] ffmpeg -version (first line^):
    for /f "usebackq delims=" %%L in (`ffmpeg -hide_banner -version 2^>nul`) do (
        echo [%TIME%] %%L
        goto :_ff_done
    )
)
:_ff_done

if not exist "clip_studio_app.py" (
    echo.
    echo [%TIME%] ERROR: clip_studio_app.py was not found in:
    echo            %CD%
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [%TIME%] ERROR: Virtual environment not found at:
    echo            %CD%\.venv
    echo.
    echo Create it from this folder:
    echo   py -m venv .venv
    echo   .venv\Scripts\activate
    echo   py -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [%TIME%] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [%TIME%] ERROR: Could not activate .venv
    pause
    exit /b 1
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo [%TIME%] ERROR: streamlit.exe not found in .venv\Scripts
    echo            Run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [%TIME%] Python:
where python 2>nul
echo [%TIME%] Launching Streamlit: clip_studio_app.py
echo [%TIME%] URL: http://localhost:8501/ (browser may open automatically)
echo.

start "" /B cmd /c "timeout /t 3 /nobreak >nul 2>nul && start http://localhost:8501/"

streamlit run clip_studio_app.py
set "CLIP_STUDIO_EXIT=%ERRORLEVEL%"

echo.
if not "%CLIP_STUDIO_EXIT%"=="0" (
    echo [%TIME%] Streamlit exited with code %CLIP_STUDIO_EXIT%.
) else (
    echo [%TIME%] Streamlit stopped normally.
)
echo.
pause
endlocal & exit /b %CLIP_STUDIO_EXIT%
