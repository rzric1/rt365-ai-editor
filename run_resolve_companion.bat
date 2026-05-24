@echo off

cd /d "%~dp0"

if exist "%~dp0scripts\set_ffmpeg_path_for_session.bat" (
  call "%~dp0scripts\set_ffmpeg_path_for_session.bat"
)

if not exist ".venv\Scripts\activate.bat" (

    echo Create .venv and pip install -r requirements.txt first.

    pause

    exit /b 1

)

call ".venv\Scripts\activate.bat"

echo Launching DaVinci Resolve marker companion (legacy)...

start "" /B cmd /c "timeout /t 3 /nobreak >nul 2>nul && start http://localhost:8501/"

streamlit run app.py

pause


