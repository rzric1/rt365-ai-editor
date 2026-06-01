@echo off
setlocal EnableDelayedExpansion
echo ================================================
echo  RT365 AI Clip Studio - Windows Setup
echo ================================================
echo.

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found.
    echo Download from: https://www.python.org/downloads/release/python-3119/
    echo Check "Add Python to PATH" during install.
    pause & exit /b 1
)
echo [OK] Python 3.11 found.

if not exist ".venv311" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv311
    if errorlevel 1 ( echo [ERROR] venv creation failed. && pause && exit /b 1 )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

call .venv311\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 ( echo [ERROR] Package install failed. && pause && exit /b 1 )
echo [OK] Packages installed.

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg not found. Download from https://ffmpeg.org/download.html
) else (
    echo [OK] ffmpeg found.
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Created .env — open it and add your OPENAI_API_KEY
    )
)

echo.
python check_environment.py
echo.
echo ================================================
echo  Setup complete. Run launch_ai_clip_studio.bat
echo ================================================
pause
