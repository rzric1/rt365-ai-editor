@echo off
setlocal EnableDelayedExpansion
echo ================================================
echo  RT365 AI Clip Studio - Windows Setup
echo  Python 3.11 ONLY — .venv311
echo ================================================
echo.

cd /d "%~dp0"

REM Block using Python 3.14 for this project
py -3.14 --version >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Python 3.14 is installed — Clip Studio will NOT use it.
    echo        Use .venv311 with Python 3.11 only.
    echo.
)

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found.
    echo Download: https://www.python.org/downloads/release/python-3119/
    echo Enable "Add Python to PATH" and py launcher.
    pause & exit /b 1
)
echo [OK] Python 3.11 found.

if not exist ".venv311" (
    echo Creating .venv311 ...
    py -3.11 -m venv .venv311
    if errorlevel 1 ( echo [ERROR] venv creation failed. && pause && exit /b 1 )
    echo [OK] Virtual environment created.
) else (
    echo [OK] .venv311 already exists.
)

call .venv311\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] requirements.txt install failed. && pause && exit /b 1 )

if exist requirements-ai-upgrades.txt (
    echo Installing AI upgrades ^(opencv, ultralytics, sentence-transformers^)...
    pip install -r requirements-ai-upgrades.txt -q
)

echo Installing CUDA PyTorch for RTX 4090 ^(cu121 wheels^)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

echo.
echo Verifying packages...
python -c "import streamlit, numpy, psutil, openai; print('[OK] core imports')"
python -c "import faster_whisper, ctranslate2; print('[OK] whisper stack')" 2>nul || echo [WARN] faster-whisper/ctranslate2 - check CUDA DLLs
python -c "import torch; print('[OK] torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>nul || echo [WARN] torch optional
python -c "import cv2; print('[OK] opencv')" 2>nul || echo [WARN] opencv optional

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg not on PATH. Install FFmpeg or set FFMPEG_BINARY in .env
) else (
    echo [OK] ffmpeg found.
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Created .env — add OPENAI_API_KEY
    )
)

if not exist "logs" mkdir logs
python -c "from clip_engine.environment_check import write_environment_check_log; write_environment_check_log(); print('[OK] wrote logs/environment_check.txt')"

echo.
echo ================================================
echo  Setup complete.
echo  Launch: launch_ai_clip_studio.ps1
echo ================================================
pause
