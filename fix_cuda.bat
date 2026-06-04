@echo off
setlocal EnableDelayedExpansion
echo ================================================
echo  RT365 CUDA Fix A — PyTorch cu128 wheels
echo  Virtual env: .venv311 (Python 3.11)
echo ================================================
echo.

cd /d "%~dp0"

if not exist ".venv311\Scripts\activate.bat" (
    echo [ERROR] .venv311 not found. Run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv311\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate .venv311
    pause
    exit /b 1
)

echo [INFO] Python: 
python --version
echo.
echo [INFO] Installing torch / torchvision / torchaudio from cu128 index (force reinstall)...
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

echo.
echo [OK] CUDA fix install finished.
echo [INFO] Re-run verification:
echo        .venv311\Scripts\python.exe cuda_verify.py
echo.
pause
