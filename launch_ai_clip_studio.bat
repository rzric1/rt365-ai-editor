@echo off
setlocal
if not exist ".venv311\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup_windows.bat first.
    pause & exit /b 1
)
call .venv311\Scripts\activate.bat
streamlit run clip_studio_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
