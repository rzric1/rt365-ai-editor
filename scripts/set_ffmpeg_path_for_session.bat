@echo off
REM Prepend FFmpeg bin to PATH for this CMD session (winget Gyan.FFmpeg).
REM IMPORTANT: Do not use "FOR ... DO ( SET PATH=...%PATH%... )" — PATH often
REM contains "Program Files (x86)" and the "(" breaks CMD block parsing.

for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0find_ffmpeg_bin.ps1"`) do call :prepend_ffmpeg "%%F"
goto :eof

:prepend_ffmpeg
if exist "%~1\ffmpeg.exe" set "PATH=%~1;%PATH%"
exit /b 0
