@echo off

setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"



call "%~dp0scripts\set_ffmpeg_path_for_session.bat"



where ffmpeg >nul 2>&1

if errorlevel 1 (

  echo.

  echo ERROR: ffmpeg was not found on PATH.

  echo.

  echo Install FFmpeg, then open a NEW Command Prompt and run this script again:

  echo   winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements

  echo.

  echo If winget is unavailable, install from: https://www.gyan.dev/ffmpeg/builds/

  echo   ^(download "ffmpeg-release-essentials.zip", extract, add the ^"bin^" folder to PATH^)

  echo.

  echo Chocolatey ^(admin^): choco install ffmpeg

  echo.

  pause

  exit /b 1

)



echo.

echo --- FFmpeg ---

ffmpeg -version 2>nul | findstr /i "ffmpeg version"

echo.



set /p "INFILE=Drag and drop your video here, or type the full path, then press Enter: "

set "INFILE=!INFILE:"=!"



if not defined INFILE (

  echo No input file. Exiting.

  pause

  exit /b 1

)



if not exist "!INFILE!" (

  echo File not found: !INFILE!

  pause

  exit /b 1

)



set "OUTFILE=%~dp0compressed.mp4"



echo.

echo Input : !INFILE!

echo Output: !OUTFILE!

echo Preset: H.264 ^(NVENC if available^), AAC, width 1280, quality ~CRF 30 — good for AI clip analysis

echo.



set "PYEXE=%~dp0.venv\Scripts\python.exe"

if exist "!PYEXE!" (

  echo Using project Python + GPU/CPU auto encoder…

  "!PYEXE!" "%~dp0scripts\compress_encode.py" "!INFILE!" "!OUTFILE!"

  if errorlevel 1 (

    echo.

    echo compress_encode.py failed.

    pause

    exit /b 1

  )

  goto :done_ok

)



echo No .venv Python — using inline ffmpeg ^(NVENC then CPU fallback^)…

set "TMPFILE=%~dp0compressed.tmp.mp4"



ffmpeg -y -hide_banner -loglevel warning -stats -i "!INFILE!" -vf "scale=1280:-2:flags=lanczos,format=yuv420p" -c:v h264_nvenc -preset p4 -tune hq -rc vbr -cq 30 -b:v 0 -c:a aac -b:a 128k -ac 2 -movflags +faststart "!TMPFILE!" 2>nul

if errorlevel 1 (

  echo NVENC failed or unavailable — using libx264…

  ffmpeg -y -hide_banner -loglevel warning -stats -i "!INFILE!" -vf "scale=1280:-2:flags=lanczos,format=yuv420p" -c:v libx264 -preset medium -crf 30 -c:a aac -b:a 128k -ac 2 -movflags +faststart "!OUTFILE!"

  if errorlevel 1 (

    echo FFmpeg reported an error.

    pause

    exit /b 1

  )

) else (

  move /y "!TMPFILE!" "!OUTFILE!" >nul

)



:done_ok

echo.

echo Done. Upload !OUTFILE! in Clip Studio ^(MP4^).

echo.

pause

exit /b 0


