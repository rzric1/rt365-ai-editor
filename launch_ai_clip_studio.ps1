#Requires -Version 5.1
<#
.SYNOPSIS
  AI Clip Studio — Streamlit launcher (visible console for logs).

  Double-click the desktop shortcut, or run:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1"
#>

$ErrorActionPreference = 'Stop'

# --- Project root (fixed path per machine setup) ---
$ProjectRoot = 'C:\dev\rt365-ai-editor'
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    Write-Host "ERROR: Project folder not found: $ProjectRoot" -ForegroundColor Red
    Write-Host "Edit ProjectRoot in launch_ai_clip_studio.ps1 if your clone lives elsewhere."
    Read-Host 'Press Enter to exit'
    exit 1
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "`n=== AI Clip Studio ===" -ForegroundColor Cyan
Write-Host "Directory: $ProjectRoot`n"

# Prefer first NVIDIA GPU (e.g. RTX 4090) for CUDA / NVENC sessions
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:NVIDIA_TF32_OVERRIDE = '1'

# CUDA 12.9 toolkit bin (cuBLAS, nvrtc — helps local Whisper / DLL resolution)
$Cuda129Bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
if (Test-Path -LiteralPath $Cuda129Bin) {
    $env:PATH = "$Cuda129Bin;$env:PATH"
    Write-Host "PATH: prepended CUDA 12.9 bin" -ForegroundColor Green
} else {
    Write-Host "Note: CUDA 12.9 bin not found (optional): $Cuda129Bin" -ForegroundColor DarkYellow
}

# FFmpeg on PATH (WinGet / local helper)
$ffmpegBat = Join-Path $ProjectRoot 'scripts\set_ffmpeg_path_for_session.bat'
if (Test-Path -LiteralPath $ffmpegBat) {
    cmd.exe /c "`"$ffmpegBat`""
}

$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) {
    Write-Host "WARNING: ffmpeg not on PATH. Install FFmpeg or set FFMPEG_BINARY in .env" -ForegroundColor Yellow
} else {
    Write-Host "ffmpeg: $($ff.Source)" -ForegroundColor Green
    & ffmpeg.exe -hide_banner -version 2>$null | Select-Object -First 1
}

$venvActivate = Join-Path $ProjectRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path -LiteralPath $venvActivate)) {
    Write-Host "ERROR: Missing venv: $venvActivate" -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

. $venvActivate

$st = Join-Path $ProjectRoot '.venv\Scripts\streamlit.exe'
if (-not (Test-Path -LiteralPath $st)) {
    Write-Host "ERROR: streamlit not installed in .venv" -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

# Desktop shortcuts (idempotent update)
$scScript = Join-Path $ProjectRoot 'scripts\create_desktop_shortcuts.ps1'
if (Test-Path -LiteralPath $scScript) {
    & $scScript -ProjectRoot $ProjectRoot
}

Write-Host "`nOpening http://localhost:8501/ in ~2s (separate window) …" -ForegroundColor DarkGray
Start-Process cmd.exe -ArgumentList '/c', 'timeout /t 2 /nobreak >nul && start http://localhost:8501/' -WindowStyle Hidden

Write-Host "`n--- Streamlit (Ctrl+C to stop) ---`n" -ForegroundColor Cyan
Set-Location -LiteralPath $ProjectRoot
& $st run clip_studio_app.py
$code = $LASTEXITCODE

Write-Host "`nStreamlit exited with code $code" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Red' })
Read-Host 'Press Enter to close this window'
