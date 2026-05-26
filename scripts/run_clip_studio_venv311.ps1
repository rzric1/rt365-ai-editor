#Requires -Version 5.1
<#
.SYNOPSIS
  Run AI Clip Studio with .venv311 (CUDA PyTorch + embeddings on RTX 4090).
#>

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\dev\rt365-ai-editor'
Set-Location -LiteralPath $ProjectRoot

$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'

$Cuda129Bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
if (Test-Path -LiteralPath $Cuda129Bin) {
    $env:PATH = "$Cuda129Bin;$env:PATH"
    Write-Host "PATH: prepended CUDA 12.9 bin" -ForegroundColor Green
}

$ffmpegBat = Join-Path $ProjectRoot 'scripts\set_ffmpeg_path_for_session.bat'
if (Test-Path -LiteralPath $ffmpegBat) {
    cmd.exe /c "`"$ffmpegBat`""
}

$python = Join-Path $ProjectRoot '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "ERROR: .venv311 not found. Run scripts\setup_python311_ai_env.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== AI Clip Studio (.venv311) ===" -ForegroundColor Cyan
Write-Host "Directory: $ProjectRoot`n"
& $python -m streamlit run clip_studio_app.py
exit $LASTEXITCODE
