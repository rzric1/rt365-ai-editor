#Requires -Version 5.1
<#
.SYNOPSIS
  AI Clip Studio — Streamlit launcher (Python 3.11 .venv311 only).

  Double-click or run:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\rt365-ai-editor\launch_ai_clip_studio.ps1"
#>

$ErrorActionPreference = 'Stop'

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { 'C:\dev\rt365-ai-editor' }
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    Write-Host "ERROR: Project folder not found: $ProjectRoot" -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

Set-Location -LiteralPath $ProjectRoot
$LogsDir = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-EnvLog {
    param([string]$Line)
    Add-Content -LiteralPath (Join-Path $LogsDir 'environment_check.txt') -Value $Line -Encoding UTF8
}

Write-Host ''
Write-Host '=== RT365 AI Clip Studio ===' -ForegroundColor Cyan
Write-Host "Directory: $ProjectRoot"
Write-Host ''

# Block Python 3.14 as default launcher target (Reliability Monitor crash correlation)
$py314Out = & py -3.14 --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "WARNING: Python 3.14 is installed ($py314Out). Clip Studio will NOT use it." -ForegroundColor Yellow
    Write-EnvLog 'WARN: Python 3.14 detected on system - blocked for Clip Studio'
}

# Require Python 3.11 for venv creation
try {
    $py311 = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'py -3.11 failed' }
    Write-Host "Python 3.11: $py311" -ForegroundColor Green
    Write-EnvLog "OK: $py311"
}
catch {
    Write-Host 'ERROR: Python 3.11 required. Install from https://www.python.org/downloads/release/python-3119/' -ForegroundColor Red
    Write-EnvLog 'FAIL: Python 3.11 not found'
    Read-Host 'Press Enter to exit'
    exit 1
}

$venvPath = Join-Path $ProjectRoot '.venv311'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$venvStreamlit = Join-Path $venvPath 'Scripts\streamlit.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creating .venv311 ...' -ForegroundColor Cyan
    & py -3.11 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'ERROR: venv creation failed' -ForegroundColor Red
        exit 1
    }
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
    $aiUp = Join-Path $ProjectRoot 'requirements-ai-upgrades.txt'
    if (Test-Path -LiteralPath $aiUp) {
        & $venvPython -m pip install -r $aiUp
    }
    Write-Host 'Installing CUDA PyTorch (cu121) ...' -ForegroundColor Cyan
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Write-EnvLog 'OK: created .venv311 and installed packages'
}

if (-not (Test-Path -LiteralPath $venvStreamlit)) {
    Write-Host 'ERROR: streamlit missing in .venv311. Run setup_windows.bat' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

# CUDA / FFmpeg session PATH
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$Cuda129Bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
if (Test-Path -LiteralPath $Cuda129Bin) {
    $env:PATH = "$Cuda129Bin;$env:PATH"
    Write-Host 'PATH: prepended CUDA 12.9 bin' -ForegroundColor Green
}

$ffmpegBat = Join-Path $ProjectRoot 'scripts\set_ffmpeg_path_for_session.bat'
if (Test-Path -LiteralPath $ffmpegBat) {
    cmd.exe /c "`"$ffmpegBat`""
}

# Environment validation (Python via check_environment.py — no inline Python in this .ps1)
Write-Host 'Running environment check ...' -ForegroundColor DarkGray
$envCheckScript = Join-Path $ProjectRoot 'check_environment.py'
& $venvPython $envCheckScript
if ($LASTEXITCODE -ne 0) {
    Write-Host 'ERROR: environment check failed. See logs/environment_check.txt' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

# Single-instance preflight (one-line Python -c — no here-strings)
$preflightCode = 'from clip_engine.app_lock import preflight_single_instance; import sys; ok, msg = preflight_single_instance(); print(msg); sys.exit(0 if ok else 2)'
$preflightOut = & $venvPython -c $preflightCode 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host $preflightOut -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 2
}

$scScript = Join-Path $ProjectRoot 'scripts\create_desktop_shortcuts.ps1'
if (Test-Path -LiteralPath $scScript) {
    & $scScript -ProjectRoot $ProjectRoot
}

Write-Host ''
Write-Host 'Opening http://localhost:8501/ ...' -ForegroundColor DarkGray
Start-Process cmd.exe -ArgumentList '/c', 'timeout /t 2 /nobreak >nul && start http://localhost:8501/' -WindowStyle Hidden

Write-Host ''
Write-Host '--- Streamlit (.venv311 Python 3.11) Ctrl+C to stop ---' -ForegroundColor Cyan
Write-Host ''
Set-Location -LiteralPath $ProjectRoot
try {
    & $venvStreamlit run clip_studio_app.py --server.headless true --server.port 8501
}
finally {
    $releaseCode = 'from clip_engine.app_lock import release_app_lock; release_app_lock()'
    & $venvPython -c $releaseCode 2>$null
}
$code = $LASTEXITCODE
if ($code -eq 0) {
    Write-Host "`nStreamlit exited with code $code" -ForegroundColor Green
}
else {
    Write-Host "`nStreamlit exited with code $code" -ForegroundColor Red
}
Read-Host 'Press Enter to close this window'
