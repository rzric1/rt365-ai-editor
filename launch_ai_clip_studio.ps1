#Requires -Version 5.1
<#
.SYNOPSIS
  AI Clip Studio — Streamlit launcher (Python 3.11 .venv311 only).
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

$venvPath = Join-Path $ProjectRoot '.venv311'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$venvStreamlit = Join-Path $venvPath 'Scripts\streamlit.exe'
$traceScript = Join-Path $ProjectRoot 'scripts\launcher_trace_event.py'

function Invoke-LauncherTrace {
    param([string]$Message)
    if ((Test-Path -LiteralPath $venvPython) -and (Test-Path -LiteralPath $traceScript)) {
        try {
            & $venvPython $traceScript $Message 2>$null | Out-Null
        }
        catch {
            # Trace must never block launch
        }
    }
}

function Write-EnvLog {
    param([string]$Line)
    Add-Content -LiteralPath (Join-Path $LogsDir 'environment_check.txt') -Value $Line -Encoding UTF8
}

Invoke-LauncherTrace 'launcher started'

Write-Host ''
Write-Host '=== RT365 AI Clip Studio ===' -ForegroundColor Cyan
Write-Host "Directory: $ProjectRoot"
Write-Host ''

$py314Out = & py -3.14 --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "WARNING: Python 3.14 is installed ($py314Out). Clip Studio will NOT use it." -ForegroundColor Yellow
    Write-EnvLog 'WARN: Python 3.14 detected on system - blocked for Clip Studio'
}

try {
    $py311 = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'py -3.11 failed' }
    Write-Host "Python 3.11: $py311" -ForegroundColor Green
    Write-EnvLog "OK: $py311"
}
catch {
    Write-Host 'ERROR: Python 3.11 required.' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host 'Creating .venv311 ...' -ForegroundColor Cyan
    & py -3.11 -m venv $venvPath
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
    $aiUp = Join-Path $ProjectRoot 'requirements-ai-upgrades.txt'
    if (Test-Path -LiteralPath $aiUp) {
        & $venvPython -m pip install -r $aiUp
    }
    & $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

if (-not (Test-Path -LiteralPath $venvStreamlit)) {
    Write-Host 'ERROR: streamlit missing in .venv311. Run setup_windows.bat' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$Cuda129Bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
if (Test-Path -LiteralPath $Cuda129Bin) {
    $env:PATH = "$Cuda129Bin;$env:PATH"
}

$ffmpegBat = Join-Path $ProjectRoot 'scripts\set_ffmpeg_path_for_session.bat'
if (Test-Path -LiteralPath $ffmpegBat) {
    cmd.exe /c "`"$ffmpegBat`""
}

Invoke-LauncherTrace 'environment check started'
Write-Host 'Running environment check ...' -ForegroundColor DarkGray
& $venvPython (Join-Path $ProjectRoot 'check_environment.py')
if ($LASTEXITCODE -ne 0) {
    Invoke-LauncherTrace 'environment check FAILED'
    Read-Host 'Press Enter to exit'
    exit 1
}
Invoke-LauncherTrace 'environment check passed'

$preflightCode = 'from clip_engine.app_lock import preflight_single_instance; import sys; ok, msg = preflight_single_instance(); print(msg); sys.exit(0 if ok else 2)'
$preflightOut = & $venvPython -c $preflightCode 2>&1
if ($LASTEXITCODE -ne 0) {
    Invoke-LauncherTrace 'preflight lock check FAILED'
    Write-Host $preflightOut -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 2
}
Invoke-LauncherTrace 'preflight lock check passed'

$scScript = Join-Path $ProjectRoot 'scripts\create_desktop_shortcuts.ps1'
if (Test-Path -LiteralPath $scScript) {
    & $scScript -ProjectRoot $ProjectRoot
}

Invoke-LauncherTrace 'streamlit command about to start'
Write-Host ''
Write-Host 'Opening http://localhost:8501/ ...' -ForegroundColor DarkGray
Start-Process cmd.exe -ArgumentList '/c', 'timeout /t 2 /nobreak >nul && start http://localhost:8501/' -WindowStyle Hidden

Write-Host ''
Write-Host '--- Streamlit (.venv311) — Ctrl+C to stop ---' -ForegroundColor Cyan
Write-Host ''

Set-Location -LiteralPath $ProjectRoot

# Run Streamlit in foreground; do NOT release lock until process exits (finally after streamlit only)
& $venvStreamlit run clip_studio_app.py --server.headless true --server.port 8501
$code = $LASTEXITCODE

Invoke-LauncherTrace "streamlit exited code=$code"
Write-Host "`nStreamlit exited with code $code" -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Red' })

$releaseCode = 'from clip_engine.app_lock import release_app_lock; from clip_engine.startup_trace import trace; trace("lock released by launcher"); release_app_lock()'
& $venvPython -c $releaseCode 2>$null

Read-Host 'Press Enter to close this window'
