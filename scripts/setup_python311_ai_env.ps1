#Requires -Version 5.1
<#
.SYNOPSIS
  Side-by-side Python 3.11 venv with CUDA-enabled PyTorch for RTX 4090 embeddings.

  Does NOT modify or delete the existing .venv.
#>

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\dev\rt365-ai-editor'
Set-Location -LiteralPath $ProjectRoot

Write-Host ''
Write-Host '=== RT365 AI Clip Studio - Python 3.11 + CUDA PyTorch setup ===' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host ''

Write-Host 'Checking Python 3.11...' -ForegroundColor DarkGray
try {
    $pyVer = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'py -3.11 failed' }
    Write-Host "Found: $pyVer" -ForegroundColor Green
} catch {
    Write-Host 'ERROR: Python 3.11 not found. Install from https://www.python.org/downloads/ and enable py launcher.' -ForegroundColor Red
    exit 1
}

$venvPath = Join-Path $ProjectRoot '.venv311'
if (-not (Test-Path -LiteralPath $venvPath)) {
    Write-Host 'Creating venv at .venv311 ...' -ForegroundColor Cyan
    & py -3.11 -m venv $venvPath
} else {
    Write-Host 'Using existing .venv311' -ForegroundColor Green
}

$python = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "ERROR: venv python missing: $python" -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'Upgrading pip...' -ForegroundColor DarkGray
& $python -m pip install --upgrade pip

Write-Host ''
Write-Host 'Installing requirements.txt ...' -ForegroundColor Cyan
& $python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')

$aiUpgrades = Join-Path $ProjectRoot 'requirements-ai-upgrades.txt'
if (Test-Path -LiteralPath $aiUpgrades) {
    Write-Host 'Installing requirements-ai-upgrades.txt ...' -ForegroundColor Cyan
    & $python -m pip install -r $aiUpgrades
}

Write-Host ''
Write-Host 'Installing CUDA PyTorch (cu121 wheels for Python 3.11) ...' -ForegroundColor Cyan
& $python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host ''
Write-Host 'Verifying torch CUDA...' -ForegroundColor DarkGray
$checkScript = Join-Path $env:TEMP 'rt365_torch_check.py'
$torchCheckPy = @'
import sys
try:
    import torch
    print("python", sys.version)
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device_name", torch.cuda.get_device_name(0))
    else:
        print("cuda_device_name", "CPU")
except Exception as exc:
    print("torch_check_error", repr(exc))
'@
$torchCheckPy | Set-Content -Path $checkScript -Encoding UTF8
try {
    & $python $checkScript
} finally {
    Remove-Item -LiteralPath $checkScript -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '=== Done ===' -ForegroundColor Green
Write-Host 'Launch Clip Studio with CUDA embeddings:'
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\run_clip_studio_venv311.ps1'
Write-Host ''
Write-Host 'Your original .venv is unchanged.'
Write-Host ''
