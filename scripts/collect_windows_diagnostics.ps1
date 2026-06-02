#Requires -Version 5.1
<#
.SYNOPSIS
  Collect Windows + Python + NVIDIA diagnostics for RT365 Clip Studio crash correlation.
  Output: logs/windows_diagnostics.txt
#>

$ErrorActionPreference = 'Continue'
$ProjectRoot = if ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent } else { 'C:\dev\rt365-ai-editor' }
$LogFile = Join-Path $ProjectRoot 'logs\windows_diagnostics.txt'
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

function Add-Section([string]$Title) {
    Add-Content -LiteralPath $LogFile -Value "`n======== $Title ========`n" -Encoding UTF8
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Set-Content -LiteralPath $LogFile -Value "RT365 Windows diagnostics collected $ts" -Encoding UTF8
Add-Content -LiteralPath $LogFile -Value "Project: $ProjectRoot" -Encoding UTF8

Add-Section 'Installed Python (py -0p)'
try { py -0p 2>&1 | Out-String | Add-Content $LogFile } catch { Add-Content $LogFile $_ }

Add-Section 'where python'
try { where.exe python 2>&1 | Add-Content $LogFile } catch { Add-Content $LogFile $_ }

Add-Section 'venv311 python'
$venvPy = Join-Path $ProjectRoot '.venv311\Scripts\python.exe'
if (Test-Path $venvPy) {
    & $venvPy --version 2>&1 | Add-Content $LogFile
} else {
    Add-Content $LogFile 'MISSING: .venv311'
}

Add-Section 'nvidia-smi'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    & nvidia-smi 2>&1 | Add-Content $LogFile
} else {
    Add-Content $LogFile 'nvidia-smi not found'
}

Add-Section 'Running python.exe'
Get-Process python* -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, StartTime, @{n='WS_MB';e={[math]::Round($_.WorkingSet64/1MB,1)}} |
    Format-Table -AutoSize | Out-String | Add-Content $LogFile

Add-Section 'Running ffmpeg.exe'
Get-Process ffmpeg* -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, StartTime | Format-Table -AutoSize | Out-String | Add-Content $LogFile

Add-Section 'Application Error (python.exe) last 7 days'
try {
    Get-WinEvent -FilterHashtable @{
        LogName = 'Application'
        ProviderName = 'Application Error'
        StartTime = (Get-Date).AddDays(-7)
    } -MaxEvents 40 -ErrorAction Stop |
        Where-Object { $_.Message -match 'python' } |
        Select-Object TimeCreated, Id, Message |
        Format-List | Out-String -Width 300 | Add-Content $LogFile
} catch {
    Add-Content $LogFile "Application log query failed: $_"
}

Add-Section 'System: Kernel-Power (last 7 days)'
try {
    Get-WinEvent -FilterHashtable @{
        LogName = 'System'
        ProviderName = 'Microsoft-Windows-Kernel-Power'
        StartTime = (Get-Date).AddDays(-7)
    } -MaxEvents 15 -ErrorAction Stop |
        Format-List TimeCreated, Id, Message | Out-String -Width 300 | Add-Content $LogFile
} catch { Add-Content $LogFile $_ }

Add-Section 'System: WHEA-Logger (last 7 days)'
try {
    Get-WinEvent -FilterHashtable @{
        LogName = 'System'
        ProviderName = 'Microsoft-Windows-WHEA-Logger'
        StartTime = (Get-Date).AddDays(-7)
    } -MaxEvents 15 -ErrorAction Stop |
        Format-List TimeCreated, Id, Message | Out-String -Width 300 | Add-Content $LogFile
} catch { Add-Content $LogFile $_ }

Add-Section 'System: nvlddmkm / Display (last 7 days)'
try {
    Get-WinEvent -FilterHashtable @{
        LogName = 'System'
        StartTime = (Get-Date).AddDays(-7)
    } -MaxEvents 200 -ErrorAction Stop |
        Where-Object { $_.ProviderName -match 'nvlddmkm|Display' } |
        Select-Object -First 25 TimeCreated, ProviderName, Id, Message |
        Format-List | Out-String -Width 300 | Add-Content $LogFile
} catch { Add-Content $LogFile $_ }

Add-Section 'Reliability Monitor hint'
Add-Content $LogFile 'Open: Win+R → perfmon /rel — export red X events for python.exe / ffmpeg.exe' -Encoding UTF8

Add-Section 'RT365 logs present'
@('crash_report.txt', 'environment_check.txt', 'gpu_cleanup.log', 'startup_diagnostics.txt') | ForEach-Object {
    $p = Join-Path $ProjectRoot "logs\$_"
    Add-Content $LogFile "$_ : $(if (Test-Path $p) { 'yes' } else { 'no' })" -Encoding UTF8
}

Write-Host "Wrote $LogFile" -ForegroundColor Green
