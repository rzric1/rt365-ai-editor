param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot
)

$ProjectRoot = $ProjectRoot.TrimEnd('\')
$desk = [Environment]::GetFolderPath('Desktop')
if ([string]::IsNullOrWhiteSpace($desk)) {
    Write-Host 'Desktop folder not resolved; skipping shortcuts.'
    return
}

$ps1 = Join-Path $ProjectRoot 'launch_ai_clip_studio.ps1'
$compressBat = Join-Path $ProjectRoot 'compress_video.bat'
$icon = Join-Path $ProjectRoot 'assets\ai_clip_studio.ico'
$pwsh = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'

if (-not (Test-Path -LiteralPath $ps1)) {
    Write-Host "Missing launcher: $ps1"
    return
}

function New-OrUpdateShortcut {
    param(
        [string]$Path,
        [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$Description,
        [string]$IconLocation
    )
    try {
        $w = New-Object -ComObject WScript.Shell
        $s = $w.CreateShortcut($Path)
        $s.TargetPath = $TargetPath
        $s.Arguments = $Arguments
        $s.WorkingDirectory = $WorkingDirectory
        $s.WindowStyle = 1
        $s.Description = $Description
        if ($IconLocation -and (Test-Path -LiteralPath $IconLocation)) {
            $s.IconLocation = "$IconLocation,0"
        }
        $s.Save()
    }
    catch {
        Write-Host "Shortcut failed ($Path): $($_.Exception.Message)"
    }
}

$argsClip = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$ps1`""
New-OrUpdateShortcut `
    -Path (Join-Path $desk 'AI Clip Studio.lnk') `
    -TargetPath $pwsh `
    -Arguments $argsClip `
    -WorkingDirectory $ProjectRoot `
    -Description 'AI Clip Studio (Streamlit)' `
    -IconLocation $icon

if (Test-Path -LiteralPath $compressBat) {
    New-OrUpdateShortcut `
        -Path (Join-Path $desk 'Compress Video for AI Clip Studio.lnk') `
        -TargetPath $compressBat `
        -Arguments '' `
        -WorkingDirectory $ProjectRoot `
        -Description 'Compress video for upload to AI Clip Studio' `
        -IconLocation $icon
}

Write-Host 'Desktop shortcuts updated: AI Clip Studio, Compress Video for AI Clip Studio' -ForegroundColor Green
