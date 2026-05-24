$base = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
if (-not (Test-Path $base)) { exit 1 }
$dirs = Get-ChildItem -Path $base -Directory -Filter 'Gyan.FFmpeg_*' -ErrorAction SilentlyContinue
foreach ($d in $dirs) {
  $exe = Get-ChildItem -Path $d.FullName -Recurse -Filter 'ffmpeg.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($exe) {
    Write-Output $exe.DirectoryName
    exit 0
  }
}
exit 1
