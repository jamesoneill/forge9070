# Windows-only contract checks for the public PowerShell startup helper.
# Uses a fake native process, so no model or GPU is required.
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Run this test on Windows'
}
$Root = Split-Path -Parent $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot 'configure-native-windows.ps1'
$TestRoot = Join-Path $env:TEMP ("nextora-launch-test-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $TestRoot | Out-Null
try {
    $MockExe = Join-Path $TestRoot 'mock-nextora.cmd'
    $Capture = Join-Path $TestRoot 'captured-args.txt'
    $Models = Join-Path $TestRoot 'models with spaces'
    New-Item -ItemType Directory -Path $Models | Out-Null
    $Profile = Join-Path $TestRoot 'new profile.json'
    $cmd = "@echo off`r`necho %* > `"$Capture`"`r`nexit /b 0"
    [System.IO.File]::WriteAllText($MockExe, $cmd, [Text.Encoding]::ASCII)
    & $Launcher -Executable $MockExe -NoBrowser -ModelsDir $Models -SettingsFile $Profile -Backend cpu -Port 9071
    if ($LASTEXITCODE -ne 0) { throw 'Fake launcher did not exit successfully' }
    $line = Get-Content -LiteralPath $Capture -Raw
    foreach ($expected in @('serve', '--backend cpu', '--settings-file', '--models-dir', '--port 9071', 'models with spaces', 'new profile.json')) {
        if (-not $line.Contains($expected)) { throw "Missing launcher argument '$expected' in: $line" }
    }
    if ($line.Contains('--open-ui')) { throw '-NoBrowser still opened the browser' }
    if (Test-Path -LiteralPath $Profile) { throw 'Launcher unexpectedly wrote/overwrote the profile' }
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $Profile))) { throw 'Profile parent directory missing' }
    $missing = Join-Path $TestRoot 'not-a-model-directory'
    $rejected = $false
    try {
        & $Launcher -Executable $MockExe -NoBrowser -ModelsDir $missing -SettingsFile $Profile | Out-Null
    } catch {
        $rejected = $_.Exception.Message.Contains('model folder does not exist')
    }
    if (-not $rejected) { throw 'Missing model directories must fail with an actionable message' }
    $rejected = $false
    try {
        & $Launcher -Executable (Join-Path $TestRoot 'missing.exe') -NoBrowser -SettingsFile $Profile | Out-Null
    } catch {
        $rejected = $_.Exception.Message.Contains('Nextora executable not found')
    }
    if (-not $rejected) { throw 'Missing executable must suggest building Nextora' }
    Write-Host 'Windows launcher argument, profile and failure-path checks passed.'
} finally {
    Remove-Item -LiteralPath $TestRoot -Force -Recurse -ErrorAction SilentlyContinue
}
