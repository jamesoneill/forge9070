<#
.SYNOPSIS
Build (optionally) and launch Nextora's local configuration UI on Windows.
.EXAMPLE
.\scripts\configure-native-windows.ps1 -Build
.EXAMPLE
.\scripts\configure-native-windows.ps1 -ModelsDir 'F:\Models'
.EXAMPLE
.\scripts\configure-native-windows.ps1 -Build -Backend cpu -NoBrowser
#>
[CmdletBinding()]
param(
    [string]$ModelsDir,
    [string]$SettingsFile,
    [string]$Executable,
    [ValidateRange(1,65535)][int]$Port = 9070,
    [ValidateSet('auto','cpu','vulkan','hybrid')][string]$Backend = 'auto',
    [switch]$Build,
    [switch]$DebugBuild,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    throw 'This launcher is for Windows PowerShell. In WSL, see docs/WINDOWS.md for the Windows-host or Linux alternatives.'
}
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Run in Windows PowerShell or PowerShell 7 on Windows; use configure-native-wsl.sh on Linux.'
}
if ($Build -or $DebugBuild) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Cargo is unavailable. Install Rust with the MSVC toolchain from https://rustup.rs and reopen PowerShell.'
    }
    Push-Location $ProjectRoot
    try {
        $BuildArgs = @('build', '--locked', '-p', 'nextora-engine')
        if (-not $DebugBuild) { $BuildArgs += '--release' }
        & cargo @BuildArgs
        if ($LASTEXITCODE -ne 0) {
            throw 'Cargo build failed. Check the Rust MSVC target and Visual Studio 2022 C++ Build Tools (Desktop development with C++ and Windows SDK).'
        }
    } finally {
        Pop-Location
    }
}
if (-not $Executable) {
    $Profile = if ($DebugBuild) { 'debug' } else { 'release' }
    $Executable = Join-Path $ProjectRoot "target\$Profile\nextora.exe"
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Nextora executable not found: $Executable. From the repository root run: .\scripts\configure-native-windows.ps1 -Build"
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path
if ($Executable.StartsWith('\\')) {
    throw 'Windows build outputs must be on a local Windows drive (for example C:\code\nextora), not a WSL network share.'
}
if (-not $SettingsFile) {
    $ConfigBase = $env:LOCALAPPDATA
    if (-not $ConfigBase) { throw 'LOCALAPPDATA is unavailable; provide -SettingsFile as a local .json file.' }
    $SettingsFile = Join-Path $ConfigBase 'Nextora\native-service.json'
}
$SettingsFile = [System.IO.Path]::GetFullPath($SettingsFile)
if ([System.IO.Path]::GetExtension($SettingsFile) -ine '.json') {
    throw "SettingsFile must end with .json: $SettingsFile"
}
$ConfigDirectory = Split-Path -Parent $SettingsFile
if (-not (Test-Path -LiteralPath $ConfigDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $ConfigDirectory -Force | Out-Null
}
if ($ModelsDir) {
    if (-not (Test-Path -LiteralPath $ModelsDir -PathType Container)) {
        throw "The model folder does not exist: $ModelsDir. Omit -ModelsDir to choose a folder in the UI."
    }
    $ModelsDir = (Resolve-Path -LiteralPath $ModelsDir).Path
}
$EngineArguments = @('serve', '--backend', $Backend, '--settings-file', $SettingsFile, '--port', "$Port")
if ($ModelsDir) { $EngineArguments += @('--models-dir', $ModelsDir) }
if (-not $NoBrowser) { $EngineArguments += '--open-ui' }
Write-Host "Nextora UI: http://127.0.0.1:$Port/"
Write-Host "Settings profile: $SettingsFile"
if (Test-Path -LiteralPath $SettingsFile -PathType Leaf) {
    Write-Warning 'A saved profile exists and overrides the startup model/backend settings. To troubleshoot, use a new -SettingsFile path instead of deleting the existing profile.'
}
Write-Host 'Keep this terminal open while using the UI. A model is not required to start.'
Write-Host "Launching $Executable"
& $Executable @EngineArguments
if ($LASTEXITCODE -ne 0) { throw "Nextora service exited with code $LASTEXITCODE. Read the error printed above." }
