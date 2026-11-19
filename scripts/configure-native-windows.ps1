param(
    [string]$ModelsDir,
    [string]$SettingsFile,
    [string]$Executable,
    [ValidateRange(1,65535)][int]$Port = 9070,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'

# Native Forge only. This launcher neither builds the program nor downloads weights.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Executable) {
    $Executable = Join-Path $ProjectRoot 'target\release\forge-engine.exe'
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw 'Build the independent engine in a Windows-local project folder first: cargo build --release --locked -p forge-engine'
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path
if ($Executable.StartsWith('\\')) {
    throw 'Copy the project to a Windows-local folder such as C:\code\forge9070 and build there. Running generated executables from a WSL UNC path can be blocked by Windows Application Control.'
}

if (-not $SettingsFile) {
    $ConfigBase = $env:LOCALAPPDATA
    if (-not $ConfigBase) { throw 'LOCALAPPDATA is unavailable. Supply -SettingsFile with an explicit local JSON path.' }
    $SettingsFile = Join-Path $ConfigBase 'Forge9070\native-service.json'
}
$SettingsFile = [System.IO.Path]::GetFullPath($SettingsFile)
$ConfigDirectory = Split-Path -Parent $SettingsFile
if (-not (Test-Path -LiteralPath $ConfigDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $ConfigDirectory -Force | Out-Null
}

if (-not $ModelsDir -and (Test-Path -LiteralPath 'F:\' -PathType Container)) {
    $ModelsDir = 'F:\'
}
$EngineArguments = @('serve', '--backend', 'auto', '--settings-file', $SettingsFile, '--port', "$Port")
if ($ModelsDir) {
    if (-not (Test-Path -LiteralPath $ModelsDir -PathType Container)) {
        throw "The model folder does not exist: $ModelsDir"
    }
    $EngineArguments += @('--models-dir', (Resolve-Path -LiteralPath $ModelsDir).Path)
}
if (-not $NoBrowser) { $EngineArguments += '--open-ui' }

Write-Host "Forge9070 configuration: http://127.0.0.1:$Port/"
Write-Host "Settings: $SettingsFile"
Write-Host 'Choose a model folder in the UI and scan for compatible GGUF files. Existing saved settings are restored.'
& $Executable @EngineArguments
if ($LASTEXITCODE -ne 0) { throw "Forge native service exited with code $LASTEXITCODE" }
