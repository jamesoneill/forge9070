param([switch]$DebugBuild)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Run this script in native Windows PowerShell. For native Linux use build-linux.sh.'
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'Install Rust with the MSVC toolchain from https://rustup.rs, then open a new Developer PowerShell.'
}
Push-Location $ProjectRoot
try {
    $BuildArgs = @('build', '--locked', '-p', 'forge-engine', '-p', 'forge9070')
    if (-not $DebugBuild) { $BuildArgs += '--release' }
    & cargo @BuildArgs
    if ($LASTEXITCODE -ne 0) { throw 'Cargo build failed. Confirm the Visual Studio C++ Build Tools and Windows SDK are installed.' }
    $Profile = if ($DebugBuild) { 'debug' } else { 'release' }
    $Executable = Join-Path $ProjectRoot "target\$Profile\forgeai.exe"
    & $Executable doctor
    if ($LASTEXITCODE -ne 0) { throw 'The hardware diagnostic command failed.' }
    Write-Host "Built $Executable"
    Write-Host 'ForgeAI native Rust runtime. RustingStrata credited as historical inspiration.'
} finally {
    Pop-Location
}

