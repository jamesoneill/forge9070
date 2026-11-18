param(
    [Parameter(Mandatory=$true)][string]$Model,
    [ValidateRange(128,131072)][int]$Context = 4096,
    [ValidateRange(1,65535)][int]$Port = 8080
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot 'target\release\forge9070.exe'
if (-not (Test-Path -LiteralPath $Executable)) {
    throw 'Build first with cargo build --release --locked from the project directory.'
}
$ModelPath = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
$CompletionLimit = [Math]::Min(1024, $Context - 1)
& $Executable serve --model $ModelPath --ctx $Context --port $Port --max-tokens $CompletionLimit
if ($LASTEXITCODE -ne 0) { throw "Forge9070 exited with code $LASTEXITCODE" }

