# PowerShell equivalent of the Makefile (make is not installed on the Windows dev machine).
# Usage:  .\scripts\make.ps1 test      (targets: install, test, test-all, lint, format, smoke, app)
param([Parameter(Mandatory = $true)][string]$Target)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = if ($env:FEDGUARD_PY) { $env:FEDGUARD_PY } elseif (Test-Path 'C:\Users\Shrikar\.venvs\fedguard\Scripts\python.exe') { 'C:\Users\Shrikar\.venvs\fedguard\Scripts\python.exe' } else { 'python' }

function Invoke-Step([string[]]$CmdArgs) {
    Write-Host "> $Py $($CmdArgs -join ' ')" -ForegroundColor Cyan
    & $Py @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "step failed (exit $LASTEXITCODE): $($CmdArgs -join ' ')" }
}

switch ($Target) {
    'install' {
        Invoke-Step @('-m', 'pip', 'install', 'torch==2.14.0', '--index-url', 'https://download.pytorch.org/whl/cu130')
        Invoke-Step @('-m', 'pip', 'install', '-e', '.[dev]')
    }
    'test'     { Invoke-Step @('-m', 'pytest', '-q') }
    'test-all' { Invoke-Step @('-m', 'pytest', '-q', '-m', '') }
    'lint' {
        Invoke-Step @('-m', 'ruff', 'check', 'src', 'tests', 'app')
        Invoke-Step @('-m', 'black', '--check', 'src', 'tests', 'app')
    }
    'format' {
        Invoke-Step @('-m', 'ruff', 'check', '--fix', 'src', 'tests', 'app')
        Invoke-Step @('-m', 'black', 'src', 'tests', 'app')
    }
    'smoke' { Invoke-Step @('-m', 'fedguard.cli', 'info') }
    'app'   { Invoke-Step @('-m', 'streamlit', 'run', 'app/streamlit_app.py') }
    default { throw "unknown target '$Target' (install, test, test-all, lint, format, smoke, app)" }
}
