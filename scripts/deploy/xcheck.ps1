# Full-scale Flower cross-check (D19): Flower FedAvg on a local 4-SuperNode deployment with the same seed and
# config as the in-house `fedavg` seed-0 run, then score its checkpoints with the in-house evaluation.
# Usage: .\scripts\deploy\xcheck.ps1 [-Rounds 40]
param([int]$Rounds = 40)
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Py = if ($env:FEDGUARD_PY) { $env:FEDGUARD_PY } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts\python.exe' }
$Work = [Environment]::GetEnvironmentVariable('FEDGUARD_RUNS_DIR', 'User') | Split-Path
$Out = Join-Path $Work "runs\flower\xcheck_fedavg_seed0"
& (Join-Path $PSScriptRoot 'stop_local.ps1') | Out-Null
& (Join-Path $PSScriptRoot 'start_local.ps1')
Start-Sleep -Seconds 10
& (Join-Path $PSScriptRoot 'run_flower.ps1') -Rounds $Rounds -Seed 0 -OutDir $Out
& (Join-Path $PSScriptRoot 'stop_local.ps1')
& $Py -m fedguard.cli fl eval-checkpoints $Out
