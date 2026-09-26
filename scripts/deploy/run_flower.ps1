# Submit the FedGuard Flower app to a running SuperLink.
# Usage: .\scripts\deploy\run_flower.ps1 [-Connection local-deploy] [-Experiment experiments/fedavg] [-Rounds 40] [-Seed 0] [-Fast] [-OutDir <dir>]
param(
    [string]$Connection = 'local-deploy',
    [string]$Experiment = 'experiments/fedavg',
    [int]$Rounds = 40,
    [int]$Seed = 0,
    [switch]$Fast,
    [string]$OutDir = ''
)
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Scripts = if ($env:FEDGUARD_VENV) { Join-Path $env:FEDGUARD_VENV 'Scripts' } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts' }
$Work = if ($env:FEDGUARD_RUNS_DIR) { Split-Path $env:FEDGUARD_RUNS_DIR } else { [Environment]::GetEnvironmentVariable('FEDGUARD_RUNS_DIR', 'User') | Split-Path }
if (-not $env:FLWR_HOME) { $env:FLWR_HOME = Join-Path $Work 'flwr_home' }
$fastStr = if ($Fast) { 'true' } else { 'false' }
$od = $OutDir -replace '\\', '/'
$rc = "experiment='$Experiment' seed=$Seed fast=$fastStr num-server-rounds=$Rounds out-dir='$od'"
& "$Scripts\flwr.exe" run (Join-Path $Root 'src\fedguard\fl\flower_app') $Connection --run-config $rc --stream
