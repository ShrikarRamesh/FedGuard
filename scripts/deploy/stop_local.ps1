# Stop the processes started by start_local.ps1.
$Work = if ($env:FEDGUARD_RUNS_DIR) { Split-Path $env:FEDGUARD_RUNS_DIR } else { [Environment]::GetEnvironmentVariable('FEDGUARD_RUNS_DIR', 'User') | Split-Path }
$pidFile = Join-Path $Work 'flwr_logs\pids.txt'
if (Test-Path $pidFile) {
    foreach ($id in Get-Content $pidFile) { try { Stop-Process -Id ([int]$id) -Force -ErrorAction Stop } catch {} }
    Remove-Item $pidFile
}
# SuperNodes/SuperLink spawn app processes; stop any stragglers from this venv
Get-Process flwr-clientapp, flwr-serverapp, flower-supernode, flower-superlink -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host 'Flower processes stopped.'
