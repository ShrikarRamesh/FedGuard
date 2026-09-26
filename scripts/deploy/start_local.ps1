# Start a complete Flower deployment on ONE machine: 1 SuperLink + 4 SuperNodes (one per hospital node).
# Usage:  .\scripts\deploy\start_local.ps1 [-Fast]
# Then:   .\scripts\deploy\run_flower.ps1 -Connection local-deploy [-Fast]
# Stop:   .\scripts\deploy\stop_local.ps1
param([switch]$Fast)
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Scripts = if ($env:FEDGUARD_VENV) { Join-Path $env:FEDGUARD_VENV 'Scripts' } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts' }
$Work = if ($env:FEDGUARD_RUNS_DIR) { Split-Path $env:FEDGUARD_RUNS_DIR } else { [Environment]::GetEnvironmentVariable('FEDGUARD_RUNS_DIR', 'User') | Split-Path }
$DataDir = if ($env:FEDGUARD_DATA_DIR) { $env:FEDGUARD_DATA_DIR } else { [Environment]::GetEnvironmentVariable('FEDGUARD_DATA_DIR', 'User') }
$Processed = Join-Path $DataDir ($(if ($Fast) { 'processed_fast' } else { 'processed' }))
$env:FLWR_HOME = Join-Path $Work 'flwr_home'
# SuperLink/SuperNode spawn `flower-superexec`, `flwr-serverapp`, `flwr-clientapp` by name: venv must be on PATH
$env:PATH = "$Scripts;$env:PATH"
$Logs = Join-Path $Work 'flwr_logs'
New-Item -ItemType Directory -Force $env:FLWR_HOME, $Logs | Out-Null
# Named connection for `flwr run` (Flower 1.38 reads $FLWR_HOME/config.toml)
@"
[superlink]
default = "local-deploy"

[superlink.local-deploy]
address = "127.0.0.1:9093"
insecure = true
"@ | Set-Content -Encoding ascii (Join-Path $env:FLWR_HOME 'config.toml')

$pids = @()
# Flower 1.38: the Control API is HTTP on --host/--port (fleet gRPC stays on 9092)
$p = Start-Process -FilePath "$Scripts\flower-superlink.exe" -ArgumentList '--insecure', '--host', '127.0.0.1', '--port', '9093' -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$Logs\superlink.out" -RedirectStandardError "$Logs\superlink.err"
$pids += $p.Id
Start-Sleep -Seconds 4
$nodes = 'A_MICU', 'A_SICU', 'B_MICU', 'B_SICU'
for ($i = 0; $i -lt $nodes.Count; $i++) {
    $n = $nodes[$i]
    $port = 9094 + $i
    $cfg = "client='$n' data-dir='$($Processed -replace '\\', '/')'"
    $p = Start-Process -FilePath "$Scripts\flower-supernode.exe" -PassThru -WindowStyle Hidden `
        -ArgumentList '--insecure', '--superlink', '127.0.0.1:9092', '--port', "$port", '--node-config', "`"$cfg`"" `
        -RedirectStandardOutput "$Logs\supernode_$n.out" -RedirectStandardError "$Logs\supernode_$n.err"
    $pids += $p.Id
}
$pids | Set-Content (Join-Path $Logs 'pids.txt')
Write-Host "SuperLink + 4 SuperNodes started (pids $($pids -join ', ')). Logs: $Logs"
Write-Host "FLWR_HOME=$env:FLWR_HOME  (set it in the shell that runs 'flwr run')"
