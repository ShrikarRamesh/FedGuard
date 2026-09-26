# CLIENT laptop (one per hospital node): start a Flower SuperNode that trains ONLY on this node's data.
# Usage: .\scripts\deploy\start_client.ps1 -Server 192.168.1.20 -Node A_MICU -DataDir C:\fedguard\node_A_MICU
#   DataDir = output of `fedguard data export-node --node A_MICU --out <dir>` copied to this laptop.
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][ValidateSet('A_MICU', 'A_SICU', 'B_MICU', 'B_SICU')][string]$Node,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [int]$Port = 9094
)
$Scripts = if ($env:FEDGUARD_VENV) { Join-Path $env:FEDGUARD_VENV 'Scripts' } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts' }
$env:PATH = "$Scripts;$env:PATH"
if (-not (Test-Path (Join-Path $DataDir 'manifest.json'))) { throw "no processed data in $DataDir (run fedguard data export-node)" }
$cfg = "client='$Node' data-dir='$($DataDir -replace '\\', '/')'"
& "$Scripts\flower-supernode.exe" --insecure --superlink "${Server}:9092" --port $Port --node-config $cfg
