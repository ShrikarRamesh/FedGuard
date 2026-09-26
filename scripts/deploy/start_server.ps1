# SERVER laptop: start the Flower SuperLink reachable on the LAN (fleet gRPC :9092, control HTTP :9093).
# Usage: .\scripts\deploy\start_server.ps1        (keep this window open; Ctrl+C stops it)
$Scripts = if ($env:FEDGUARD_VENV) { Join-Path $env:FEDGUARD_VENV 'Scripts' } else { 'C:\Users\Shrikar\.venvs\fedguard\Scripts' }
$env:PATH = "$Scripts;$env:PATH"
Write-Host "Server IP address(es) for the client laptops:" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254*' } | ForEach-Object { "  $($_.IPAddress)  ($($_.InterfaceAlias))" }
& "$Scripts\flower-superlink.exe" --insecure --host 0.0.0.0 --port 9093
