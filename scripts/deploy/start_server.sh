#!/usr/bin/env bash
# SERVER laptop: Flower SuperLink on the LAN (fleet gRPC :9092, control HTTP :9093). Ctrl+C stops it.
set -euo pipefail
echo "Server IP address(es):"; hostname -I 2>/dev/null || ipconfig getifaddr en0 || true
exec flower-superlink --insecure --host 0.0.0.0 --port 9093
