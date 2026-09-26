#!/usr/bin/env bash
# CLIENT laptop: Flower SuperNode training ONLY on this node's data.
# Usage: scripts/deploy/start_client.sh <server-ip> <A_MICU|A_SICU|B_MICU|B_SICU> <data-dir> [port]
set -euo pipefail
SERVER=$1; NODE=$2; DATA=$3; PORT=${4:-9094}
[ -f "$DATA/manifest.json" ] || { echo "no processed data in $DATA (run fedguard data export-node)"; exit 1; }
exec flower-supernode --insecure --superlink "$SERVER:9092" --port "$PORT" --node-config "client='$NODE' data-dir='$DATA'"
