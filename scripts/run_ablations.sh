#!/usr/bin/env bash
# Ablations (seed 0) + MC-Dropout T ablation, resumable. Usage: scripts/run_ablations.sh [workers] [--fast]
set -uo pipefail
W=${1:-2}; shift || true
python "$(dirname "$0")/run_experiments.py" --stage ablations --workers "$W" "$@"
fedguard mc-ablation -e fedguard --seeds 0 "$@"
