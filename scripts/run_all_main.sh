#!/usr/bin/env bash
# All main experiments (3 seeds) + privacy sweep + baselines, resumable. Usage: scripts/run_all_main.sh [workers] [--fast]
set -uo pipefail
W=${1:-2}; shift || true
for s in tune tune_dp main baselines sweep; do python "$(dirname "$0")/run_experiments.py" --stage $s --workers "$W" "$@"; done
