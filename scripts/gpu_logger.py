"""Append GPU memory/utilisation samples (nvidia-smi) to <runs>/_gpu_log.csv every --interval seconds.

    python scripts/gpu_logger.py [--interval 30]

Columns: timestamp (ISO, local), memory_used_mib, memory_total_mib, utilization_pct, fedguard_jobs (number of FedGuard
GPU job processes running at that moment). Used by `fedguard proof` for the peak-memory section.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedguard.utils.io import runs_dir  # noqa: E402


def sample() -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()[0]  # fmt: skip
        used, total, util = (int(float(x)) for x in out.split(","))
        return used, total, util
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
        return None


def main() -> None:
    from run_experiments import running_gpu_jobs

    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    path = runs_dir() / "_gpu_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "memory_used_mib", "memory_total_mib", "utilization_pct", "fedguard_jobs"])
        while True:
            s = sample()
            if s is not None:
                w.writerow([datetime.now().isoformat(timespec="seconds"), *s, running_gpu_jobs()])
                f.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
