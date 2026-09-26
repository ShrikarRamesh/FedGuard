"""Stop hung FedGuard jobs (seen twice on the shared 6 GB GPU: a job stops making progress but never exits).

Every ``--interval`` seconds: for each unfinished run dir (no DONE) whose ``meta.json`` records a live pid, if
none of its progress files (events.jsonl, metrics.csv, train.log) changed for ``--stale-min`` minutes, kill
the process (and its venv launcher parent). The run stays unfinished, so resumable queues rerun it.

    python scripts/watchdog.py [--stale-min 45] [--interval 300]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import psutil

from fedguard.utils.io import runs_dir

PROGRESS = ("events.jsonl", "metrics.csv", "train.log", "meta.json")


def check(stale_s: float) -> None:
    now = time.time()
    for meta in runs_dir().glob("**/meta.json"):
        d = meta.parent
        if (d / "DONE").exists():
            continue
        try:
            pid = json.loads(meta.read_text(encoding="utf-8")).get("pid")
        except (OSError, json.JSONDecodeError):
            continue
        if not pid or not psutil.pid_exists(pid):
            continue
        last = max((d / f).stat().st_mtime for f in PROGRESS if (d / f).exists())
        if now - last < stale_s:
            continue
        proc = psutil.Process(pid)
        if "fedguard.cli" not in " ".join(proc.cmdline()):
            continue  # pid reused by an unrelated process
        victims = [proc]
        parent = proc.parent()
        if parent is not None and "fedguard.cli" in " ".join(parent.cmdline()):
            victims.append(parent)  # Windows venv launcher
        for p in victims:
            p.kill()
        print(f"[{datetime.now():%H:%M}] killed hung job pid={pid} ({d}); no progress for {(now - last) / 60:.0f} min",
              flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-min", type=float, default=45)
    ap.add_argument("--interval", type=float, default=300)
    a = ap.parse_args()
    print(f"watchdog: stale after {a.stale_min} min, checking every {a.interval} s", flush=True)
    while True:
        check(a.stale_min * 60)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
