"""End-to-end --fast smoke pipeline (cross-platform; used by `make smoke` and `scripts\\make.ps1 smoke`).

download check -> process -> EDA -> centralized/local training -> FL (FedAvg, FedProx, FedGuard: async + DP + MC)
-> alerts -> explanations -> attack -> report -> export (schema-validated) -> Streamlit page smoke tests on the
fast artefacts. Everything runs on the tiny deterministic subset; it proves the pipeline works, not results.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

STEPS = [
    ["data", "download"],
    ["data", "process"],
    ["data", "eda"],
    ["train", "centralized"],
    ["train", "centralized", "--model", "lgbm"],
    ["train", "local", "--node", "A_MICU"],
    ["fl", "run", "-c", "experiments/fedavg"],
    ["fl", "run", "-c", "experiments/fedprox"],
    ["fl", "run", "-c", "experiments/fedguard"],
    ["fl", "run", "-c", "experiments/fedguard_async_nodp"],
    ["mc-predict", "-e", "async_nodp"],
    ["alerts", "-e", "async_nodp"],
    ["alerts", "-e", "fedguard"],
    ["explain", "-e", "fedguard", "--seeds", "0"],
    ["explain", "-e", "async_nodp", "--seeds", "0"],
    ["attack", "-e", "fedavg"],
    ["report"],
    ["export"],
]


def main() -> None:
    t0 = time.time()
    for s in STEPS:
        cmd = [sys.executable, "-m", "fedguard.cli", *s, "--fast"]
        print(f"> fedguard {' '.join(s)} --fast", flush=True)
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            sys.exit(f"smoke step failed: fedguard {' '.join(s)} --fast")
        print(f"  ok ({time.time() - t:.0f} s)", flush=True)
    from fedguard.utils.io import out_root, runs_dir

    env = {**os.environ, "FEDGUARD_SMOKE_RESULTS_DIR": str(out_root(True)), "FEDGUARD_SMOKE_RUNS_DIR": str(runs_dir())}
    print("> pytest tests/test_app.py (pages render on the fast artefacts)", flush=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_app.py", "tests/test_export.py"], env=env)
    if r.returncode != 0:
        sys.exit("app smoke tests failed")
    print(f"SMOKE OK in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
