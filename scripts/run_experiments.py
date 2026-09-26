"""Experiment queue runner (cross-platform). Every job is a `fedguard` CLI call; completed runs are skipped by
the CLI itself (same experiment + seed + config hash), so the whole queue is resumable.

    python scripts/run_experiments.py --stage main [--workers 2] [--fast] [--dry-run]
    stages: tune | main | sweep | baselines | ablations | all

Wrappers: scripts/run_all_main.{ps1,sh}, scripts/run_ablations.{ps1,sh}.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SEEDS = [0, 1, 2]
EPS = [1.0, 2.0, 3.0, 5.0, 8.0]
NODES_4 = ["A_MICU", "A_SICU", "B_MICU", "B_SICU"]


def jobs_tune() -> list[list[str]]:
    base = ["fl", "run", "-c", "experiments/fedguard_async_nodp", "--seed", "0", "--name", "tune_async"]
    out = [base + ["-o", f"fl.alpha0={a}", "-o", "fl.staleness_lambda=0.35"] for a in (0.5, 1.0, 2.0)]
    # stage 2 (lambda) uses the alpha0 chosen from stage 1 and written into configs/fl/fedguard_async.yaml
    out += [base + ["-o", f"fl.staleness_lambda={lam}"] for lam in (0.1, 1.0)]
    return out


def jobs_tune_dp() -> list[list[str]]:
    """DP-SGD hyperparameters (logical batch, patient-epochs per round, lr) selected on VALIDATION AUPRC,
    FedAvg + uniform DP at eps = 3, seed 0 (D24). Same model and rounds as every other run."""
    base = ["fl", "run", "-c", "experiments/fedavg_dp", "--seed", "0", "--name", "tune_dp", "-o", "privacy.physical_batch_size=256"]
    grid = [(1024, 5, 5e-4), (1024, 2, 1e-3), (2048, 2, 1e-3), (1024, 5, 2e-3)]
    return [base + ["-o", f"privacy.logical_batch_size={b}", "-o", f"privacy.local_epochs={e}", "-o", f"privacy.lr={lr}"]
            for b, e, lr in grid]  # fmt: skip


def jobs_main() -> list[list[str]]:
    j: list[list[str]] = []
    for s in SEEDS:
        j.append(["train", "centralized", "--seed", str(s)])
        j.append(["train", "local", "--all-nodes", "--seed", str(s)])
        for exp in ("fedavg", "fedprox", "fedavg_dp", "fedguard", "fedguard_async_nodp"):
            j.append(["fl", "run", "-c", f"experiments/{exp}", "--seed", str(s)])
    return j


def jobs_sweep() -> list[list[str]]:
    """Privacy-utility sweep: eps in EPS for FedGuard (adaptive) and FedGuard with the uniform rule (3 seeds),
    FedAvg+DP (1 seed), plus the like-for-like no-DP reference (async, public normalisation)."""
    j: list[list[str]] = []
    for s in SEEDS:
        j.append(["fl", "run", "-c", "experiments/fedguard_async_nodp", "--seed", str(s), "--name", "sweep_nodp_public",
                  "-o", "fl.norm=public", "-o", "eval.mc_dropout=false"])  # fmt: skip
        for e in EPS:
            if e != 3.0:  # eps = 3 is the main FedGuard run (experiment "fedguard")
                j.append(["fl", "run", "-c", "experiments/fedguard", "--seed", str(s), "--name", "sweep_adaptive",
                          "-o", f"privacy.epsilon={e}", "-o", "eval.mc_dropout=false"])  # fmt: skip
            j.append(["fl", "run", "-c", "experiments/fedguard", "--seed", str(s), "--name", "sweep_uniform",
                      "-o", "privacy.budget_rule=uniform", "-o", f"privacy.epsilon={e}", "-o", "eval.mc_dropout=false"])  # fmt: skip
    for e in EPS:
        if e != 3.0:
            j.append(["fl", "run", "-c", "experiments/fedavg_dp", "--seed", "0", "--name", "sweep_fedavg_dp",
                      "-o", f"privacy.epsilon={e}"])  # fmt: skip
    return j


def jobs_baselines() -> list[list[str]]:
    j = [["train", "centralized", "--model", m, "--seed", "0"] for m in ("lr", "lgbm")]
    j += [["train", "centralized", "--model", "gru", "--seed", str(s)] for s in SEEDS]
    return j


def jobs_ablations() -> list[list[str]]:
    """Single-seed ablations (seed 0), documented in docs/decisions.md (D21)."""
    j: list[list[str]] = []
    # node count: 2 (hospital) / 4 (unit, main) / 8 (Dirichlet) with FedAvg and FedGuard-async (no DP)
    for part, extra in (("hospital", ["-o", "data.unk_policy=exclude"]), ("dirichlet", ["-o", "data.dirichlet_k=8"])):
        for exp in ("fedavg", "fedguard_async_nodp"):
            j.append(["fl", "run", "-c", f"experiments/{exp}", "--seed", "0", "--name", f"abl_nodes_{part}_{exp}",
                      "-o", f"data.partition={part}", *extra])  # fmt: skip
    # sync vs async under client dropout: B_MICU offline for a long stretch
    off = "fl.offline=[[B_MICU,300.0,1500.0]]"
    for exp in ("fedavg", "fedguard_async_nodp"):
        j.append(["fl", "run", "-c", f"experiments/{exp}", "--seed", "0", "--name", f"abl_dropout_{exp}", "-o", off])
    # sync vs async under stronger load imbalance
    for exp in ("fedavg", "fedguard_async_nodp"):
        j.append(["fl", "run", "-c", f"experiments/{exp}", "--seed", "0", "--name", f"abl_imbalance_{exp}",
                  "-o", "fl.client_speed=[1.0,1.0,1.0,5.0]"])  # fmt: skip
    # budget rules at eps = 3 (FedGuard async)
    for rule in ("inverse", "equal_noise"):
        j.append(["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", f"abl_rule_{rule}",
                  "-o", f"privacy.budget_rule={rule}", "-o", "eval.mc_dropout=false"])  # fmt: skip
    # lookback L in {12, 48} (24 is main), centralized PatchTST
    for L in (12, 48):
        j.append(["train", "centralized", "--seed", "0", "--name", f"abl_lookback{L}", "-o", f"data.lookback={L}",
                  "-o", "eval.mc_dropout=false"])  # fmt: skip
    return j


STAGES = {"tune": jobs_tune, "tune_dp": jobs_tune_dp, "main": jobs_main, "sweep": jobs_sweep, "baselines": jobs_baselines, "ablations": jobs_ablations}


def run_job(args: list[str], fast: bool, log_dir: Path) -> tuple[list[str], int, float]:
    cmd = [sys.executable, "-m", "fedguard.cli", *args, *(["--fast"] if fast else [])]
    name = "_".join(a.replace("/", "-").replace("=", "-")[:24] for a in args if not a.startswith("-"))[:120]
    log = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return args, rc, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=[*STAGES, "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--match", action="append", default=[], help="keep jobs whose command contains this text")
    ap.add_argument("--exclude", action="append", default=[], help="drop jobs whose command contains this text")
    a = ap.parse_args()
    stages = list(STAGES) if a.stage == "all" else [a.stage]
    if a.fast and "tune_dp" in stages:  # logical batches of 1024+ patients exceed the fast subset
        print("skipping stage tune_dp in --fast mode (batch sizes exceed the fast patient subset)")
        stages.remove("tune_dp")
    jobs = [j for s in stages for j in STAGES[s]()]
    text = lambda j: " ".join(j)  # noqa: E731
    if a.match:
        jobs = [j for j in jobs if any(m in text(j) for m in a.match)]
    jobs = [j for j in jobs if not any(x in text(j) for x in a.exclude)]
    from fedguard.utils.io import runs_dir

    log_dir = runs_dir() / "_queue_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs ({', '.join(stages)}), {a.workers} worker(s); logs in {log_dir}", flush=True)
    if a.dry_run:
        for j in jobs:
            print("  fedguard", " ".join(j))
        return
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_job, j, a.fast, log_dir) for j in jobs]
        for fut in as_completed(futs):
            args, rc, dt = fut.result()
            status = "ok" if rc == 0 else f"FAILED ({rc})"
            print(f"[{datetime.now():%H:%M:%S}] {status:10s} {dt / 60:6.1f} min  fedguard {' '.join(args)}", flush=True)
            if rc != 0:
                failed.append(args)
    print(f"done: {len(jobs) - len(failed)} ok, {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
