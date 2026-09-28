"""Experiment queue runner (cross-platform). Every job is a `fedguard` CLI call; completed runs are skipped by
the CLI itself (same experiment + seed + config hash), so the whole queue is resumable.

    python scripts/run_experiments.py --stage main [--workers 2] [--fast] [--dry-run]
    stages: tune | main | sweep | baselines | ablations | all

Wrappers: scripts/run_all_main.{ps1,sh}, scripts/run_ablations.{ps1,sh}.
"""

from __future__ import annotations

import argparse
import os
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
    # lambda in {0.1, 1.0} at the chosen alpha0 is run in the ablation stage (D20)
    # 4.0 added after 2.0 beat 0.5 at the edge of the original grid (D20)
    return [base + ["-o", f"fl.alpha0={a}", "-o", "fl.staleness_lambda=0.35"] for a in (0.5, 1.0, 2.0, 4.0)]


def jobs_tune_dp() -> list[list[str]]:
    """DP-SGD hyperparameters (logical batch, patient-epochs per round, lr) selected on VALIDATION AUPRC,
    FedAvg + uniform DP at eps = 3, seed 0 (D24). Same model and rounds as every other run."""
    base = ["fl", "run", "-c", "experiments/fedavg_dp", "--seed", "0", "--name", "tune_dp", "-o", "privacy.physical_batch_size=256"]
    grid = [(1024, 5, 5e-4), (1024, 2, 1e-3), (2048, 2, 1e-3), (1024, 5, 2e-3)]
    jobs = [base + ["-o", f"privacy.logical_batch_size={b}", "-o", f"privacy.local_epochs={e}", "-o", f"privacy.lr={lr}"]
            for b, e, lr in grid]  # fmt: skip
    # k windows per sampled patient, per-patient clipping (D25)
    for b, e, lr, k in [(512, 2, 1e-3, 8), (1024, 2, 1e-3, 4)]:
        jobs.append(base + ["-o", f"privacy.logical_batch_size={b}", "-o", f"privacy.local_epochs={e}",
                            "-o", f"privacy.lr={lr}", "-o", f"privacy.windows_per_patient={k}"])  # fmt: skip
    return jobs


SMALL_MODEL = ["-o", "model.d_model=64", "-o", "model.n_layers=2", "-o", "model.n_heads=4", "-o", "model.d_ff=128",
               "-o", "model.head_hidden=32"]  # fmt: skip


def jobs_tune_dp_small() -> list[list[str]]:
    """One time-boxed run (D27): same patient-level DP (FedAvg + uniform, eps = 3, seed 0, the better of the plain
    DP settings), smaller PatchTST (d_model 64, 2 layers) to cut the noise dimension."""
    return [["fl", "run", "-c", "experiments/fedavg_dp", "--seed", "0", "--name", "tune_dp_small",
             "-o", "privacy.physical_batch_size=256", "-o", "privacy.logical_batch_size=1024",
             "-o", "privacy.local_epochs=5", "-o", "privacy.lr=0.0005", *SMALL_MODEL]]  # fmt: skip


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


def jobs_sweep_ext() -> list[list[str]]:
    """Single-seed extension of the privacy sweep to large eps (D31): the planned grid (eps <= 8) showed no recovery."""
    return [["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", "sweep_adaptive_ext",
             "-o", f"privacy.epsilon={e}", "-o", "eval.mc_dropout=false"] for e in (16.0, 32.0, 64.0, 256.0)]  # fmt: skip


def jobs_dp_diag() -> list[list[str]]:
    """Diagnose the eps sweep (accuracy fell as eps grew, D32): same FedGuard DP pipeline, seed 0.
    sigma = 0 runs have NO privacy guarantee; they measure the ceiling of the DP training recipe itself."""
    base = ["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", "dp_diag", "-o", "eval.mc_dropout=false"]
    return [
        base + ["-o", "privacy.noise_multiplier_override=0.0"],
        base + ["-o", "privacy.noise_multiplier_override=0.0", "-o", "privacy.lr=0.0001"],
        base + ["-o", "privacy.epsilon=16.0", "-o", "privacy.lr=0.0001"],
        # round 2 (sigma = 0 still degrades at lr 1e-4): isolate optimizer-state carry-over and clipping bias
        base + ["-o", "privacy.noise_multiplier_override=0.0", "-o", "privacy.reset_optimizer=true"],
        base + ["-o", "privacy.noise_multiplier_override=0.0", "-o", "privacy.max_grad_norm=1000.0"],
    ]


def jobs_dp_diag_sgd() -> list[list[str]]:
    """D38: is the flat / inverted eps curve an AdamW artefact? Same FedGuard DP pipeline, seed 0, SGD with
    momentum 0.9 instead of AdamW. lr grid at eps = 8 (selection on validation AUPRC), the sigma = 0 ceiling, and
    eps = 1 to see whether utility now depends on eps. Diagnostic only; not part of the main results."""
    base = ["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", "dp_diag_sgd", "-o", "eval.mc_dropout=false",
            "-o", "privacy.optimizer=sgd"]  # fmt: skip
    j = [base + ["-o", "privacy.epsilon=8.0", "-o", f"privacy.lr={lr}"] for lr in (0.02, 0.1, 0.5)]
    j += [base + ["-o", "privacy.noise_multiplier_override=0.0", "-o", "privacy.lr=0.1"]]
    j += [base + ["-o", "privacy.epsilon=1.0", "-o", "privacy.lr=0.1"]]
    return j


def jobs_dp_lr_rule() -> list[list[str]]:
    """D38: DP-Adam behaves like DP-SGD with step lr * B / (sigma C) when noise dominates, so a fixed lr tuned at
    eps = 3 is far too aggressive at small sigma. Test a sigma-scaled lr (effective step 0.045, calibrated from the
    lr tuned at eps = 3) across eps, FedGuard seed 0."""
    base = ["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", "dp_lr_rule", "-o", "eval.mc_dropout=false",
            "-o", "privacy.lr_rule=sigma_scaled", "-o", "privacy.effective_lr=0.045"]  # fmt: skip
    return [base + ["-o", f"privacy.epsilon={e}"] for e in (3.0, 8.0, 32.0, 256.0, 1.0)]


def jobs_dp_lr_rule_floor() -> list[list[str]]:
    """D38/D39: sigma-scaled lr floored at the non-DP tuned lr (1e-4). The floor is active only where the scaled lr
    falls below 1e-4 (sigma < ~2.1: eps = 32, 256); at eps = 1, 3, 8 the lr equals dp_lr_rule's exactly."""
    base = ["fl", "run", "-c", "experiments/fedguard", "--seed", "0", "--name", "dp_lr_rule_floor", "-o", "eval.mc_dropout=false",
            "-o", "privacy.lr_rule=sigma_scaled", "-o", "privacy.effective_lr=0.045", "-o", "privacy.lr_min=0.0001"]  # fmt: skip
    return [base + ["-o", f"privacy.epsilon={e}"] for e in (32.0, 256.0)]


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
    # staleness decay lambda (alpha0 from configs/fl/fedguard_async.yaml), async without DP
    for lam in (0.1, 1.0):
        j.append(["fl", "run", "-c", "experiments/fedguard_async_nodp", "--seed", "0", "--name", "tune_async",
                  "-o", f"fl.staleness_lambda={lam}"])  # fmt: skip
    # lookback L in {12, 48} (24 is main), centralized PatchTST
    for L in (12, 48):
        j.append(["train", "centralized", "--seed", "0", "--name", f"abl_lookback{L}", "-o", f"data.lookback={L}",
                  "-o", "eval.mc_dropout=false"])  # fmt: skip
    return j


def jobs_posthoc() -> list[list[str]]:
    """Analyses of finished runs: attack on FedAvg seed 0, explanations, MC-Dropout T ablation (D30 model)."""
    return [
        ["attack", "-e", "fedavg", "--seed", "0", "--n", "30", "--iters", "300", "--restarts", "2"],
        ["explain", "-e", "async_nodp", "--seeds", "0"],
        ["explain", "-e", "fedguard", "--seeds", "0"],
        ["mc-ablation", "-e", "async_nodp", "--seeds", "0"],
    ]


STAGES = {"posthoc": jobs_posthoc, "tune": jobs_tune, "tune_dp": jobs_tune_dp, "tune_dp_small": jobs_tune_dp_small, "main": jobs_main, "sweep": jobs_sweep, "sweep_ext": jobs_sweep_ext, "dp_diag": jobs_dp_diag, "dp_diag_sgd": jobs_dp_diag_sgd, "dp_lr_rule": jobs_dp_lr_rule, "dp_lr_rule_floor": jobs_dp_lr_rule_floor, "baselines": jobs_baselines, "ablations": jobs_ablations}


JOB_TIMEOUT_S = 8 * 3600  # jobs can take 4 h+ when the GPU is shared; genuine hangs are caught by scripts/watchdog.py
GPU_COMMANDS = {"train", "fl", "attack", "mc-ablation", "mc-predict", "explain", "alerts"}


def running_gpu_jobs() -> int:
    """Number of FedGuard training/eval jobs running on this machine (any queue). A job may appear as a launcher +
    child process pair (venv shim on Windows); only processes whose parent is not itself a job are counted."""
    import psutil

    jobs = {}
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        cmd = p.info.get("cmdline") or []
        if "fedguard.cli" in cmd:
            i = cmd.index("fedguard.cli")
            if i + 1 < len(cmd) and cmd[i + 1] in GPU_COMMANDS:
                jobs[p.info["pid"]] = p.info["ppid"]
    return sum(1 for pid, ppid in jobs.items() if ppid not in jobs)


def _slot_lock() -> Path:
    from fedguard.utils.io import runs_dir

    return runs_dir() / "_slots.lock"


def acquire_slot(max_jobs: int, poll_s: float = 15.0) -> None:
    """Atomically wait for a free GPU slot: an exclusive lock file serialises "count jobs, then start one" across
    all runner processes (two runners checking at the same moment used to both launch). The caller must start its
    job and then call ``release_slot`` once the new process is visible."""
    lock = _slot_lock()
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except (FileExistsError, PermissionError):  # Windows: EACCES while another runner is deleting the lock
            try:
                if time.time() - lock.stat().st_mtime > 300:  # stale lock from a crashed runner
                    lock.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(poll_s)
            continue
        if running_gpu_jobs() < max_jobs:
            return  # keep holding the lock until the job has started
        lock.unlink(missing_ok=True)
        time.sleep(poll_s)


def _job_visible(pid: int) -> bool:
    """True once the started process (or its venv-launched child) shows ``fedguard.cli`` in its command line."""
    import psutil

    try:
        p = psutil.Process(pid)
        return any("fedguard.cli" in " ".join(q.cmdline()) for q in [p, *p.children(recursive=True)])
    except psutil.Error:
        return False


def release_slot() -> None:
    _slot_lock().unlink(missing_ok=True)


def run_job(args: list[str], fast: bool, log_dir: Path, max_jobs: int = 3) -> tuple[list[str], int, float]:
    if not fast:
        acquire_slot(max_jobs)
    cmd = [sys.executable, "-m", "fedguard.cli", *args, *(["--fast"] if fast else [])]
    name = "_".join(a.replace("/", "-").replace("=", "-")[:24] for a in args if not a.startswith("-"))[:120]
    log = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        if not fast:  # hold the slot lock until the new job is visible to other runners' job counts
            deadline = time.time() + 60
            while time.time() < deadline and proc.poll() is None and not _job_visible(proc.pid):
                time.sleep(1)
            release_slot()
        try:  # a job that hangs (e.g. after a CUDA OOM in another process) must not hold a GPU slot forever
            rc = proc.wait(timeout=JOB_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            f.write(f"\nTIMEOUT after {JOB_TIMEOUT_S} s\n")
            rc = 124
    return args, rc, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=[*STAGES, "all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gpu-jobs", type=int, default=3, help="machine-wide cap on concurrent GPU jobs (6 GB card)")
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
        futs = []
        for j in jobs:  # stagger submissions so parallel workers see each other's processes when checking slots
            futs.append(ex.submit(run_job, j, a.fast, log_dir, a.max_gpu_jobs))
            time.sleep(5 if a.workers > 1 and not a.fast else 0)
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
