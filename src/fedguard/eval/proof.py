"""Proof-of-compute appendix: docs/proof_of_compute.md + docs/figures/compute_timeline.png, built only from files on
disk. Every value records its source.

Sources
  * run directories (<runs>/<experiment>/[<node>/]<timestamp>_<seed>/, including _superseded/): start = meta.json
    "created"; end = timestamp written into the DONE marker (finished runs) or, for runs that never finished, the
    last modification time of any file in the directory (labelled "last file write"); GPU = meta.json versions;
    metrics = summary.json.
  * runner logs (<runs>/_queue_logs/*.log) for jobs that create no run directory (attack, explain, mc-ablation,
    mc-predict, alerts): start = timestamp in the file name, end = log file mtime; status from markers in the log.
  * GPU memory: <runs>/_gpu_log.csv (scripts/gpu_logger.py, nvidia-smi every 30 s) and per-run
    summary.json["peak_gpu_mem_mib"] (torch.cuda.max_memory_allocated) where present; plus manual nvidia-smi
    readings in docs/manual_gpu_observations.json, reported separately and labelled as manual.
  * git log --stat of the repository.
Fast-mode (smoke-test) runs are counted but listed separately; they are not project compute.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fedguard.utils.io import repo_root, runs_dir  # noqa: E402

CPU_MODELS = {"lr", "lgbm"}
LOG_COMMANDS = {"attack", "explain", "mc-ablation", "mc-predict", "alerts"}
SUCCESS_MARKERS = {
    "attack": "written to",
    "explain": "top variables",
    "mc-ablation": "T=",
    "mc-predict": "added to preds",
    "alerts": "tau_r=",
}
STATUS_COLORS = {
    "done": "#2E7D5B",
    "done (finalized)": "#6FAF8F",
    "training done; eval killed": "#D08C3A",
    "running": "#1F6E8C",
    "killed/failed": "#C8413A",
    "timeout": "#B8790F",
    "superseded (done)": "#9BB0A6",
    "superseded (killed/failed)": "#E3A39D",
    "no success marker": "#C8413A",
}


@dataclass
class Job:
    name: str
    seed: str
    start: datetime
    end: datetime
    status: str
    end_basis: str
    device: str
    source: str
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    fast: bool = False
    end_known: bool = True

    @property
    def hours(self) -> float:
        if not self.end_known:
            return float("nan")
        return max((self.end - self.start).total_seconds(), 0.0) / 3600

    @property
    def gpu(self) -> bool:
        return not self.device.startswith("CPU")


def _finalized(d: Path) -> dict[str, float] | None:
    """Timing split of a run finished by ``fedguard fl finalize`` (D37), else None."""
    try:
        return json.loads((d / "metrics.json").read_text(encoding="utf-8")).get("finalized_from_checkpoint")
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(pid)
        return "fedguard.cli" in " ".join(p.cmdline())
    except Exception:  # noqa: BLE001 - any psutil/OS error means "not our live job"
        return False


def _config_detail(cfg_path: Path) -> tuple[str, str]:
    """(short config description, model name) from a run's config.yaml."""
    import yaml

    try:
        c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return "", ""
    parts = []
    model = (c.get("model") or {}).get("name", "")
    fl = c.get("fl") or {}
    if fl:
        parts.append(f"{fl.get('algorithm')}/{fl.get('mode')}")
    p = c.get("privacy") or {}
    if p.get("enabled"):
        if p.get("noise_multiplier_override") is not None:
            parts.append(f"DP diag sigma={p['noise_multiplier_override']} (no guarantee)")
        else:
            parts.append(f"DP eps={p.get('epsilon')} {p.get('budget_rule')}")
        if p.get("reset_optimizer"):
            parts.append("opt-reset")
    elif model:
        parts.append(model)
    d = c.get("data") or {}
    if d.get("partition") and d.get("partition") != "unit":
        parts.append(f"partition={d['partition']}")
    if d.get("lookback") and d.get("lookback") != 24:
        parts.append(f"L={d['lookback']}")
    return ", ".join(str(x) for x in parts if x), model


def collect_run_dirs(root: Path) -> list[Job]:
    jobs = []
    for meta in root.rglob("meta.json"):
        d = meta.parent
        rel = d.relative_to(root)
        if rel.parts[0] in ("_queue_logs",):
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        start = datetime.fromisoformat(m["created"])
        superseded = rel.parts[0] == "_superseded"
        # run dirs are <experiment>/[<node>/]<timestamp>_<seed>; archived ones may have been renamed
        if re.fullmatch(r"\d{8}-\d{6}(-\d+)?_\d+", rel.parts[-1]):
            name = "/".join(rel.parts[:-1])
        else:
            name = "/".join(rel.parts)
        seed = str(m.get("seed", rel.parts[-1].rsplit("_", 1)[-1]))
        detail, model = _config_detail(d / "config.yaml")
        device = (
            "CPU (sklearn)" if model in CPU_MODELS else (m.get("versions", {}).get("cuda_device") or "CPU")
        )
        done = d / "DONE"
        metrics: dict[str, Any] = {}
        if done.exists():
            end = datetime.fromisoformat(done.read_text(encoding="utf-8").strip())
            status, basis = "done", "DONE marker"
            try:
                s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                metrics = {
                    k: s.get(k)
                    for k in ("test_auroc", "test_auprc", "peak_gpu_mem_mib")
                    if s.get(k) is not None
                }
            except (OSError, json.JSONDecodeError):
                pass
        elif _pid_alive(m.get("pid")):
            end, status, basis = datetime.now(), "running", "still running (now)"
        else:
            end = datetime.fromtimestamp(max(f.stat().st_mtime for f in d.rglob("*") if f.is_file()))
            status, basis = "killed/failed", "last file write"
        if superseded:
            status = f"superseded ({'done' if status == 'done' else 'killed/failed'})"
        fin = _finalized(d)
        if status == "done" and fin is not None:  # D37: training process died in evaluation; finished later
            t_end = start + timedelta(seconds=fin["training_wall_s"])
            jobs.append(Job(name, seed, start, t_end, "training done; eval killed", "events.jsonl 'done' wall time",
                            device, "run dir", detail, {}, fast=rel.parts[0].endswith("_fast")))  # fmt: skip
            start, basis = end - timedelta(seconds=fin["eval_wall_s"]), "DONE marker (fl finalize, D37)"
            status = "done (finalized)"
        jobs.append(Job(name, seed, start, end, status, basis, device, "run dir", detail, metrics,
                        fast=rel.parts[0].endswith("_fast")))  # fmt: skip
    return jobs


def collect_queue_logs(root: Path) -> list[Job]:
    jobs = []
    for log in sorted((root / "_queue_logs").glob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first = text.splitlines()[0] if text else ""
        mcmd = re.search(r"fedguard\.cli (\S+)(.*)$", first)
        if not mcmd or mcmd.group(1) not in LOG_COMMANDS:
            continue
        cmd, args = mcmd.group(1), mcmd.group(2).strip()
        start = datetime.strptime(log.name[:15], "%Y%m%d-%H%M%S")
        end = datetime.fromtimestamp(log.stat().st_mtime)
        if "TIMEOUT" in text:
            status = "timeout"
        elif SUCCESS_MARKERS[cmd] in text:
            status = "done"
        else:
            status = "no success marker"
        mseed = re.search(r"--seeds? (\S+)", args)
        seed = mseed.group(1) if mseed else "-"
        # a runner log only gains lines when the job writes output; a job killed before writing anything leaves just the
        # command line, so its end time is not recorded anywhere -> unknown, excluded from totals (never shown as 0 h)
        end_known = len([ln for ln in text.splitlines() if ln.strip()]) > 1
        basis = (
            "log file mtime"
            if end_known
            else "unknown (log holds only the command; killed before any output)"
        )
        jobs.append(Job(f"{cmd} {args}".strip(), seed, start, end, status, basis, "GPU (runner job)", "queue log",
                        fast="--fast" in args, end_known=end_known))  # fmt: skip
    return jobs


def union_hours(intervals: list[tuple[datetime, datetime]]) -> float:
    """Total length of the union of intervals, hours."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += (cur_e - cur_s).total_seconds()
    return tot / 3600


def max_concurrency(intervals: list[tuple[datetime, datetime]]) -> int:
    ev = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, dlt in ev:
        cur += dlt
        best = max(best, cur)
    return best


def gantt(jobs: list[Job], path: Path) -> None:
    js = sorted(jobs, key=lambda j: j.start)
    fig, ax = plt.subplots(figsize=(14, max(4.0, 0.2 * len(js) + 1.5)))
    for i, j in enumerate(js):
        col = STATUS_COLORS.get(j.status, "#566A7B")
        ax.barh(i, (j.end - j.start).total_seconds() / 86400, left=matplotlib.dates.date2num(j.start), height=0.75,
                color=col, edgecolor="none", hatch="///" if j.status.startswith("superseded") else None)  # fmt: skip
    ax.set_yticks(range(len(js)), [f"{j.name[:48]} s{j.seed}" for j in js], fontsize=6)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %H:%M"))
    ax.set_xlabel(
        "local time (start from meta.json / log name; end from DONE marker, last file write or log mtime)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS.values()]
    ax.legend(handles, list(STATUS_COLORS), fontsize=7, loc="lower right", frameon=False)
    ax.set_title(
        "FedGuard compute timeline: every job, including killed / timed-out / superseded runs", fontsize=10
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gpu_log_section(root: Path) -> list[str]:
    p = root / "_gpu_log.csv"
    if not p.exists():
        return ["_No automated GPU log yet (`scripts/gpu_logger.py` not started)._"]
    df = pd.read_csv(p)
    if df.empty:
        return ["_Automated GPU log exists but has no samples yet._"]
    pk = df.loc[df.memory_used_mib.idxmax()]
    return [
        f"Automated samples: {len(df)} (nvidia-smi every 30 s) from {df.timestamp.iloc[0]} to {df.timestamp.iloc[-1]}.",
        "",
        f"- **Peak dedicated GPU memory in the automated log: {int(pk.memory_used_mib)} / {int(pk.memory_total_mib)} MiB** "
        f"at {pk.timestamp} ({int(pk.fedguard_jobs)} FedGuard job(s) running, utilisation {int(pk.utilization_pct)}%).",
        f"- Mean memory used: {df.memory_used_mib.mean():.0f} MiB; mean utilisation: {df.utilization_pct.mean():.0f}%.",
        "- Samples by number of concurrent FedGuard jobs (max memory MiB): "
        + ", ".join(
            f"{int(k)} job(s): {int(v)}" for k, v in df.groupby("fedguard_jobs").memory_used_mib.max().items()
        ),
    ]


def manual_section(docs: Path) -> list[str]:
    p = docs / "manual_gpu_observations.json"
    if not p.exists():
        return []
    obs = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| time | memory used / total (MiB) | concurrent jobs | context |", "|---|---|---|---|"]
    rows += [
        f"| {o['time']} | {o['used_mib']} / {o['total_mib']} | {o['jobs']} | {o['context']} |" for o in obs
    ]
    return [
        "These readings were taken by hand with `nvidia-smi` during the session, **before automated GPU logging existed**. "
        "They are real observations but are not backed by a log file; they are listed separately for that reason.",
        "",
        *rows,
    ]


def git_log() -> str:
    try:
        return subprocess.run(["git", "log", "--stat", "--date=iso"], cwd=repo_root(), capture_output=True, text=True,
                              check=True, timeout=60).stdout  # fmt: skip
    except (subprocess.SubprocessError, OSError) as e:
        return f"(git log unavailable: {e})"


def run_proof(out_md: Path | None = None) -> dict[str, Any]:
    root = runs_dir()
    docs = repo_root() / "docs"
    fig_dir = docs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    all_jobs = collect_run_dirs(root) + collect_queue_logs(root)
    jobs = [j for j in all_jobs if not j.fast]
    fast = [j for j in all_jobs if j.fast]
    gpu_jobs = [j for j in jobs if j.gpu]
    unknown = [j for j in gpu_jobs if not j.end_known]
    iv = [(j.start, j.end) for j in gpu_jobs if j.end_known]
    job_hours = sum(j.hours for j in gpu_jobs if j.end_known)
    busy_hours = union_hours(iv)
    by_status = pd.Series([j.status for j in gpu_jobs]).value_counts().to_dict()
    hours_by_status = (
        pd.DataFrame([(j.status, j.hours) for j in gpu_jobs], columns=["status", "h"])
        .groupby("status")
        .h.sum()
    )
    gantt(jobs, fig_dir / "compute_timeline.png")
    peak_run = max(
        (j for j in jobs if "peak_gpu_mem_mib" in j.metrics),
        key=lambda j: j.metrics["peak_gpu_mem_mib"],
        default=None,
    )

    rows = []
    for j in sorted(jobs, key=lambda j: j.start):
        rows.append({
            "run": j.name, "seed": j.seed, "config": j.detail, "start": j.start.strftime("%Y-%m-%d %H:%M"),
            "end": j.end.strftime("%Y-%m-%d %H:%M"), "end basis": j.end_basis, "wall clock (h)": round(j.hours, 2) if j.end_known else "unknown",
            "status": j.status, "device": j.device,
            "test AUROC": round(j.metrics["test_auroc"], 4) if "test_auroc" in j.metrics else "",
            "test AUPRC": round(j.metrics["test_auprc"], 4) if "test_auprc" in j.metrics else "",
        })  # fmt: skip
    table = pd.DataFrame(rows)
    first, last = min(j.start for j in jobs), max(j.end for j in jobs)
    md = [
        "# Proof of compute",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `fedguard proof` from files on disk only (see the source list in "
        "`src/fedguard/eval/proof.py`). Nothing in this document is estimated or typed in by hand, except the section "
        "explicitly labelled as manual GPU readings. Regenerate with `fedguard proof`._",
        "",
        "## Summary",
        "",
        f"- Jobs recorded (excluding {len(fast)} fast-mode smoke-test jobs): **{len(jobs)}** "
        f"({len(gpu_jobs)} GPU jobs, {len(jobs) - len(gpu_jobs)} CPU-only sklearn baselines), "
        f"from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M}.",
        f"- **GPU job-hours: {job_hours:.1f} h.** This is the sum of the wall-clock durations of all GPU jobs, including failed, killed "
        "and superseded ones. Jobs that shared the GPU are each counted in full.",
        f"- **GPU busy wall-clock: {busy_hours:.1f} h.** This is the union of GPU job intervals, i.e. time during which at least one job was running.",
        f"- Maximum number of GPU jobs running at the same time: **{max_concurrency(iv)}**.",
        f"- GPU jobs with **unknown duration** (excluded from the totals above): {len(unknown)}"
        + (" (" + "; ".join(f"{j.name}, started {j.start:%Y-%m-%d %H:%M}" for j in unknown) + ")" if unknown else "") + ".",
        "- GPU jobs by status: " + ", ".join(f"{k}: {v} ({hours_by_status.get(k, 0):.1f} h)" for k, v in by_status.items()) + ".",
        "- GPU: " + ", ".join(sorted({j.device for j in gpu_jobs if j.device and not j.device.startswith('GPU (')})) + ".",
        "",
        "Failed, killed, timed-out and superseded runs are kept on purpose. They document the bugs and operational problems found and "
        "fixed during the project (docs/decisions.md D19–D38): an optimizer-state bug that invalidated the first DP results, a "
        "fixed DP learning rate that made the ε sweep collapse at large ε (superseded by the σ-scaled rule), hung jobs, watchdog "
        "false positives (including a kill of a new job whose pid Windows had reused), GPU memory overflow with several concurrent "
        "DP jobs, system-RAM exhaustion (MemoryError) caused by the OneDrive client holding ~31 GB, and a LightGBM "
        "early-stopping bug.",
        "",
        "## What this does and does not capture",
        "",
        "- **Captured:** every job that created a run directory (all training and FL runs, including killed, timed-out and "
        "archived ones), and every attack / explain / MC-ablation / mc-predict / alerts job launched through "
        "`scripts/run_experiments.py`, via its log.",
        "- **Not captured:** commands launched directly from the CLI outside the runner that create no run directory. "
        "Examples are direct `fedguard alerts` / `mc-predict` calls and two short gradient-inversion feasibility checks. "
        "Their compute is missing from the totals, so the totals are lower bounds.",
        "- **End time of runs that never finished** is their last file write, i.e. when they stopped making progress. A hung job "
        "may have held the GPU until it was killed later, so those durations are lower bounds too.",
        "- Fast-mode smoke-test jobs are excluded from the totals.",
        "",
        "## Timeline",
        "",
        "![compute timeline](figures/compute_timeline.png)",
        "",
        "## Peak GPU memory",
        "",
        "### From automated logs",
        "",
        *gpu_log_section(root),
        "",
        (f"- Highest per-run peak allocation recorded by a run itself (torch.cuda.max_memory_allocated): "
         f"{peak_run.metrics['peak_gpu_mem_mib']:.0f} MiB ({peak_run.name}, seed {peak_run.seed})."
         if peak_run else "- No run has recorded its own peak allocation yet (added to summary.json from 2026-09-27; later runs will)."),
        "",
        "### Manual readings (not logged)",
        "",
        *manual_section(docs),
        "",
        "## Every run",
        "",
        table.to_markdown(index=False) if len(table) else "_no runs_",
        "",
        "## Commit timeline (`git log --stat`)",
        "",
        "```",
        git_log().rstrip(),
        "```",
        "",
    ]  # fmt: skip
    out_md = out_md or docs / "proof_of_compute.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    return {"jobs": len(jobs), "gpu_job_hours": job_hours, "gpu_busy_hours": busy_hours, "path": str(out_md),
            "figure": str(fig_dir / "compute_timeline.png"), "max_concurrency": max_concurrency(iv)}  # fmt: skip
