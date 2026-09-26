"""Artefact loading for the app. Real pipeline artefacts only; synthetic data ONLY with --demo (labelled).

Artefacts (root = FEDGUARD_RESULTS_DIR or <repo>/results):
    results.json, results_full.json    `fedguard export`
    attack/attack.json                 `fedguard attack`
    tables/*.md                        `fedguard report`
Training logs: <runs>/<experiment>/<run>/events.jsonl (`fedguard fl run`), <runs>/flower/<run>/events.jsonl
(Flower deployment, live mode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from fedguard.utils.io import results_dir, runs_dir

COMMANDS = {
    "results": "fedguard export",
    "attack": "fedguard attack",
    "tables": "fedguard report",
    "events": "fedguard fl run -c experiments/fedguard",
}


def demo_mode() -> bool:
    return "--demo" in sys.argv or os.environ.get("FEDGUARD_DEMO") == "1"


def banner() -> None:
    """Persistent banner whenever synthetic demo data is shown."""
    if demo_mode():
        st.markdown('<div class="demo-banner">DEMO DATA: synthetic numbers for layout only, not results</div>',
                    unsafe_allow_html=True)  # fmt: skip


def missing(what: str, extra: str = "") -> None:
    st.warning(
        f"No {what} found in `{results_dir()}`. Run `{COMMANDS.get(what, what)}` first. {extra}", icon="⚠️"
    )


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> Any:  # mtime busts the cache when the file changes
    return _read(Path(path))


def load(name: str) -> Any | None:
    """results.json / results_full.json / attack/attack.json, or synthetic data in demo mode."""
    if demo_mode():
        return DEMO[name]()
    p = results_dir() / name
    return _load_json(str(p), p.stat().st_mtime) if p.exists() else None


def tables() -> dict[str, str]:
    d = results_dir() / "tables"
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))} if d.exists() else {}


def event_runs() -> dict[str, Path]:
    """Recorded FL runs with events.jsonl, labelled "experiment · seed N"."""
    out: dict[str, Path] = {}
    root = runs_dir()
    if not root.exists():
        return out
    for ev in sorted(root.glob("*/*/events.jsonl")):
        exp, run = ev.parent.parent.name, ev.parent.name
        if exp.startswith("_") or exp == "flower":
            continue
        if not (ev.parent / "DONE").exists():
            continue
        out[f"{exp} · seed {run.rsplit('_', 1)[-1]}"] = ev
    return out


def latest_live_run() -> Path | None:
    runs = (
        sorted((runs_dir() / "flower").glob("*/events.jsonl"), key=lambda p: p.stat().st_mtime)
        if (runs_dir() / "flower").exists()
        else []
    )
    return runs[-1] if runs else None


def read_events(path: Path) -> list[dict]:
    """Read events.jsonl (tolerates a partially written last line during live training)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


@st.cache_data(show_spinner=False)
def cached_events(path: str, mtime: float) -> list[dict]:
    return read_events(Path(path))


# ------------------------------------------------------------------------------------ demo (synthetic)
def _demo_results() -> dict:
    rng = np.random.default_rng(7)
    H = 48
    pats = []
    for name, onset in (("Demo patient 1 (synthetic)", 34), ("Demo patient 2 (synthetic)", None)):
        t = np.arange(H)
        d = 1 / (1 + np.exp(-(t - 27) / 2.5)) if onset else np.zeros(H)
        pats.append({
            "name": name, "onset_hour": onset, "hr": (82 + 28 * d + rng.normal(0, 3, H)).tolist(),
            "map": (84 - 22 * d + rng.normal(0, 3, H)).tolist(), "resp": (16 + 9 * d + rng.normal(0, 1, H)).tolist(),
            "temp": (37 + 1.5 * d + rng.normal(0, .1, H)).tolist(), "spo2": (97 - 4 * d + rng.normal(0, .5, H)).tolist(),
            "risk_mean": np.clip(0.06 + 0.8 * d + rng.normal(0, .02, H), 0, 1).tolist(),
            "risk_std": (0.05 + 0.05 * rng.random(H)).tolist(), "attr_features": ["HR", "MAP", "Resp"],
            "attr": rng.normal(0, 1, (H, 3)).tolist(),
        })  # fmt: skip
    return {"methods": [{"name": n, "auroc": a, "auprc": b} for n, a, b in
                        (("Local-only", .70, .05), ("FedAvg", .78, .08), ("FedProx", .78, .08), ("FedAvg + DP", .72, .05),
                         ("FedGuard", .74, .06), ("Centralized", .80, .10))],
            "prevalence": 0.02, "privacy": {"eps": [1, 2, 3, 5, 8], "uniform": [.6, .62, .64, .66, .68],
                                            "adaptive": [.61, .63, .65, .67, .69], "no_dp": .76},
            "alerts": {"threshold_only": {"false_per_100h": 3.0, "sensitivity": .8}, "fedguard": {"false_per_100h": 2.4, "sensitivity": .79}},
            "patients": pats}  # fmt: skip


def _demo_full() -> dict:
    return {"results": _demo_results(), "missing": [], "patients_meta": {"tau_r": 0.5, "tau_s": 0.08, "refractory": 6},
            "main_table": [], "privacy_table": [], "alerts_per_seed": [], "per_run": []}  # fmt: skip


def _demo_attack() -> dict:
    rng = np.random.default_rng(3)
    t = np.arange(24)
    true = {
        "HR": (90 + 10 * np.sin(t / 4)).tolist(),
        "MAP": (75 - 5 * np.cos(t / 5)).tolist(),
        "Resp": (18 + 2 * np.sin(t / 3)).tolist(),
    }
    rec = {
        c: {k: (np.array(v) + rng.normal(0, s, 24)).tolist() for k, v in true.items()}
        for c, s in (("no_dp", 2), ("eps_3", 15))
    }
    return {"n": 2, "vitals": list(true), "conditions": {c: {"sigma": s, "epsilon": e, "mean_r": r, "mean_r_ci95": [r - .1, r + .1],
                                                            "mse": 1.0, "label_accuracy": 1.0}
                                                        for c, r, s, e in (("no_dp", .9, 0.0, None), ("eps_3", .05, 1.5, 3.0))},
            "examples": [{"patient_id": "demo", "label": 0, "true": true, "reconstructed": rec}]}  # fmt: skip


DEMO = {"results.json": _demo_results, "results_full.json": _demo_full, "attack/attack.json": _demo_attack}
