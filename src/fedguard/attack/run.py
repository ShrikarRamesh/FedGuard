"""Run the gradient-inversion experiment for one victim model -> results/attack/attack.json + plots."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from fedguard.attack.gradient_inversion import VITAL_IDX, attack_condition
from fedguard.data.features import DYN
from fedguard.data.windows import WindowDataset
from fedguard.eval.run_artifacts import load_model, run_config, run_norm, run_scenario
from fedguard.privacy.accounting import calibrate_noise, planned_steps
from fedguard.train import loops
from fedguard.utils.io import write_json


def select_windows(ds: WindowDataset, n: int, seed: int) -> np.ndarray:
    """n test windows with a full 24 h history, one per patient, seeded; half from septic-labelled hours."""
    rng = np.random.default_rng(seed)
    full = np.flatnonzero(ds.hour >= ds.L - 1)
    by_patient: dict[int, list[int]] = {}
    for i in full:
        by_patient.setdefault(int(ds.patient_pos[i]), []).append(int(i))
    pats = np.array(sorted(by_patient))
    rng.shuffle(pats)
    y = ds.labels()
    pos = [p for p in pats if any(y[i] for i in by_patient[p])]
    neg = [p for p in pats if p not in set(pos)]
    chosen = pos[: n // 2] + neg[: n - min(n // 2, len(pos))]
    out = []
    for p in chosen:
        cand = [i for i in by_patient[p] if (y[i] == 1) == (p in set(pos))] or by_patient[p]
        out.append(cand[int(rng.integers(len(cand)))])
    return np.array(out)


def sigma_for(eps: float, privacy_cfg, n_patients: int) -> float:
    """Noise multiplier a FedGuard client with ``n_patients`` training patients uses for total ``eps``."""
    p = privacy_cfg
    q, steps = planned_steps(n_patients, int(p.logical_batch_size), int(p.local_epochs), int(p.r_max))
    return calibrate_noise(eps, float(p.delta), q, steps)


def run_attack(victim_dir: Path, out_dir: Path, privacy_cfg, n: int = 50, eps_list=(1.0, 3.0, 8.0), iters: int = 400,
               restarts: int = 3, seed: int = 0) -> dict[str, Any]:  # fmt: skip
    """``privacy_cfg`` = the DP settings FedGuard trains with (noise is calibrated exactly as in training)."""
    t0 = time.time()
    cfg = run_config(victim_dir)
    sc = run_scenario(cfg)
    norm = run_norm(victim_dir)
    device = loops.get_device("auto")
    arr = sc.arrays(None, "test", norm)
    ds = WindowDataset(arr, sc.cfg.lookback)
    model = load_model(victim_dir, cfg, arr.spec.n_channels, device)
    for p in model.parameters():
        p.requires_grad_(True)
    idx = select_windows(ds, n, seed)
    x, m, y = ds.gather(idx)
    windows = [(x[i], m[i], float(y[i])) for i in range(len(idx))]
    pw = (
        loops.pos_weight_from(sc.arrays(None, "train", norm).label)
        if norm != "public"
        else float(cfg.privacy.get("pos_weight", 10.0))
    )
    # the largest client needs the least noise for a given eps -> the attacker's best case among clients
    n_big = max(sc.n_train_patients(c) for c in sc.client_names)
    # clip-only is omitted: cosine gradient matching is scale-invariant, so clipping alone changes nothing
    # (verified: identical to no_dp in a 2-window check, D35)
    conditions: list[tuple[str, float | None, float, float | None]] = [("no_dp", None, 0.0, None)]
    for e in eps_list:
        conditions.append(
            (f"eps_{e:g}", float(privacy_cfg.max_grad_norm), sigma_for(e, privacy_cfg, n_big), e)
        )
    results: dict[str, Any] = {"victim_run": str(victim_dir), "n": len(idx), "iters": iters, "restarts": restarts,
                               "vitals": [DYN[i] for i in VITAL_IDX], "conditions": {}}  # fmt: skip
    # raw-unit conversion for display (value channels were standardised with the run's stats)
    if norm == "public":
        from fedguard.data.windows import NormStats

        st = NormStats.public()
    else:
        st = None
    patients_of = arr.patient_idx[ds.patient_pos[idx]]
    for name, clip, sigma, eps in conditions:
        rows = attack_condition(model, windows, pw, clip, sigma, device, iters, restarts, seed)
        rs = np.array([r["mean_r"] for r in rows], float)
        mses = np.array([r["mse"] for r in rows], float)
        rng = np.random.default_rng(seed)
        boots = [np.nanmean(rng.choice(rs, len(rs))) for _ in range(1000)]
        label_acc = float(np.mean([r["label_true"] == r["label_inferred"] for r in rows]))
        results["conditions"][name] = {
            "clip": clip, "sigma": sigma, "epsilon": eps, "mean_r": float(np.nanmean(rs)),
            "mean_r_ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
            "mse": float(np.nanmean(mses)), "label_accuracy": label_acc, "per_window": rows,
        }  # fmt: skip
    # example curves (raw units where available) for the app
    ex = []
    for k in range(min(5, len(idx))):
        pid = int(patients_of[k])
        rec = {"patient_idx": pid, "patient_id": str(sc.data.patients.loc[pid, "patient_id"]), "label": windows[k][2],
               "true": {}, "reconstructed": {}}  # fmt: skip
        for vi, v in enumerate(VITAL_IDX):
            mean_v, std_v = _stats_for(sc, st, norm, pid, v)
            rec["true"][DYN[v]] = (x[k][:, v] * std_v + mean_v).tolist()
            for name in results["conditions"]:
                xr = np.asarray(results["conditions"][name]["per_window"][k]["x_rec_vitals"])[:, vi]
                rec["reconstructed"].setdefault(name, {})[DYN[v]] = (xr * std_v + mean_v).tolist()
        ex.append(rec)
    results["examples"] = ex
    results["wall_clock_s"] = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "attack.json", results)
    return results


def _stats_for(sc, public_stats, norm: str, pid: int, v: int) -> tuple[float, float]:
    if norm == "public":
        return float(public_stats.mean[v]), float(public_stats.std[v])
    if norm == "pooled":
        s = sc.stats(None)
    else:
        s = sc.stats(sc.clients.iloc[pid])
    return float(s.mean[v]), float(s.std[v])
