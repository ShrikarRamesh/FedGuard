"""MC-Dropout T ablation: re-run MC inference with T in {5, 10, 20, 50} for one finished run, then tune alert
thresholds on validation and evaluate on test for each T. Writes mc_ablation.json into the run dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fedguard.alerts.tuning import apply, tune
from fedguard.data.windows import WindowDataset
from fedguard.eval import metrics as M
from fedguard.eval.predictions import Predictions
from fedguard.eval.run_artifacts import load_model, run_config, run_norm, run_scenario
from fedguard.train import loops
from fedguard.utils.io import write_json


def run_mc_ablation(run_dir: Path, Ts=(5, 10, 20, 50), seed: int = 0) -> dict[str, Any]:
    cfg = run_config(run_dir)
    sc = run_scenario(cfg)
    norm = run_norm(run_dir)
    device = loops.get_device("auto")
    arrs = {s: sc.arrays(None, s, norm) for s in ("val", "test")}
    dss = {s: WindowDataset(a, sc.cfg.lookback) for s, a in arrs.items()}
    model = load_model(run_dir, cfg, arrs["val"].spec.n_channels, device)
    p_det = {s: loops.predict(model, dss[s], device) for s in dss}
    out: dict[str, Any] = {"T": list(Ts), "rows": []}
    for T in Ts:
        preds = {}
        for s in dss:
            mu, sd = loops.predict_mc(model, dss[s], device, T=T, seed=seed)
            preds[s] = Predictions.from_arrays(arrs[s], p_det[s], p_mc_mean=mu, p_mc_std=sd)
        params = tune(preds["val"])
        res = apply(preds["test"], params)
        y = preds["test"].y.astype(float)
        out["rows"].append({
            "T": T, "ece_mc": M.ece(y, preds["test"].p_mc_mean.astype(float)), "auroc_mc": M.auroc(y, preds["test"].p_mc_mean),
            "mean_std": float(np.mean(preds["test"].p_mc_std)), "tau_r": params["tau_r"], "tau_s": params["tau_s"],
            "threshold_only": res["threshold_only"], "fedguard": res["fedguard"],
        })  # fmt: skip
    out["ece_deterministic"] = M.ece(preds["test"].y.astype(float), p_det["test"].astype(float))
    write_json(Path(run_dir) / "mc_ablation.json", out)
    return out
