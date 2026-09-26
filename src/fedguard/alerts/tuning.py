"""Tune alert thresholds on VALIDATION predictions only, then apply them unchanged to test (D22).

Operating point:
  1. tau_r* = the threshold maximising the validation utility of threshold-only alerting.
  2. At tau_r*, tau_s* = the uncertainty gate minimising validation false alarms per 100 patient-hours
     subject to patient-level sensitivity >= sensitivity(threshold-only at tau_r*) - max_sens_drop.
     If no gate satisfies the constraint, tau_s* = inf (no gating).
Also reported: the jointly utility-optimal (tau_r, tau_s) for the gated policy, and the full grids.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedguard.alerts.metrics import evaluate_alerts
from fedguard.alerts.policy import threshold_only, uncertainty_gated
from fedguard.eval.predictions import Predictions
from fedguard.eval.utility import normalized_utility

TAU_R_GRID = np.round(np.linspace(0.05, 0.97, 47), 3)


def _risk_grid(mean: np.ndarray) -> np.ndarray:
    """Candidate risk thresholds: the fixed grid plus quantiles of the model's own VALIDATION risk (50th-99.9th
    percentile). Models with compressed outputs (e.g. DP models predicting near the 1.7% base rate, max < 0.01)
    would otherwise never alert at any fixed threshold >= 0.05 (D29)."""
    qs = np.quantile(mean, np.concatenate([np.linspace(0.50, 0.99, 50), [0.995, 0.998, 0.999]]))
    return np.unique(np.concatenate([TAU_R_GRID, np.round(qs, 6)]))


def _std_grid(std: np.ndarray, n: int = 20) -> np.ndarray:
    qs = np.quantile(std, np.linspace(0.05, 0.95, n))
    return np.unique(np.round(np.concatenate([qs, [np.inf]]), 5))


def _mc(p: Predictions) -> tuple[np.ndarray, np.ndarray]:
    if p.p_mc_mean is None or p.p_mc_std is None:
        raise ValueError("predictions have no MC-Dropout outputs; run with eval.mc_dropout=true")
    return p.p_mc_mean.astype(float), p.p_mc_std.astype(float)


def tune(val: Predictions, refractory: int = 6, max_sens_drop: float = 0.02) -> dict[str, Any]:
    mean, std = _mc(val)
    y, offs = val.y, val.offsets()
    r_grid = _risk_grid(mean)
    grid_thr = []
    for tr in r_grid:
        a = threshold_only(mean, tr)
        grid_thr.append({"tau_r": float(tr), "utility": normalized_utility(y, a.astype(np.int8), offs)})
    tau_r = max(grid_thr, key=lambda r: r["utility"])["tau_r"]
    base = evaluate_alerts(threshold_only(mean, tau_r), y, offs, refractory)
    s_grid = _std_grid(std)
    grid_gate = []
    for ts in s_grid:
        m = evaluate_alerts(uncertainty_gated(mean, std, tau_r, ts), y, offs, refractory)
        grid_gate.append({"tau_r": tau_r, "tau_s": float(ts), **m.as_dict()})
    ok = [g for g in grid_gate if g["sensitivity"] >= base.sensitivity - max_sens_drop - 1e-12]
    chosen = min(ok, key=lambda g: (g["false_per_100h"], -g["tau_s"])) if ok else {"tau_s": float("inf")}
    # jointly utility-optimal gated policy
    joint = []
    for tr in r_grid[:: max(1, len(r_grid) // 50)]:  # joint search on a thinned grid (cost)
        for ts in s_grid:
            a = uncertainty_gated(mean, std, tr, ts)
            joint.append(
                {
                    "tau_r": float(tr),
                    "tau_s": float(ts),
                    "utility": normalized_utility(y, a.astype(np.int8), offs),
                }
            )
    best_joint = max(joint, key=lambda r: r["utility"])
    return {
        "tau_r": tau_r,
        "tau_s": chosen["tau_s"],
        "utility_opt": {"tau_r": best_joint["tau_r"], "tau_s": best_joint["tau_s"]},
        "val_threshold_only": base.as_dict(),
        "grid_threshold_only": grid_thr,
        "grid_gate_at_tau_r": grid_gate,
        "grid_joint_utility": joint,
        "max_sens_drop": max_sens_drop,
        "refractory": refractory,
    }


def apply(test: Predictions, params: dict[str, Any], ambiguous: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate threshold-only and gated policies on test at the validation-chosen thresholds."""
    mean, std = _mc(test)
    y, offs, ref = test.y, test.offsets(), int(params["refractory"])
    tr, ts = params["tau_r"], params["tau_s"]
    u = params["utility_opt"]
    out = {
        "threshold_only": evaluate_alerts(threshold_only(mean, tr), y, offs, ref, ambiguous).as_dict(),
        "fedguard": evaluate_alerts(uncertainty_gated(mean, std, tr, ts), y, offs, ref, ambiguous).as_dict(),
        "fedguard_utility_opt": evaluate_alerts(uncertainty_gated(mean, std, u["tau_r"], u["tau_s"]), y, offs, ref,
                                                ambiguous).as_dict(),  # fmt: skip
    }
    for k in ("threshold_only", "fedguard"):
        out[k]["tau_r"] = tr
    out["fedguard"]["tau_s"] = ts
    out["fedguard_utility_opt"].update(u)
    return out
