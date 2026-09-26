"""Alert + calibration evaluation of one finished run: tune on preds_val, report on preds_test -> alerts.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedguard.alerts.tuning import apply, tune  # noqa: E402
from fedguard.eval import metrics as M  # noqa: E402
from fedguard.eval.run_artifacts import load_preds  # noqa: E402
from fedguard.utils.io import write_json  # noqa: E402


def calibration(preds, n_bins: int = 15) -> dict[str, Any]:
    y = preds.y.astype(float)
    out = {"deterministic": {"ece": M.ece(y, preds.p.astype(float), n_bins), "brier": M.brier(y, preds.p.astype(float)),
                             "curve": M.reliability_curve(y, preds.p.astype(float), n_bins)}}  # fmt: skip
    if preds.p_mc_mean is not None:
        pm = preds.p_mc_mean.astype(float)
        out["mc_dropout"] = {
            "ece": M.ece(y, pm, n_bins),
            "brier": M.brier(y, pm),
            "curve": M.reliability_curve(y, pm, n_bins),
        }
    return out


def plot_reliability(cal: dict[str, Any], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#566A7B", lw=1)
    for key, col, lab in (
        ("deterministic", "#566A7B", "deterministic"),
        ("mc_dropout", "#1F6E8C", "MC-Dropout mean"),
    ):
        if key in cal:
            c = cal[key]["curve"]
            mp, fp, cnt = (np.asarray(c[k], float) for k in ("mean_pred", "frac_pos", "count"))
            ok = cnt > 0
            ax.plot(mp[ok], fp[ok], marker="o", color=col, label=f"{lab} (ECE {cal[key]['ece']:.3f})")
    ax.set_xlabel("mean predicted risk")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_alerts(
    run_dir: Path, patients: pd.DataFrame, refractory: int = 6, max_sens_drop: float = 0.02
) -> dict[str, Any]:
    val, test = load_preds(run_dir, "val"), load_preds(run_dir, "test")
    params = tune(val, refractory=refractory, max_sens_drop=max_sens_drop)
    amb_by_patient = patients["onset_ambiguous"].to_numpy().astype(bool)
    test_patients = test.patient_idx[test.offsets()[:-1]]
    res_all = apply(test, params)
    res_unamb = apply(test, params, ambiguous=amb_by_patient[test_patients])
    for k in res_all:
        res_all[k]["median_lead_h_excl_ambiguous"] = res_unamb[k]["median_lead_h"]
    cal = {"test": calibration(test), "val": calibration(val)}
    plot_reliability(cal["test"], Path(run_dir) / "reliability_test.png", Path(run_dir).parent.name)
    out = {"params": {k: v for k, v in params.items() if not k.startswith("grid")}, "test": res_all,
           "calibration": cal, "grids": {k: v for k, v in params.items() if k.startswith("grid")}}  # fmt: skip
    write_json(Path(run_dir) / "alerts.json", out)
    return out
