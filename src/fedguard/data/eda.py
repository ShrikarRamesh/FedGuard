"""Exploratory data analysis: per-node summary tables and plots (CSV + PNG into results/eda/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedguard.config import DataConfig  # noqa: E402
from fedguard.data.features import DYN, VITALS  # noqa: E402
from fedguard.data.physionet2019 import assign_clients  # noqa: E402
from fedguard.data.windows import ProcessedData  # noqa: E402
from fedguard.utils.io import write_json  # noqa: E402

NODE_COLORS = {"A_MICU": "#2E7D7A", "A_SICU": "#4C5FB8", "B_MICU": "#B7791F", "B_SICU": "#B0487A"}
INK, MUTED = "#15283A", "#566A7B"


def node_summary(data: ProcessedData, group: pd.Series) -> pd.DataFrame:
    """Per-group: patients, septic patients/rate, rows, positive row rate, LOS stats, split counts."""
    p = data.patients.assign(group=group.to_numpy()).dropna(subset=["group"])
    label = np.asarray(data.label)
    pos_rows = np.add.reduceat(label.astype(np.int64), data.offsets[:-1]) if len(label) else np.zeros(0)
    p = p.assign(pos_rows=pos_rows[p.index])
    rows = [_summary_row(g, d) for g, d in p.groupby("group", sort=True)]
    rows.append(_summary_row("ALL", p))
    return pd.DataFrame(rows)


def _summary_row(name: str, d: pd.DataFrame) -> dict:
    return {
        "node": name,
        "patients": len(d),
        "septic_patients": int(d["ever_septic"].sum()),
        "septic_patient_rate": d["ever_septic"].mean(),
        "rows": int(d["n_rows"].sum()),
        "positive_rows": int(d["pos_rows"].sum()),
        "positive_row_rate": d["pos_rows"].sum() / d["n_rows"].sum(),
        "los_median_h": d["n_rows"].median(),
        "los_p10_h": d["n_rows"].quantile(0.10),
        "los_p90_h": d["n_rows"].quantile(0.90),
        "los_max_h": int(d["n_rows"].max()),
        "train_patients": int((d["split"] == 0).sum()),
        "val_patients": int((d["split"] == 1).sum()),
        "test_patients": int((d["split"] == 2).sum()),
        "onset_ambiguous": int(d["onset_ambiguous"].sum()),
    }


def missingness(data: ProcessedData, group: pd.Series) -> pd.DataFrame:
    """Fraction of patient-hours where each variable was NOT measured, per group (rows=variables)."""
    measured = ~np.isnan(np.asarray(data.values_raw))
    out = {}
    for g in sorted(group.dropna().unique()):
        idx = np.flatnonzero((group == g).to_numpy())
        rows = data.rows_of(idx)
        out[g] = 1.0 - measured[rows].mean(0)
    out["ALL"] = 1.0 - measured[data.rows_of(np.flatnonzero(group.notna().to_numpy()))].mean(0)
    return pd.DataFrame(out, index=DYN)


def _plot_prevalence(summary: pd.DataFrame, path: Path) -> None:
    s = summary[summary["node"] != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = [NODE_COLORS.get(n, MUTED) for n in s["node"]]
    axes[0].bar(s["node"], 100 * s["septic_patient_rate"], color=cols)
    axes[0].set_ylabel("% patients ever septic")
    axes[1].bar(s["node"], 100 * s["positive_row_rate"], color=cols)
    axes[1].set_ylabel("% patient-hours labelled positive")
    for ax, key in zip(axes, ["septic_patient_rate", "positive_row_rate"], strict=True):
        for i, v in enumerate(s[key]):
            ax.text(i, 100 * v, f"{100 * v:.2f}%", ha="center", va="bottom", fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sepsis prevalence per node (PhysioNet 2019, measured)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_los(data: ProcessedData, group: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 340, 6)
    for g in sorted(group.dropna().unique()):
        los = data.patients.loc[(group == g).to_numpy(), "n_rows"]
        ax.hist(
            los, bins=bins, histtype="step", lw=1.8, color=NODE_COLORS.get(g, MUTED), label=g, density=True
        )
    ax.set_xlabel("ICU length of stay in the record (hours)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_missingness(miss: pd.DataFrame, path: Path) -> None:
    m = miss.drop(columns=["ALL"])
    fig, ax = plt.subplots(figsize=(5.5, 9))
    im = ax.imshow(100 * m.to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=100)
    ax.set_yticks(range(len(m.index)), m.index, fontsize=8)
    ax.set_xticks(range(len(m.columns)), m.columns, fontsize=9)
    for i, v in enumerate(m.index):
        if v in VITALS:
            ax.get_yticklabels()[i].set_fontweight("bold")
    fig.colorbar(im, ax=ax, label="% of patient-hours not measured")
    ax.set_title("Missingness per variable and node", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(data: ProcessedData, cfg: DataConfig, out_dir: Path) -> dict:
    """Write all EDA artefacts to ``out_dir`` and return a JSON-able summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = data.patients["stratum"]
    by_stratum = node_summary(data, strata)
    by_stratum.to_csv(out_dir / "summary_by_stratum.csv", index=False)

    policies = {}
    for part, pol in [("unit", "exclude"), ("unit", "separate"), ("hospital", "merge_into_hospital")]:
        c = cfg.model_copy(update={"partition": part, "unk_policy": pol})
        s = node_summary(data, assign_clients(data.patients, c))
        s.to_csv(out_dir / f"summary_{part}_{pol}.csv", index=False)
        policies[f"{part}/{pol}"] = s.to_dict(orient="records")

    main = node_summary(data, assign_clients(data.patients, cfg))
    miss = missingness(data, assign_clients(data.patients, cfg))
    miss.to_csv(out_dir / "missingness.csv")
    _plot_prevalence(main, out_dir / "prevalence.png")
    _plot_los(data, assign_clients(data.patients, cfg), out_dir / "los.png")
    _plot_missingness(miss, out_dir / "missingness.png")
    summary = {
        "main_partition": f"{cfg.partition}/{cfg.unk_policy}",
        "main": main.to_dict(orient="records"),
        "by_stratum": by_stratum.to_dict(orient="records"),
        "policies": policies,
        "missingness_all_mean": float(miss["ALL"].mean()),
    }
    write_json(out_dir / "eda_summary.json", summary)
    return summary
