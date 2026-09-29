"""Aggregate every completed run into results/summary.csv, Markdown/LaTeX tables and figures.

Only artefacts that exist are reported; anything missing is shown as "not run yet". Values are mean +- std
over seeds (sample std, ddof=1; a single seed shows "n=1").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fedguard.utils.io import out_root, read_json, runs_dir, write_json  # noqa: E402

INK, ACCENT, ALERT, MUTED, OK = "#15283A", "#1F6E8C", "#C8413A", "#566A7B", "#2E7D5B"
METHOD_ORDER = ["Local-only", "FedAvg", "FedProx", "FedAvg + DP", "FedGuard", "Centralized"]
MAIN = {  # experiment dir -> method label (main table)
    "local_patchtst": "Local-only", "fedavg": "FedAvg", "fedprox": "FedProx", "fedavg_dp": "FedAvg + DP",
    "fedguard": "FedGuard", "centralized_patchtst": "Centralized",
}  # fmt: skip
EXTRA = {"async_nodp": "Async FL (no DP)", "centralized_gru": "GRU (centralized)", "centralized_lr": "Logistic regression (centralized)",
         "centralized_lgbm": "LightGBM (centralized)"}  # fmt: skip

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150})  # fmt: skip


# ------------------------------------------------------------------------------------------ collect
def _flat(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _best_version(met: dict) -> int | None:
    """Global-model version of the best-on-validation checkpoint of an FL run (0 = the untrained model; D38)."""
    ev = [e for e in met.get("fl", {}).get("evals", []) if e.get("val_auprc") == e.get("val_auprc")]
    return int(max(ev, key=lambda e: e["val_auprc"])["version"]) if ev else None


def collect(fast: bool = False) -> pd.DataFrame:
    """One row per completed run (any experiment), with config fields and test metrics."""
    rows = []
    root = runs_dir()
    for done in root.rglob("DONE"):
        d = done.parent
        rel = d.relative_to(root).parts
        exp = rel[0]
        if exp.startswith("_") or exp.endswith("_fast") != fast:
            continue
        exp = exp.removesuffix("_fast")
        try:
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            summ = read_json(d / "summary.json")
            met = read_json(d / "metrics.json")
        except (OSError, json.JSONDecodeError):
            continue
        g = met["test"]["global"]
        row = {
            "experiment": exp, "node": rel[1] if len(rel) == 3 else None, "seed": int(cfg.get("seed", -1)), "run_dir": str(d),
            "model": cfg.get("model", {}).get("name"), "partition": cfg["data"]["partition"], "lookback": cfg["data"]["lookback"],
            "algorithm": cfg.get("fl", {}).get("algorithm", cfg.get("train", {}).get("mode")),
            "fl_mode": cfg.get("fl", {}).get("mode"), "dp": bool(cfg.get("privacy", {}).get("enabled", False)),
            "epsilon": cfg.get("privacy", {}).get("epsilon") if cfg.get("privacy", {}).get("enabled") else None,
            "budget_rule": cfg.get("privacy", {}).get("budget_rule") if cfg.get("privacy", {}).get("enabled") else None,
            "alpha0": cfg.get("fl", {}).get("alpha0"), "staleness_lambda": cfg.get("fl", {}).get("staleness_lambda"),
            "norm": met.get("norm"), "test_auroc": g["auroc"], "test_auprc": g["auprc"], "prevalence": g["prevalence"],
            "test_ece": g["ece"], "test_brier": g["brier"], "auroc_lo": g.get("ci95", {}).get("auroc", [np.nan])[0],
            "auroc_hi": g.get("ci95", {}).get("auroc", [np.nan, np.nan])[1], "auprc_lo": g.get("ci95", {}).get("auprc", [np.nan])[0],
            "auprc_hi": g.get("ci95", {}).get("auprc", [np.nan, np.nan])[1], "wall_clock_s": summ.get("wall_clock_s"),
            "sim_time": summ.get("sim_time"), "bytes_total": summ.get("bytes_total"),
            "best_val_auprc": met.get("fl", {}).get("best_val_auprc"),
            "best_version": _best_version(met),
        }  # fmt: skip
        if row["node"]:
            own = met["test"]["per_client"].get(row["node"], {})
            row.update({"own_test_auroc": own.get("auroc"), "own_test_auprc": own.get("auprc")})
        for c, e in (summ.get("eps") or {}).items():
            row[f"eps_spent.{c}"] = e
        rows.append(row)
    if not rows:  # nothing run yet: keep the schema so every table renders "not run yet"
        return pd.DataFrame(columns=EMPTY_COLUMNS)
    return pd.DataFrame(rows)


EMPTY_COLUMNS = [
    "experiment", "node", "seed", "run_dir", "model", "partition", "lookback", "algorithm", "fl_mode", "dp", "epsilon",
    "budget_rule", "alpha0", "staleness_lambda", "norm", "test_auroc", "test_auprc", "prevalence", "test_ece",
    "test_brier", "auroc_lo", "auroc_hi", "auprc_lo", "auprc_hi", "wall_clock_s", "sim_time", "bytes_total",
    "best_val_auprc", "best_version", "own_test_auroc", "own_test_auprc",
]  # fmt: skip


def ms(x: pd.Series, fmt: str = "{:.3f}") -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "not run yet"
    if len(x) == 1:
        return fmt.format(x.iloc[0]) + " (n=1)"
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1))}"


# ------------------------------------------------------------------------------------------ tables
def main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Methods x (AUROC, AUPRC, prevalence, ECE) on the global test set, mean +- std over seeds.

    Local-only: per seed, the average over the node models of each model's global-test metric; the own-node
    test metric is reported in a separate column."""
    rows = []
    for exp, label in {**MAIN, **EXTRA}.items():
        sub = df[(df.experiment == exp) & (df.partition == "unit") & (df.lookback == 24)]
        if exp in ("fedguard", "fedavg_dp"):
            sub = sub[sub.epsilon == 3.0]
        if exp == "local_patchtst" and len(sub):
            per_seed = sub.groupby("seed")[
                ["test_auroc", "test_auprc", "test_ece", "own_test_auroc", "own_test_auprc", "prevalence"]
            ].mean()
            rows.append({"method": label, "n_seeds": len(per_seed), "auroc": ms(per_seed.test_auroc), "auprc": ms(per_seed.test_auprc),
                         "own_node_auroc": ms(per_seed.own_test_auroc), "own_node_auprc": ms(per_seed.own_test_auprc),
                         "prevalence": ms(per_seed.prevalence, "{:.4f}"), "ece": ms(per_seed.test_ece),
                         "auroc_mean": per_seed.test_auroc.mean(), "auprc_mean": per_seed.test_auprc.mean(),
                         "auroc_std": per_seed.test_auroc.std(ddof=1), "auprc_std": per_seed.test_auprc.std(ddof=1)})  # fmt: skip
            continue
        rows.append({"method": label, "n_seeds": sub.seed.nunique(), "auroc": ms(sub.test_auroc), "auprc": ms(sub.test_auprc),
                     "own_node_auroc": "", "own_node_auprc": "", "prevalence": ms(sub.prevalence, "{:.4f}"), "ece": ms(sub.test_ece),
                     "auroc_mean": sub.test_auroc.mean() if len(sub) else np.nan, "auprc_mean": sub.test_auprc.mean() if len(sub) else np.nan,
                     "auroc_std": sub.test_auroc.std(ddof=1) if len(sub) > 1 else np.nan,
                     "auprc_std": sub.test_auprc.std(ddof=1) if len(sub) > 1 else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def gap_recovered(mt: pd.DataFrame, method: str = "FedGuard", key: str = "auroc_mean") -> float:
    v = mt.set_index("method")[key]
    lo, hi = v.get("Local-only", np.nan), v.get("Centralized", np.nan)
    return (
        float((v.get(method, np.nan) - lo) / (hi - lo)) if np.isfinite(hi - lo) and hi != lo else float("nan")
    )


def privacy_table(df: pd.DataFrame) -> pd.DataFrame:
    """eps x method (FedGuard adaptive / FedGuard uniform rule / FedAvg + DP) test AUROC & AUPRC."""
    spec = {"FedGuard (adaptive)": ("fedguard", "sweep_adaptive", "sweep_adaptive_ext"),
            "FedGuard (uniform rule)": ("sweep_uniform",), "FedAvg + DP (uniform)": ("fedavg_dp", "sweep_fedavg_dp")}  # fmt: skip
    rows = []
    ext = sorted(float(e) for e in df[df.experiment == "sweep_adaptive_ext"].epsilon.dropna().unique())
    for eps in [1.0, 2.0, 3.0, 5.0, 8.0, *ext]:  # large-eps extension (seed 0 only) appended when run
        row: dict[str, Any] = {"epsilon": eps}
        for label, exps in spec.items():
            sub = df[df.experiment.isin(exps) & (df.epsilon == eps) & (df.partition == "unit")]
            # D38: a run whose best checkpoint is version 0 never improved on the untrained model
            init = "" if not len(sub) or not (sub.best_version == 0).any() else f" [{int((sub.best_version == 0).sum())} at init]"
            row[f"{label} AUROC"] = ms(sub.test_auroc) + init
            row[f"{label} AUPRC"] = ms(sub.test_auprc)
            row[f"_{label}_auroc"] = sub.test_auroc.mean() if len(sub) else np.nan
            row[f"_{label}_auroc_std"] = sub.test_auroc.std(ddof=1) if len(sub) > 1 else np.nan
        rows.append(row)
    ref = df[df.experiment == "sweep_nodp_public"]
    rows.append({"epsilon": "no DP (same normalisation)", "FedGuard (adaptive) AUROC": ms(ref.test_auroc),
                 "FedGuard (adaptive) AUPRC": ms(ref.test_auprc), "_nodp_auroc": ref.test_auroc.mean() if len(ref) else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def alerts_table(exps: dict[str, str], fast: bool = False) -> tuple[pd.DataFrame, dict]:
    """Threshold-only vs uncertainty-gated at the validation-chosen operating point, mean +- std over seeds."""
    from fedguard.utils.runs import completed_runs

    rows, raw = [], {}
    for exp, label in exps.items():
        recs = []
        for s, d in sorted(completed_runs(exp + ("_fast" if fast else "")).items()):
            p = d / "alerts.json"
            if p.exists():
                recs.append((s, read_json(p)))
        raw[label] = recs
        if not recs:
            rows.append({"model": label, "policy": "-", "false_per_100h": "not run yet"})
            continue
        for pol, pl in (("threshold_only", "threshold-only"), ("fedguard", "uncertainty-gated (FedGuard)"),
                        ("fedguard_utility_opt", "gated, utility-optimal")):  # fmt: skip
            t = pd.DataFrame([r["test"][pol] for _, r in recs])
            rows.append({"model": label, "policy": pl, "n_seeds": len(recs), "false_per_100h": ms(t.false_per_100h, "{:.2f}"),
                         "sensitivity": ms(t.sensitivity), "median_lead_h": ms(t.median_lead_h, "{:.1f}"),
                         "utility": ms(t.utility), "alarms_per_100h": ms(t.alarms_per_100h, "{:.2f}"),
                         "_fa": t.false_per_100h.mean(), "_sens": t.sensitivity.mean()})  # fmt: skip
    return pd.DataFrame(rows), raw


def calibration_table(raw: dict) -> pd.DataFrame:
    rows = []
    for label, recs in raw.items():
        if not recs:
            continue
        det = pd.Series([r["calibration"]["test"]["deterministic"]["ece"] for _, r in recs])
        mc = pd.Series([r["calibration"]["test"].get("mc_dropout", {}).get("ece", np.nan) for _, r in recs])
        rows.append(
            {"model": label, "ECE deterministic": ms(det, "{:.4f}"), "ECE MC-Dropout mean": ms(mc, "{:.4f}")}
        )
    return pd.DataFrame(rows)


def ablation_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    # node count
    rows = []
    for label, exp_pairs in (("2 nodes (hospital)", "abl_nodes_hospital_"), ("4 nodes (unit, main)", None),
                             ("8 nodes (Dirichlet a=0.5)", "abl_nodes_dirichlet_")):  # fmt: skip
        for algo, main_exp in (("fedavg", "fedavg"), ("fedguard_async_nodp", "async_nodp")):
            sub = (
                df[(df.experiment == f"{exp_pairs}{algo}")]
                if exp_pairs
                else df[(df.experiment == main_exp) & (df.seed == 0)]
            )
            rows.append({"partition": label, "algorithm": "FedAvg (sync)" if algo == "fedavg" else "Async (no DP)",
                         "AUROC": ms(sub.test_auroc), "AUPRC": ms(sub.test_auprc)})  # fmt: skip
    out["node_count"] = pd.DataFrame(rows)
    # sync vs async under stress
    rows = []
    for label, pre in (("default speeds", None), ("client dropout (B_MICU offline 300-1500 s)", "abl_dropout_"),
                       ("strong imbalance (B_SICU 5x slower)", "abl_imbalance_")):  # fmt: skip
        for algo, main_exp in (("fedavg", "fedavg"), ("fedguard_async_nodp", "async_nodp")):
            sub = (
                df[df.experiment == f"{pre}{algo}"]
                if pre
                else df[(df.experiment == main_exp) & (df.seed == 0)]
            )
            rows.append({"scenario": label, "algorithm": "FedAvg (sync)" if algo == "fedavg" else "Async (no DP)",
                         "AUROC": ms(sub.test_auroc), "AUPRC": ms(sub.test_auprc),
                         "simulated time (s)": ms(sub.sim_time, "{:.0f}")})  # fmt: skip
    out["sync_async"] = pd.DataFrame(rows)
    # budget rules at eps = 3
    rows = []
    for label, sub in (("adaptive (FedGuard)", df[(df.experiment == "fedguard") & (df.epsilon == 3.0) & (df.seed == 0)]),
                       ("uniform", df[(df.experiment == "sweep_uniform") & (df.epsilon == 3.0) & (df.seed == 0)]),
                       ("inverse", df[df.experiment == "abl_rule_inverse"]), ("equal noise", df[df.experiment == "abl_rule_equal_noise"])):  # fmt: skip
        eps_cols = [c for c in df.columns if c.startswith("eps_spent.")]
        spent = ", ".join(
            f"{c.split('.', 1)[1]} {sub[c].iloc[0]:.2f}"
            for c in eps_cols
            if len(sub) and pd.notna(sub[c].iloc[0])
        )
        rows.append(
            {
                "rule": label,
                "AUROC": ms(sub.test_auroc),
                "AUPRC": ms(sub.test_auprc),
                "eps spent": spent or "not run yet",
            }
        )
    out["budget_rules"] = pd.DataFrame(rows)
    # lookback
    rows = []
    for L, sub in ((12, df[df.experiment == "abl_lookback12_patchtst"]), (24, df[(df.experiment == "centralized_patchtst") & (df.seed == 0)]),
                   (48, df[df.experiment == "abl_lookback48_patchtst"])):  # fmt: skip
        rows.append({"lookback L (h)": L, "AUROC": ms(sub.test_auroc), "AUPRC": ms(sub.test_auprc)})
    out["lookback"] = pd.DataFrame(rows)
    return out


def tuning_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df.experiment == "tune_async"].sort_values(["alpha0", "staleness_lambda"])
    return sub[["alpha0", "staleness_lambda", "best_val_auprc", "test_auroc"]].rename(
        columns={
            "best_val_auprc": "best val AUPRC (selection)",
            "test_auroc": "test AUROC (not used for selection)",
        }
    )


# ------------------------------------------------------------------------------------------ figures
def fig_methods(mt: pd.DataFrame, prevalence: float, out: Path) -> None:
    m = mt.set_index("method").reindex(METHOD_ORDER).dropna(subset=["auroc_mean"])
    if m.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        cols = [
            ACCENT if i == "FedGuard" else OK if i == "Centralized" else ALERT if i == "Local-only" else MUTED
            for i in m.index
        ]
        ax.bar(range(len(m)), m[f"{key}_mean"], yerr=m[f"{key}_std"].fillna(0), color=cols, capsize=3)
        ax.set_xticks(range(len(m)), m.index, rotation=25, ha="right")
        ax.set_ylabel(f"test {lab} (mean ± std, 3 seeds)")
        if key == "auroc":
            ax.set_ylim(0.5, max(0.9, float(np.nanmax(m.auroc_mean)) + 0.03))
        else:
            ax.axhline(prevalence, ls="--", color=INK, lw=1)
            ax.text(
                len(m) - 0.5, prevalence, f" prevalence {prevalence:.4f}", va="bottom", ha="right", fontsize=8
            )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_privacy(pt: pd.DataFrame, out: Path) -> None:
    eps_rows = pt[pt.epsilon.apply(lambda e: isinstance(e, float))]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for label, col, lw in (
        ("FedGuard (adaptive)", ACCENT, 2.5),
        ("FedGuard (uniform rule)", MUTED, 2),
        ("FedAvg + DP (uniform)", ALERT, 1.5),
    ):
        y = eps_rows[f"_{label}_auroc"].astype(float)
        e = eps_rows[f"_{label}_auroc_std"].astype(float).fillna(0)
        ok = y.notna()
        if ok.any():
            ax.errorbar(
                eps_rows.epsilon[ok], y[ok], yerr=e[ok], marker="o", color=col, lw=lw, capsize=3, label=label
            )
    ref = pt["_nodp_auroc"].dropna() if "_nodp_auroc" in pt else pd.Series(dtype=float)
    if len(ref):
        ax.axhline(float(ref.iloc[0]), ls="--", color=OK, lw=1.2, label="no DP (same normalisation)")
    if not ax.lines and not ax.containers:  # nothing run yet: no figure
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 8], ["1", "2", "3", "5", "8"])
    ax.set_xlabel("total ε per client (δ = 1e-5)")
    ax.set_ylabel("test AUROC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_sync_async(out: Path) -> None:
    from fedguard.utils.runs import completed_runs

    fig, ax = plt.subplots(figsize=(6, 3.6))
    drew = False
    for exp, col, lab in (("fedavg", MUTED, "FedAvg (sync)"), ("async_nodp", ACCENT, "Async (FedGuard mixing, no DP)"),
                          ("abl_imbalance_fedavg", ALERT, "sync, B_SICU 5x slower"),
                          ("abl_imbalance_fedguard_async_nodp", OK, "async, B_SICU 5x slower")):  # fmt: skip
        runs = completed_runs(exp)
        if 0 not in runs:
            continue
        ev = read_json(runs[0] / "metrics.json")["fl"]["evals"]
        ax.plot([e["t"] for e in ev], [e["val_auroc"] for e in ev], color=col, label=lab, lw=1.8)
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("simulated wall-clock time (s)")
    ax.set_ylabel("global validation AUROC")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_alerts(at: pd.DataFrame, out: Path) -> None:
    if "_fa" not in at:
        return
    sub = at[at.policy.isin(["threshold-only", "uncertainty-gated (FedGuard)"])].dropna(subset=["_fa"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = [f"{m}\n{p.split(' (')[0]}" for m, p in zip(sub.model, sub.policy, strict=True)]
    cols = [ACCENT if "gated" in p else MUTED for p in sub.policy]
    axes[0].bar(labels, sub._fa, color=cols)
    axes[0].set_ylabel("false alarms / 100 patient-hours")
    axes[1].bar(labels, sub._sens, color=cols)
    axes[1].set_ylabel("patient-level sensitivity")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_attack(out: Path, fast: bool = False) -> None:
    p = out_root(fast) / "attack" / "attack.json"
    if not p.exists():
        return
    a = read_json(p)
    ex = a["examples"][0]
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)
    for ax, v in zip(axes, ["HR", "MAP", "Resp"], strict=True):
        ax.plot(ex["true"][v], color=INK, lw=2, label="true")
        ax.plot(ex["reconstructed"]["no_dp"][v], color=ALERT, ls="--", label="reconstructed, no DP")
        if "eps_3" in ex["reconstructed"]:
            ax.plot(ex["reconstructed"]["eps_3"][v], color=ACCENT, ls=":", label="reconstructed, DP ε=3")
        ax.set_ylabel(v)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("hour in window")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def to_md(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if not str(c).startswith("_") and not str(c).endswith(("_mean", "_std"))]
    return df[cols].to_markdown(index=False) if len(df) else "_not run yet_"


def run_report(fast: bool = False) -> dict[str, Any]:
    out = out_root(fast)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    df = collect(fast)
    df.to_csv(out / "summary.csv", index=False)
    mt = main_table(df)
    prev = float(df[(df.experiment == "centralized_patchtst")].prevalence.mean()) if len(df) else float("nan")
    pt = privacy_table(df)
    at, raw_alerts = alerts_table({"fedguard": "FedGuard (ε=3)", "centralized_patchtst": "Centralized"}, fast)
    ct = calibration_table(raw_alerts)
    abl = ablation_tables(df)
    tt = tuning_table(df)
    tables = {
        "main": mt,
        "privacy": pt,
        "alerts": at,
        "calibration": ct,
        "async_tuning": tt,
        **{f"ablation_{k}": v for k, v in abl.items()},
    }
    md = []
    for name, t in tables.items():
        (tab_dir / f"{name}.md").write_text(to_md(t), encoding="utf-8")
        cols = [c for c in t.columns if not str(c).startswith("_") and not str(c).endswith(("_mean", "_std"))]
        if len(t):
            (tab_dir / f"{name}.tex").write_text(t[cols].to_latex(index=False, escape=True), encoding="utf-8")
        md.append(f"### {name}\n\n{to_md(t)}\n")
    fig_methods(mt, prev, fig_dir / "methods_auroc_auprc.png")
    fig_privacy(pt, fig_dir / "privacy_utility.png")
    fig_sync_async(fig_dir / "sync_vs_async.png")
    fig_alerts(at, fig_dir / "alerts.png")
    fig_attack(fig_dir / "attack_example.png", fast)
    summary = {"n_runs": int(len(df)), "gap_recovered_auroc": {m: gap_recovered(mt, m) for m in ("FedAvg", "FedProx", "FedAvg + DP", "FedGuard")},
               "prevalence": prev}  # fmt: skip
    write_json(out / "report_summary.json", summary)
    (out / "tables" / "ALL.md").write_text("\n".join(md), encoding="utf-8")
    return {"summary": summary, "tables": tables}
