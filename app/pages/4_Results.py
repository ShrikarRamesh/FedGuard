"""4 · Results: every number comes from results_full.json / results/tables (produced by the pipeline)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import data  # noqa: E402
import theme  # noqa: E402

theme.apply("Results")
data.banner()
st.title("How close does FedGuard get to pooling all the data?")
st.caption(
    "Local-only is the floor; centralized (pooling patients across hospitals, which is not allowed) is the ceiling."
)

res = data.load("results.json")
full = data.load("results_full.json")
if res is None or full is None:
    data.missing("results")
    st.stop()
if full.get("missing"):
    st.info("Not run yet: " + ", ".join(full["missing"]))

mt = {r["method"]: r for r in full.get("main_table", [])}
order = ["Local-only", "FedAvg", "FedProx", "FedAvg + DP", "FedGuard", "Centralized"]
colors = {"FedGuard": theme.ACCENT, "Centralized": theme.OK, "Local-only": theme.ALERT}
c1, c2 = st.columns(2)
for col, key, title in ((c1, "auroc", "AUROC"), (c2, "auprc", "AUPRC")):
    with col:
        names = [m["name"] for m in res["methods"] if m[key] is not None]
        vals = [m[key] for m in res["methods"] if m[key] is not None]
        err = [mt.get(n, {}).get(f"{key}_std") for n in names]
        err = [e if e is not None and np.isfinite(e) else 0 for e in err]
        fig = go.Figure(go.Bar(x=names, y=vals, error_y=dict(type="data", array=err), marker_color=[colors.get(n, theme.MUTED) for n in names],
                               text=[f"{v:.3f}" for v in vals], textposition="outside"))  # fmt: skip
        if key == "auprc" and res.get("prevalence") is not None:
            fig.add_hline(y=res["prevalence"], line=dict(color=theme.INK, dash="dash"),
                          annotation_text=f"random guessing = prevalence {res['prevalence']:.4f}")  # fmt: skip
        fig.update_layout(title=f"Test {title} (mean ± std over 3 seeds)", height=380,
                          yaxis=dict(range=[0.5, 1.0]) if key == "auroc" else dict(rangemode="tozero"))  # fmt: skip
        st.plotly_chart(fig, width="stretch")

c3, c4 = st.columns(2)
with c3:
    pr = res["privacy"]
    fig = go.Figure()
    for k, lab, col, w in (
        ("uniform", "same ε for all", theme.MUTED, 2.5),
        ("adaptive", "FedGuard adaptive ε", theme.ACCENT, 4),
    ):
        ys = pr[k]
        if any(v is not None for v in ys):
            fig.add_trace(
                go.Scatter(x=pr["eps"], y=ys, mode="lines+markers", name=lab, line=dict(color=col, width=w))
            )
    if pr.get("no_dp") is not None:
        fig.add_hline(
            y=pr["no_dp"],
            line=dict(color=theme.OK, dash="dash"),
            annotation_text="no privacy (same normalisation)",
        )
    fig.update_layout(title="Privacy vs accuracy (FedGuard, async)", xaxis=dict(type="log", title="total ε per hospital (δ = 1e-5)",
                      tickvals=pr["eps"]), yaxis_title="test AUROC", height=380)  # fmt: skip
    st.plotly_chart(fig, width="stretch")
with c4:
    st.markdown("**False alarms (test set, validation-tuned thresholds)**")
    a = res["alerts"]
    t, g = a["threshold_only"], a["fedguard"]
    if t["false_per_100h"] is None:
        data.missing("results", "(alerts: `fedguard alerts -e fedguard`)")
    else:
        x, y = st.columns(2)
        with x:
            theme.big_number(
                f"{t['false_per_100h']:.2f}",
                f"threshold only: false alarms / 100 patient-hours · sensitivity {100 * t['sensitivity']:.0f}%",
            )
        with y:
            theme.big_number(
                f"{g['false_per_100h']:.2f}",
                f"FedGuard (uncertainty-gated) · sensitivity {100 * g['sensitivity']:.0f}%",
            )
        red = 100 * (1 - g["false_per_100h"] / t["false_per_100h"]) if t["false_per_100h"] else float("nan")
        st.markdown(
            f"**{red:.0f}% fewer false alarms** for a {100 * (t['sensitivity'] - g['sensitivity']):.1f}-point change in sensitivity."
        )
    als = full.get("alerts_per_seed") or []
    if als:
        cal = als[0]["calibration"]["test"]
        f = go.Figure()
        f.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=theme.MUTED, dash="dash"), showlegend=False)
        )
        for k, lab, col in (
            ("deterministic", "deterministic", theme.MUTED),
            ("mc_dropout", "MC-Dropout mean", theme.ACCENT),
        ):
            if k in cal:
                cu = cal[k]["curve"]
                pts = [
                    (mp, fp)
                    for mp, fp, n in zip(cu["mean_pred"], cu["frac_pos"], cu["count"], strict=True)
                    if n
                ]
                f.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                       name=f"{lab} (ECE {cal[k]['ece']:.3f})", line=dict(color=col)))  # fmt: skip
        f.update_layout(
            title="Calibration (FedGuard seed 0, test)",
            xaxis_title="mean predicted risk",
            yaxis_title="observed rate",
            height=330,
        )
        st.plotly_chart(f, width="stretch")

st.subheader("Tables")
tabs = data.tables()
if not tabs:
    data.missing("tables")
else:
    for name, md in tabs.items():
        if name == "ALL":
            continue
        with st.expander(name.replace("_", " ")):
            st.markdown(md)
