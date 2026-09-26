"""2 · Try to steal data: gradient-inversion attack outputs (real `fedguard attack` results)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import data  # noqa: E402
import theme  # noqa: E402

theme.apply("Try to steal data")
data.banner()
st.title("Can an attacker rebuild a patient from an update?")
st.caption("A curious server sees the gradient of ONE patient's 24-hour window (batch size 1, dropout off: the "
           "attacker's best case) and optimises a fake window until its gradient matches (DLG/iDLG, cosine matching).")  # fmt: skip

att = data.load("attack/attack.json")
if att is None:
    data.missing("attack", "(`fedguard attack -e fedavg --seed 0`)")
    st.stop()

cond_names = list(att["conditions"])


def _label(c: str) -> str:
    if c == "no_dp":
        return "No privacy noise"
    if c == "clip_only":
        return "Clipping only (no noise)"
    eps = att["conditions"][c].get("epsilon")
    return f"FedGuard DP, ε = {eps:g}" if eps is not None else c


pretty = {c: _label(c) for c in cond_names}
left, right = st.columns([1.5, 1])
with right:
    cond = st.radio("Update sent with", cond_names, format_func=pretty.get, index=0)
    c = att["conditions"][cond]
    lo, hi = c["mean_r_ci95"]
    theme.big_number(f"r = {c['mean_r']:.2f}", f"mean Pearson correlation, true vs reconstructed vitals (95% CI {lo:.2f} to {hi:.2f}, "
                     f"{att['n']} test patients)")  # fmt: skip
    st.markdown(f"MSE (standardised units): **{c['mse']:.2f}** · label recovered for **{100 * c['label_accuracy']:.0f}%** of windows"
                + (f" · noise multiplier σ = **{c['sigma']:.2f}**" if c.get("sigma") else ""))  # fmt: skip
    ex_i = st.selectbox("Example patient", range(len(att["examples"])),
                        format_func=lambda i: f"Test patient {att['examples'][i]['patient_id']}")  # fmt: skip
with left:
    ex = att["examples"][ex_i]
    vit = [v for v in ("HR", "MAP", "Resp") if v in ex["true"]]
    fig = make_subplots(rows=len(vit), cols=1, shared_xaxes=True, subplot_titles=vit, vertical_spacing=0.08)
    for r, v in enumerate(vit, 1):
        fig.add_trace(go.Scatter(y=ex["true"][v], name="real patient (never leaves the hospital)", line=dict(color=theme.INK, width=3),
                                 showlegend=r == 1), row=r, col=1)  # fmt: skip
        fig.add_trace(go.Scatter(y=ex["reconstructed"][cond][v], name="attacker's reconstruction", line=dict(color=theme.ALERT, width=3, dash="dash"),
                                 showlegend=r == 1), row=r, col=1)  # fmt: skip
    fig.update_layout(height=520, legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="hour in the 24 h window", row=len(vit), col=1)
    st.plotly_chart(fig, width="stretch")

st.subheader("All conditions")
rows = [{"condition": pretty[k], "σ": v.get("sigma"), "mean r": v["mean_r"], "95% CI": f"{v['mean_r_ci95'][0]:.2f} to {v['mean_r_ci95'][1]:.2f}",
         "MSE": v["mse"], "label accuracy": v["label_accuracy"]} for k, v in att["conditions"].items()]  # fmt: skip
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Vitals are shown in clinical units after inverting the model's input standardisation. Metrics use every "
           "vital whose true 24 h series is not constant.")  # fmt: skip
