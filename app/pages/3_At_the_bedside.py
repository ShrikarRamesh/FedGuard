"""3 · At the bedside: replay exported test-set patients hour by hour with MC-Dropout alerts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import data  # noqa: E402
import theme  # noqa: E402
from fedguard.alerts.metrics import EARLY, LATE, episode_starts  # noqa: E402

theme.apply("At the bedside")
data.banner()
st.title("Replay an ICU stay, hour by hour")
st.caption("The FedGuard model runs 50 times with dropout on. It alerts only when the average risk is high AND the "
           "50 answers agree (low spread). Thresholds default to the values tuned on validation data.")  # fmt: skip

res = data.load("results.json")
full = data.load("results_full.json") or {}
if res is None or not res.get("patients"):
    data.missing(
        "results",
        "(needs `fedguard alerts -e fedguard`, `fedguard explain -e fedguard`, then `fedguard export`)",
    )
    st.stop()
pats = res["patients"]
meta = full.get("patients_meta", {})
REF = int(meta.get("refractory", 6))

with st.sidebar:
    pi = st.radio("Patient", range(len(pats)), format_func=lambda i: pats[i]["name"]
                  + (" · sepsis" if pats[i]["onset_hour"] is not None else " · no sepsis"))  # fmt: skip
    p = pats[pi]
    if p.get("selection"):
        st.caption(f"Selected as: {p['selection']} (rule in docs/demo.md)")
    tau_s_default = meta.get("tau_s", 0.1)
    tau_s_default = 0.3 if tau_s_default is None or not np.isfinite(tau_s_default) else float(tau_s_default)
    thr_r = st.slider("Alert when risk is above", 0.05, 0.97, float(meta.get("tau_r", 0.5)), 0.01)
    thr_s = st.slider("…and uncertainty (std) is below", 0.0, max(0.3, tau_s_default), tau_s_default, 0.005)
    st.caption(
        f"Validation-tuned: risk > {meta.get('tau_r', '–')}, std < {meta.get('tau_s', '–')}, refractory {REF} h."
    )

mean = np.asarray(p["risk_mean"], float)
std = np.asarray(p["risk_std"], float)
H = len(mean)
key = f"hour::{pi}"
st.session_state.setdefault(key, 0)
st.session_state.setdefault("bed_play", False)
c1, c2, _ = st.columns([1, 1, 6])
if c1.button("Play / pause"):
    st.session_state.bed_play = not st.session_state.bed_play
if c2.button("Restart"):
    st.session_state[key] = 0


def is_false(h: int) -> bool:
    on = p["onset_hour"]
    return on is None or not (on - EARLY <= h <= on + LATE)


def fmt(v, dp=0) -> str:
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{dp}f}"


@st.fragment(run_every=0.45 if st.session_state.get("bed_play") else None)
def bedside() -> None:
    if st.session_state.get("bed_play"):
        st.session_state[key] = min(H - 1, st.session_state[key] + 1)
    t = st.slider("ICU hour", 0, H - 1, key=key)
    left, right = st.columns([1.55, 1])
    with left:
        vit = [
            ("hr", "HR", "bpm", 0),
            ("map", "MAP", "mmHg", 0),
            ("spo2", "SpO₂", "%", 0),
            ("resp", "Resp", "/min", 0),
            ("temp", "Temp", "°C", 1),
        ]
        cells = "".join(
            f'<div class="vital" style="color:{theme.VITAL_COLORS[k]}"><div class="nm"><span>{n}</span><span>{u}</span></div>'
            f'<div class="val">{fmt(p[k][t], d)}</div></div>'
            for k, n, u, d in vit
        )
        m, s = mean[t], std[t]
        if m > thr_r and s < thr_s:
            b = f'<div class="banner fire">⚠ Sepsis alert <small>Risk {m:.2f} ± {s:.3f}. Review the patient.</small></div>'
        elif m > thr_r:
            b = f'<div class="banner hold">High risk, but the model is unsure <small>Risk {m:.2f} ± {s:.3f}: alert held back.</small></div>'
        else:
            b = f'<div class="banner calm">No alert <small>Risk {m:.2f} ± {s:.3f}</small></div>'
        st.markdown(f'<div class="monitor"><div style="display:flex;justify-content:space-between"><b>{p["name"]}</b>'
                    f'<span>ICU hour {t}</span></div><div class="vitals" style="margin-top:8px">{cells}</div>{b}</div>',
                    unsafe_allow_html=True)  # fmt: skip
        thr_alerts = mean[: t + 1] > thr_r
        fg_alerts = thr_alerts & (std[: t + 1] < thr_s)
        a_thr, a_fg = episode_starts(thr_alerts, REF), episode_starts(fg_alerts, REF)
        hrs = np.arange(t + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[hrs, hrs[::-1]], y=np.r_[np.clip(mean[: t + 1] + std[: t + 1], 0, 1), np.clip(mean[: t + 1] - std[: t + 1], 0, 1)[::-1]],
                                 fill="toself", fillcolor="rgba(31,110,140,0.22)", line=dict(width=0), hoverinfo="skip", name="± 1 std (MC-Dropout)"))  # fmt: skip
        fig.add_trace(
            go.Scatter(x=hrs, y=mean[: t + 1], line=dict(color=theme.ACCENT, width=3), name="mean risk")
        )
        fig.add_hline(y=thr_r, line=dict(color=theme.ALERT, dash="dash"), annotation_text="alert threshold")
        if a_thr:
            fig.add_trace(go.Scatter(x=a_thr, y=[0.02] * len(a_thr), mode="markers", marker=dict(symbol="triangle-up", size=14, color=theme.MUTED),
                                     name="threshold-only alarm"))  # fmt: skip
        if a_fg:
            fig.add_trace(go.Scatter(x=a_fg, y=mean[a_fg], mode="markers", marker=dict(size=15, color=theme.ALERT, line=dict(color="white", width=2)),
                                     name="FedGuard alert"))  # fmt: skip
        if p["onset_hour"] is not None:
            fig.add_vline(
                x=p["onset_hour"],
                line=dict(color=theme.INK, dash="dot"),
                annotation_text="sepsis onset (hidden from model)",
            )
        fig.update_layout(height=330, xaxis=dict(range=[0, H - 1], title="ICU hour"), yaxis=dict(range=[0, 1], title="risk of sepsis in 6 h"),
                          legend=dict(orientation="h", y=1.15))  # fmt: skip
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("**Alarms so far**")
        a, b = st.columns(2)
        with a:
            theme.big_number(str(len(a_thr)), f"threshold only · {sum(is_false(h) for h in a_thr)} false")
        with b:
            lead = (
                [p["onset_hour"] - h for h in a_fg if not is_false(h)] if p["onset_hour"] is not None else []
            )
            theme.big_number(
                str(len(a_fg)),
                f"FedGuard · {sum(is_false(h) for h in a_fg)} false"
                + (f" · {lead[0]} h before onset" if lead else ""),
            )
        st.markdown("**Why: what drove the risk (last 12 hours)**")
        attr = np.asarray(p.get("attr") or [[0.0]], float)
        feats = p.get("attr_features") or ["–"]
        lo = max(0, t - 11)
        z = attr[lo : t + 1].T
        hm = go.Figure(go.Heatmap(z=z, x=list(range(lo, t + 1)), y=feats, colorscale=[[0, "#2E7D5B"], [0.5, "#F8FBFA"], [1, theme.ALERT]],
                                  zmid=0, colorbar=dict(title="IG", thickness=10)))  # fmt: skip
        hm.update_layout(height=360, xaxis_title="ICU hour", margin=dict(l=70, r=10, t=10, b=40))
        st.plotly_chart(hm, width="stretch", config={"displayModeBar": False})
        st.caption("Integrated Gradients per variable (value + measured flag + time since measured) for the prediction at each hour; "
                   "red pushes risk up, green down.")  # fmt: skip


bedside()
