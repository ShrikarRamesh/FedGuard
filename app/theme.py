"""FedGuard visual identity (matches app/static/fedguard_demo.html) + Plotly template."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PAPER, PANEL, INK, MUTED, LINE = "#EDF2F1", "#F8FBFA", "#15283A", "#566A7B", "#C9D6D6"
ACCENT, ALERT, WARN, OK = "#1F6E8C", "#C8413A", "#B8790F", "#2E7D5B"
NODE_COLORS = {"A_MICU": "#2E7D7A", "A_SICU": "#4C5FB8", "B_MICU": "#B7791F", "B_SICU": "#B0487A"}
EXTRA_NODE_COLORS = ["#2E7D7A", "#4C5FB8", "#B7791F", "#B0487A", "#5B8C3A", "#8C5B3A", "#3A5B8C", "#8C3A6B"]
MON_BG, MON_LINE, MON_TEXT = "#0B1620", "#1E3242", "#CFE0EA"
VITAL_COLORS = {"hr": "#63D98F", "map": "#FF6B6B", "spo2": "#5CC8F0", "resp": "#F2D15C", "temp": "#E9EEF2"}
FONT = "Atkinson Hyperlegible, Segoe UI, system-ui, sans-serif"


def node_color(name: str, i: int = 0) -> str:
    return NODE_COLORS.get(name, EXTRA_NODE_COLORS[i % len(EXTRA_NODE_COLORS)])


pio.templates["fedguard"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=INK, size=15),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        colorway=[ACCENT, ALERT, OK, WARN, MUTED, "#4C5FB8", "#B0487A"],
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, title_font=dict(size=14)),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, title_font=dict(size=14)),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13)),
        hoverlabel=dict(font=dict(family=FONT)),
    )
)
pio.templates.default = "fedguard"

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: {FONT}; }}
.big-number {{ font-size: 2.6rem; font-weight: 700; line-height: 1.05; color: {INK}; }}
.big-label {{ font-size: 0.95rem; color: {MUTED}; }}
.demo-banner {{ position: sticky; top: 0; z-index: 999; background: {ALERT}; color: #fff; font-weight: 700;
  text-align: center; padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; letter-spacing: .04em; }}
.monitor {{ background: {MON_BG}; color: {MON_TEXT}; border-radius: 16px; padding: 14px; border: 1px solid {MON_LINE}; }}
.vitals {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
.vital {{ border: 1px solid {MON_LINE}; border-radius: 12px; padding: 8px 10px; }}
.vital .nm {{ font-size: 0.85rem; opacity: .85; display: flex; justify-content: space-between; }}
.vital .val {{ font-size: 2.4rem; font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }}
.banner {{ margin-top: 10px; border-radius: 12px; padding: 10px 14px; font-size: 1.1rem; font-weight: 700; }}
.banner.calm {{ background: #12303A; color: #BFE3EE; }}
.banner.hold {{ background: #3D3113; color: #F7DC8F; }}
.banner.fire {{ background: #5A1714; color: #FFD9D5; }}
.banner small {{ font-weight: 400; opacity: .85; }}
@media (max-width: 900px) {{ .vitals {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
</style>
"""


def apply(page_title: str) -> None:
    st.set_page_config(page_title=f"FedGuard · {page_title}", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def big_number(value: str, label: str) -> None:
    st.markdown(
        f'<div class="big-number">{value}</div><div class="big-label">{label}</div>', unsafe_allow_html=True
    )
