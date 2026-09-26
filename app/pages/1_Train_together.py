"""1 · Train together: replay recorded FL events, or follow a live Flower deployment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import data  # noqa: E402
import theme  # noqa: E402

theme.apply("Train together")
data.banner()
st.title("Four hospitals train one model")
st.caption("Each hospital trains on its own patients; only model updates travel to the server. "
           "Everything here is replayed from a real run's `events.jsonl` (time = simulated seconds, or wall-clock for Flower).")  # fmt: skip

res = data.load("results.json")
ref = {m["name"]: m["auroc"] for m in res["methods"]} if res else {}

mode = st.radio("Source", ["Replay a recorded run", "Live Flower deployment"], horizontal=True)


def client_names(events: list[dict]) -> list[str]:
    cfg = next((e for e in events if e["type"] == "config"), {})
    if cfg.get("clients"):
        return list(cfg["clients"])
    idx = sorted({e.get("client_index") for e in events if e.get("client_index") is not None})
    return [f"client {i}" for i in idx]


def client_of(e: dict, names: list[str]) -> str | None:
    if e.get("client") is not None:
        return e["client"]
    if e.get("client_index") is not None and e["client_index"] < len(names):
        return names[e["client_index"]]
    return None


def draw(events: list[dict], upto: int) -> None:
    names = client_names(events)
    cfg = next((e for e in events if e["type"] == "config"), {})
    shown = events[: upto + 1]
    state = {c: "training" for c in names}
    for e in shown:
        c = client_of(e, names)
        if c is None:
            continue
        state[c] = {"dispatch": "training", "update_received": "sent update", "merge": "merged", "update_lost": "offline",
                    "online": "training", "budget_exhausted": "budget used up", "dropped_stale": "update too stale"}.get(e["type"], state.get(c, ""))  # fmt: skip
    pos = {c: (math.cos(2 * math.pi * i / max(len(names), 1) + math.pi / 4), math.sin(2 * math.pi * i / max(len(names), 1) + math.pi / 4))
           for i, c in enumerate(names)}  # fmt: skip
    recent = [e for e in shown[-6:] if client_of(e, names)]
    fig = go.Figure()
    for i, c in enumerate(names):
        x, y = pos[c]
        hot = [e for e in recent if client_of(e, names) == c]
        col = theme.node_color(c, i)
        fig.add_trace(go.Scatter(x=[x, 0], y=[y, 0], mode="lines", line=dict(color=col if hot else theme.LINE, width=6 if hot else 2),
                                 hoverinfo="skip", showlegend=False))  # fmt: skip
        n = cfg.get("n_samples", {}).get(c)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=46, color=col, line=dict(color="white", width=2)),
                                 text=[f"<b>{c}</b><br>{state.get(c, '')}" + (f"<br>{n:,} windows" if n else "")],
                                 textposition="bottom center" if y < 0 else "top center", hoverinfo="skip", showlegend=False))  # fmt: skip
    version = max([e.get("version", 0) or 0 for e in shown] + [0])
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", marker=dict(size=70, color=theme.PANEL, line=dict(color=theme.ACCENT, width=4)),
                             text=[f"<b>Server</b><br>model v{version}"], textposition="middle center", showlegend=False, hoverinfo="skip"))  # fmt: skip
    fig.update_layout(height=430, xaxis=dict(visible=False, range=[-1.6, 1.6]), yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x"),
                      margin=dict(l=0, r=0, t=10, b=0))  # fmt: skip
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        t_now = shown[-1]["t"] if shown else 0.0
        a, b, c = st.columns(3)
        evals = [e for e in shown if e["type"] == "eval" and e.get("val_auroc") is not None]
        with a:
            theme.big_number(f"{evals[-1]['val_auroc']:.3f}" if evals else "–", "global validation AUROC")
        with b:
            theme.big_number(str(version), "global model version")
        with c:
            theme.big_number(
                f"{t_now:,.0f} s",
                "time (simulated)" if cfg.get("runtime") != "flower-deployment" else "time (wall-clock)",
            )
    with right:
        st.markdown("**Accuracy over time**")
        allev = [e for e in events if e["type"] == "eval" and e.get("val_auroc") is not None]
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=[e["t"] for e in evals], y=[e["val_auroc"] for e in evals], mode="lines+markers",
                                line=dict(color=theme.ACCENT, width=3), name="global model (validation)"))  # fmt: skip
        for lab, col in (("Centralized", theme.OK), ("Local-only", theme.ALERT)):
            if ref.get(lab) is not None:
                f2.add_hline(y=ref[lab], line=dict(color=col, dash="dash"), annotation_text=f"{lab} (test, 3-seed mean)",
                             annotation_position="bottom right")  # fmt: skip
        xmax = max([e["t"] for e in allev] + [1])
        f2.update_layout(height=260, xaxis_title="time (s)", yaxis_title="AUROC", xaxis_range=[0, xmax * 1.02], showlegend=False,
                         margin=dict(l=50, r=10, t=10, b=40))  # fmt: skip
        st.plotly_chart(f2, width="stretch", config={"displayModeBar": False})
        st.markdown("**Privacy budget used (total ε per hospital)**")
        budgets = cfg.get("budgets") or {}
        eps_now = next((e["eps"] for e in reversed(shown) if isinstance(e.get("eps"), dict)), {})
        if cfg.get("dp") and budgets:
            f3 = go.Figure()
            cs = list(budgets)
            f3.add_trace(
                go.Bar(
                    y=cs,
                    x=[budgets[c]["target_eps"] or 0 for c in cs],
                    orientation="h",
                    marker_color=theme.LINE,
                    name="budget",
                )
            )
            f3.add_trace(go.Bar(y=cs, x=[eps_now.get(c, 0) for c in cs], orientation="h", name="spent",
                                marker_color=[theme.node_color(c, i) for i, c in enumerate(cs)]))  # fmt: skip
            f3.update_layout(
                barmode="overlay",
                height=190,
                xaxis_title="ε",
                showlegend=False,
                margin=dict(l=70, r=10, t=5, b=35),
            )
            st.plotly_chart(f3, width="stretch", config={"displayModeBar": False})
            st.caption(
                f"δ = {next(iter(budgets.values()))['delta']:g}; unit of privacy = patient (RDP accountant, Opacus)."
            )
        else:
            st.caption("This run has no differential privacy: updates could leak patient data (see page 2).")
        st.markdown("**Server log**")
        lines = []
        for e in reversed(shown[-12:]):
            c = client_of(e, names) or "server"
            extra = ""
            if e["type"] == "merge":
                extra = f" (staleness {e.get('staleness')}, weight {e.get('weight', 0):.2f})"
            elif e["type"] == "eval" and e.get("val_auroc") is not None:
                extra = f" (val AUROC {e['val_auroc']:.3f})"
            lines.append(f"`{e['t']:8.1f}s` **{c}**: {e['type'].replace('_', ' ')}{extra}")
        st.markdown("  \n".join(lines) if lines else "_no events yet_")


if mode == "Replay a recorded run":
    runs = data.event_runs()
    if data.demo_mode() or not runs:
        if not runs:
            data.missing("events", "(no finished federated runs with `events.jsonl` yet)")
        st.stop()
    labels = list(runs)
    default = next((i for i, k in enumerate(labels) if k.startswith("fedguard · seed 0")), 0)
    choice = st.selectbox("Recorded run", labels, index=default)
    events = data.cached_events(str(runs[choice]), runs[choice].stat().st_mtime)
    key = f"pos::{choice}"
    st.session_state.setdefault(key, 0)
    st.session_state.setdefault("playing", False)
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("Play / pause"):
        st.session_state.playing = not st.session_state.playing
    if c2.button("Restart"):
        st.session_state[key] = 0

    @st.fragment(run_every=0.6 if st.session_state.get("playing") else None)
    def player() -> None:
        # the slider owns the position; advance it *before* the widget is created in this fragment run
        if st.session_state.get("playing"):
            st.session_state[key] = min(len(events) - 1, st.session_state[key] + 3)
        st.slider("Event", 0, len(events) - 1, key=key, help="Scrub through the recorded events")
        draw(events, st.session_state[key])

    player()

    with st.expander("Sync vs async under the same simulated clock (real runs, seed 0)"):
        f = go.Figure()
        for lab, col in (("fedavg", theme.MUTED), ("async_nodp", theme.ACCENT), ("abl_imbalance_fedavg", theme.ALERT),
                         ("abl_imbalance_fedguard_async_nodp", theme.OK)):  # fmt: skip
            k = f"{lab} · seed 0"
            if k in runs:
                ev = [
                    e
                    for e in data.cached_events(str(runs[k]), runs[k].stat().st_mtime)
                    if e["type"] == "eval" and e.get("val_auroc") is not None
                ]
                f.add_trace(
                    go.Scatter(
                        x=[e["t"] for e in ev],
                        y=[e["val_auroc"] for e in ev],
                        name=lab,
                        line=dict(color=col, width=2.5),
                    )
                )
        if f.data:
            f.update_layout(height=320, xaxis_title="simulated time (s)", yaxis_title="validation AUROC")
            st.plotly_chart(f, width="stretch")
        else:
            st.info(
                "Run `fedguard fl run -c experiments/fedavg` and `-c experiments/fedguard_async_nodp` to compare."
            )
else:
    live = data.latest_live_run()
    if live is None:
        st.info("No Flower deployment has written events yet. Start one with `scripts/deploy/start_local.ps1` and "
                "`scripts/deploy/run_flower.ps1` (see docs/demo.md); this view refreshes every 2 s.")  # fmt: skip
        st.stop()
    st.caption(f"Following `{live}` (refreshes every 2 s)")

    @st.fragment(run_every=2)
    def live_view() -> None:
        ev = data.read_events(live)
        if ev:
            draw(ev, len(ev) - 1)

    live_view()
