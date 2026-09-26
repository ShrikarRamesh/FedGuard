"""FedGuard demo app (home page). Run:  cd app && streamlit run streamlit_app.py  [-- --demo]"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import data  # noqa: E402
import theme  # noqa: E402

theme.apply("Home")
data.banner()

st.title("FedGuard")
st.markdown(
    "#### Hospitals learn one sepsis early-warning model together. Patient data stays home.\n"
    "Privacy-preserving federated learning on PhysioNet/CinC 2019 (two hospital systems × two ICU types), "
    "patient-level differential privacy, asynchronous aggregation, and uncertainty-gated bedside alerts."
)

res = data.load("results.json")
full = data.load("results_full.json")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/1_Train_together.py", label="1 · Train together", icon="🏥")
    st.caption("Replay recorded federated training, or watch a live Flower deployment.")
with c2:
    st.page_link("pages/2_Try_to_steal_data.py", label="2 · Try to steal data", icon="🕵️")
    st.caption("Gradient-inversion attack on one hospital's update, with and without DP.")
with c3:
    st.page_link("pages/3_At_the_bedside.py", label="3 · At the bedside", icon="🩺")
    st.caption("Replay test-set ICU stays hour by hour with MC-Dropout alerts and explanations.")
with c4:
    st.page_link("pages/4_Results.py", label="4 · Results", icon="📊")
    st.caption("All tables and curves from the experiment runs.")

st.divider()
if res is None:
    data.missing("results", "The pages that need other artefacts will say so.")
else:
    m = {x["name"]: x for x in res["methods"]}
    a, b, c = st.columns(3)
    with a:
        v = m.get("FedGuard", {}).get("auroc")
        theme.big_number(
            "not run yet" if v is None else f"{v:.3f}", "FedGuard test AUROC (3-seed mean, ε = 3)"
        )
    with b:
        v = m.get("Centralized", {}).get("auroc")
        theme.big_number("not run yet" if v is None else f"{v:.3f}", "Centralized (pooled data, not allowed)")
    with c:
        v = res.get("prevalence")
        theme.big_number(
            "not run yet" if v is None else f"{100 * v:.2f}%", "positive patient-hours in the test set"
        )
    if full and full.get("missing"):
        st.info("Not run yet (shown as blank): " + ", ".join(full["missing"]))

st.divider()
html = Path(__file__).resolve().parent / "static" / "fedguard_demo.html"
if html.exists():
    st.markdown("**Stand-alone presentation front end.** The HTML demo loads the same `results.json` "
                "(use its *Load results* button).")  # fmt: skip
    st.download_button(
        "Download fedguard_demo.html", html.read_bytes(), file_name="fedguard_demo.html", mime="text/html"
    )
    st.markdown("[Open the HTML demo in a new tab](app/static/fedguard_demo.html)")
