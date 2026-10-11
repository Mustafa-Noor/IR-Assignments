import json
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from week5.systems import build_systems

RESULTS = Path(__file__).resolve().parent.parent / "week5" / "results"
ANALYSIS = Path(__file__).resolve().parent.parent / "week5" / "ANALYSIS.md"

st.set_page_config(page_title="Week 5 — Comparison", layout="wide")
st.title("Week 5 — Comparison and analysis of every model")

if not (RESULTS / "summary.csv").exists():
    st.warning("Run `python -m week5.compare` first to generate the results.")
    st.stop()

summary = pd.read_csv(RESULTS / "summary.csv", index_col=0)
tabs = st.tabs(["Scoreboard", "Per-query explorer", "Parameter sweeps", "Written analysis"])

with tabs[0]:
    st.dataframe(summary.sort_values("MAP", ascending=False), use_container_width=True)
    st.bar_chart(summary["MAP"].sort_values(ascending=False), horizontal=True, color="#2a78d6")
    c1, c2 = st.columns(2)
    c1.image(str(RESULTS / "precision_vs_recall.png"))
    c2.image(str(RESULTS / "per_query_ndcg.png"))


@st.cache_resource
def systems():
    return build_systems(ui.corpus())[0]


with tabs[1]:
    qs = ui.queries()
    q = qs[st.selectbox("Query", range(len(qs)), format_func=lambda i: f"{qs[i]['id']} — {qs[i]['text']}")]
    st.write("Relevant:", ", ".join(f"{d} ({'highly' if g == 2 else 'partly'})" for d, g in q["relevant"].items()))
    cols = {}
    for s in systems():
        top = s.run(q)[:10]
        cols[s.name] = [f"{d} {'●' if q['relevant'].get(d) == 2 else '○' if d in q['relevant'] else '✗'}" for d in top] \
            + [""] * (10 - len(top))
    st.dataframe(pd.DataFrame(cols, index=range(1, 11)), use_container_width=True)
    st.caption("● highly relevant · ○ partly relevant · ✗ not relevant")

with tabs[2]:
    sw = json.loads((RESULTS / "sweeps.json").read_text())
    st.image(str(RESULTS / "bm25_grid.png"), width=520)
    c1, c2 = st.columns(2)
    c1.markdown("**Proximal nodes — MAP by λ**")
    c1.dataframe(pd.Series(sw["proximal_lambda"], name="MAP"))
    c2.markdown("**Field-weighted BM25 — MAP by weighting**")
    c2.dataframe(pd.Series(sw["field_weights"], name="MAP"))
    st.markdown("**Region size — overlapped vs non-overlapped (set precision / recall)**")
    st.dataframe(pd.DataFrame(sw["region_size"]).T)
    st.markdown("**Sigmoid score → probability** (mean spread of P over each query's top 10)")
    st.write(sw["sigmoid"])

with tabs[3]:
    if ANALYSIS.exists():
        st.markdown(ANALYSIS.read_text(encoding="utf-8").replace("results/", "week5/results/"))
