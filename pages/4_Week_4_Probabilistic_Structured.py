import streamlit as st

import ui
from data import FIELDS
from week1.preprocess import preprocess
from week4.probabilistic import BIM, BM25, to_probabilities
from week4.structured import NODE_EDGES

st.set_page_config(page_title="Week 4 — Probabilistic & Structured", layout="wide")
st.title("Week 4 — Probabilistic and Structured Retrieval")
st.caption("Assignment: probabilistic ranking (BM25), structured text search, non-overlapped list search and "
           "proximal node-based search.")

query, test = ui.query_picker("w4")
tabs = st.tabs(["BIM vs BM25", "Structured field search", "Overlapped vs non-overlapped lists", "Proximal nodes"])

with tabs[0]:
    st.latex(r"P(\text{Relevance}\mid Q,D) = \frac{1}{1+e^{-s}}")
    c1, c2, c3 = st.columns(3)
    k1 = c1.slider("k1 (term-frequency saturation)", 0.0, 3.0, 1.2, 0.1)
    b = c2.slider("b (length normalization)", 0.0, 1.0, 0.75, 0.05)
    calibrate = c3.toggle("Standardize scores before the sigmoid", True,
                          help="Raw BM25 scores are large, so 1/(1+e^-s) is ≈1 for every result.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Binary Independence Model** — terms present/absent, RSJ weights")
        bim = BIM(ui.index())
        ui.results_table(bim.rank(query, top_k=10), test)
        st.write("Term weights:", {t: round(bim.weight(t), 2) for t in dict.fromkeys(preprocess(query))})
    with right:
        st.markdown("**BM25** — term frequency, document length, collection IDF")
        ranked = BM25(ui.index(), k1, b).rank(query, top_k=10)
        ui.results_table([(d, s, round(p, 3)) for d, s, p in to_probabilities(ranked, calibrate)], test, ["P(rel)"])

with tabs[1]:
    fields = st.multiselect("Search only these fields", FIELDS, default=["title", "abstract", "methodology", "results"])
    if fields:
        rows = ui.structured().search_fields(query, fields, top_k=10)
        ui.results_table([(d, s, ", ".join(m)) for d, s, m in rows], test, ["Matched in"])
    st.markdown("**Field-weighted BM25** (title 3, abstract 2, body sections 1, references 0.3)")
    rows = ui.structured().field_bm25(query, top_k=10)
    ui.results_table([(d, s, ", ".join(m)) for d, s, m in rows], test, ["Matched in"])

with tabs[2]:
    words = st.select_slider("Region size (words)", [25, 50, 100, 200], 50)
    s = ui.structured(words)
    st.write(f"{len(s.regions)} non-overlapping regions. One region list per query term:")
    st.write({t: len(r) for t, r in s.term_lists(query).items()})
    left, right = st.columns(2)
    with left:
        ov = s.overlapped(query)
        st.markdown(f"**Overlapped lists** — all terms in the *same* region → {len(ov)} docs")
        ui.results_table([(d, sc, regs[0].label) for d, sc, regs in ov[:10]], test, ["First region"])
        if ov:
            st.caption(f"Example region {ov[0][2][0].label}:")
            st.write(ov[0][2][0].text)
    with right:
        nov = s.non_overlapped(query)
        st.markdown(f"**Non-overlapped lists** — terms may be in *different* regions → {len(nov)} docs")
        ui.results_table([(d, sc, len(regs)) for d, sc, regs in nov[:10]], test, ["Matched regions"])

with tabs[3]:
    lam = st.slider("λ — credit for a term found in a connected node", 0.0, 1.0, 0.7, 0.05)
    st.write("Structural links (slide 17): " + ", ".join(f"{a}–{c}" for a, c in NODE_EDGES))
    rows = ui.structured().proximal(query, lam=lam, top_k=10)
    ui.results_table([(d, sc, " + ".join(n)) for d, sc, n in rows], test, ["Anchor + connected nodes used"])
