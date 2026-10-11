import pandas as pd
import streamlit as st

import ui
from data import full_text
from week1.preprocess import preprocess
from week2.filtering import simulate_stream
from week2.framework import WEIGHTING, IRSystem

st.set_page_config(page_title="Week 2 — IR Components & Models", layout="wide")
st.title("Week 2 — Basic Document Ranking Pipeline")
st.caption("Formal characterization of an IR system, a swappable indexing function, and ad-hoc retrieval vs filtering.")


@st.cache_resource
def system(scheme):
    return IRSystem(ui.docs(), weighting=scheme)


tab_adhoc, tab_filter = st.tabs(["Ad-hoc retrieval (formal model)", "Filtering (stream + profile)"])

with tab_adhoc:
    query, test = ui.query_picker("w2")
    scheme = st.radio("Indexing function  w_ij = g_i(d_j)", list(WEIGHTING), index=3, horizontal=True)
    s = system(scheme)
    candidates = s.F(query)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("|D| documents", len(s.D))
    c2.metric("|T| index terms", len(s.T))
    c3.metric("|F(q)| candidates", len(candidates))
    c4.metric("Query terms", len(preprocess(query)))

    st.latex(r"F: Q \rightarrow 2^D \qquad R(q,d) = \sum_{t \in q} w_{t,d} \qquad w_{t,d} = g_t(d)")
    ui.results_table(s.rank(query, top_k=10), test)

    with st.expander("Term weights w_ij for the top 5 documents"):
        terms = list(dict.fromkeys(preprocess(query)))
        top = [d for d, _ in s.rank(query, top_k=5)]
        st.dataframe(pd.DataFrame({d: [round(s.w(t, d), 3) for t in terms] for d in top}, index=terms))

with tab_filter:
    st.write("A **static profile** watches a **stream** of new papers (oldest arXiv id first). "
             "Every paper whose cosine score clears the threshold is pushed to the user.")
    profile = st.text_input("User profile / interests", "intrusion detection network security attacks")
    threshold = st.slider("Delivery threshold", 0.0, 0.5, 0.15, 0.01)
    stream = sorted(ui.corpus(), key=lambda d: d["arxiv_id"])
    log = simulate_stream([(d["id"], full_text(d)) for d in stream], profile, threshold)
    papers = ui.by_id()
    pushed = [r for r in log if r["delivered"]]
    st.metric("Papers pushed", f"{len(pushed)} of {len(log)}")
    df = pd.DataFrame([{**r, "title": papers[r["doc_id"]]["title"], "category": papers[r["doc_id"]]["category"]}
                       for r in log])
    st.dataframe(df, hide_index=True, use_container_width=True)
