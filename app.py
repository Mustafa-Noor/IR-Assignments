"""IR course — weekly assignments in one Streamlit app.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import ui

st.set_page_config(page_title="IR Weekly Assignments", layout="wide")
st.title("Information Retrieval — Weekly Assignments")
st.write(
    "Every week's assignment runs on the **same 40 real research papers** (downloaded from arXiv) and the "
    "**same 15 test queries**, so the models can be compared fairly in Week 5. Use the sidebar to open each week."
)

weeks = pd.DataFrame([
    ("Week 1", "Foundations", "Text processing, inverted index, searching, term-overlap ranking"),
    ("Week 2", "IR components & models", "Formal D, Q, T, F, R; swappable weighting g(d); ad-hoc vs filtering"),
    ("Week 3", "Classical models", "Boolean (AND/OR/NOT), VSM TF-IDF + cosine, Naive Bayes, K-Means, precision/recall"),
    ("Week 4", "Probabilistic & structured", "BIM, BM25, P = 1/(1+e^-s), field search, overlapped / non-overlapped lists, proximal nodes"),
    ("Week 5", "Comparison & analysis", "All models on all queries: MAP, nDCG@10, P@5, parameter sweeps, failure analysis"),
], columns=["Week", "Topic", "Implemented"])
st.dataframe(weeks, hide_index=True, use_container_width=True)

st.subheader("The corpus")
corpus = ui.corpus()
c1, c2, c3 = st.columns(3)
c1.metric("Papers", len(corpus))
c2.metric("Categories", len({d["category"] for d in corpus}))
c3.metric("Unique index terms", len(ui.index().vocabulary))

table = pd.DataFrame([{"Doc": d["id"], "Title": d["title"], "Category": d["category"], "Year": d["year"],
                       "arXiv": f"https://arxiv.org/abs/{d['arxiv_id']}"} for d in corpus])
st.dataframe(table, hide_index=True, use_container_width=True,
             column_config={"arXiv": st.column_config.LinkColumn("arXiv")})

with st.expander("Read a paper's structured fields"):
    doc_id = st.selectbox("Paper", [d["id"] for d in corpus],
                          format_func=lambda i: f"{i} — {ui.by_id()[i]['title']}")
    paper = ui.by_id()[doc_id]
    for field in ["abstract", "introduction", "methodology", "results", "conclusion", "references"]:
        st.markdown(f"**{field.title()}**")
        st.write(paper[field] or "_(not available)_")
