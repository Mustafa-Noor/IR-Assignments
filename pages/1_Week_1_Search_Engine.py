import streamlit as st

import ui
from week1.preprocess import explain
from week1.search import pipeline_trace

st.set_page_config(page_title="Week 1 — Search Engine", layout="wide")
st.title("Week 1 — A simple document search engine")
st.caption("Assignment: develop a simple document search engine — indexing, searching and ranking a small text collection.")

query, test = ui.query_picker("w1", "machine learning for intrusion detection")
idx = ui.index()
trace = pipeline_trace(idx, query, top_k=10)

st.subheader("Pipeline: Query → Preprocessing → Index lookup → Ranking → Results")
steps = explain(query)
c1, c2, c3 = st.columns(3)
c1.markdown("**1. Tokenization**")
c1.code(" | ".join(steps["tokens"]) or "—")
c2.markdown("**2. Stop-word removal**")
c2.code(" | ".join(steps["without_stopwords"]) or "—")
c3.markdown("**3. Stemming**")
c3.code(" | ".join(steps["stems"]) or "—")

st.markdown("**4. Inverted index lookup** — each term's postings list")
for term, postings in trace["lookup"].items():
    st.write(f"`{term}` → {len(postings)} docs: {', '.join(postings) or '—'}")

st.markdown("**5. Ranking** — more shared query terms ranks higher; ties broken by total term frequency")
ui.results_table([(d, shared, tf) for d, shared, tf in trace["ranking"]], test, ["Total tf"])

with st.expander(f"Index statistics ({idx.N} docs, {len(idx.vocabulary)} terms)"):
    top = sorted(idx.postings.items(), key=lambda kv: -len(kv[1]))[:20]
    st.write("Terms appearing in the most documents:")
    st.dataframe([{"Term": t, "Document frequency": len(p)} for t, p in top], hide_index=True)
