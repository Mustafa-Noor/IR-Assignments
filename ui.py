"""Shared Streamlit helpers: cached models and a common result list renderer."""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
import streamlit as st

from data import full_text, load_corpus, load_queries
from week1.index import InvertedIndex
from week4.structured import StructuredIndex


@st.cache_resource
def corpus():
    return load_corpus()


@st.cache_resource
def queries():
    return load_queries()


@st.cache_resource
def docs():
    return {d["id"]: full_text(d) for d in corpus()}


@st.cache_resource
def index():
    return InvertedIndex(docs())


@st.cache_resource
def structured(region_words=50):
    return StructuredIndex(corpus(), index(), region_words=region_words)


def by_id():
    return {d["id"]: d for d in corpus()}


def query_picker(key, default_text="intrusion detection on imbalanced datasets"):
    """Free-text box plus an optional test query (which unlocks relevance marks)."""
    qs = queries()
    options = ["(type my own)"] + [f"{q['id']} — {q['text']}" for q in qs]
    choice = st.selectbox("Test query", options, key=f"{key}_pick")
    test = None if choice == options[0] else qs[options.index(choice) - 1]
    text = st.text_input("Query", value=test["text"] if test else default_text, key=f"{key}_text_{choice}")
    return text, test


def results_table(rows, test_query=None, extra_cols=None):
    """rows: [(doc_id, score, *extra)]; extra_cols names the extra columns."""
    if not rows:
        st.info("No documents retrieved.")
        return
    papers = by_id()
    rel = test_query["relevant"] if test_query else {}
    table = []
    for rank, row in enumerate(rows, 1):
        doc_id, score, *extra = row
        rec = {"Rank": rank, "Doc": doc_id, "Score": round(float(score), 4),
               "Title": papers[doc_id]["title"], "Category": papers[doc_id]["category"]}
        if test_query:
            rec["Relevant"] = {2: "highly", 1: "partly"}.get(rel.get(doc_id), "—")
        for name, value in zip(extra_cols or [], extra):
            rec[name] = value
        table.append(rec)
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
