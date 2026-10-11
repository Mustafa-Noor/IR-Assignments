import pandas as pd
import streamlit as st

import ui
from data import full_text
from week3.boolean import BooleanModel, BooleanQueryError, to_postfix
from week3.classify_cluster import classify_cv, cluster, train_classifier
from week3.evaluate import precision_recall_f1
from week3.vsm import VectorSpaceModel

st.set_page_config(page_title="Week 3 — Classical Models", layout="wide")
st.title("Week 3 — Boolean, Vector Space, Clustering & Classification")
st.caption("Assignment: mini search engine with an inverted index, AND/OR/NOT queries, vector similarity and ranked results.")


@st.cache_resource
def vsm():
    return VectorSpaceModel(ui.docs())


@st.cache_resource
def nb_cv():
    c = ui.corpus()
    return classify_cv([full_text(d) for d in c], [d["category"] for d in c])


@st.cache_resource
def nb_model():
    c = ui.corpus()
    return train_classifier([full_text(d) for d in c], [d["category"] for d in c])


@st.cache_resource
def kmeans(k):
    c = ui.corpus()
    return cluster([full_text(d) for d in c], [d["category"] for d in c], k=k)


tabs = st.tabs(["Boolean search", "Vector ranking", "Slide 14 example", "Classification", "Clustering"])

with tabs[0]:
    qs = ui.queries()
    pick = st.selectbox("Test query", ["(type my own)"] + [f"{q['id']} — {q['boolean']}" for q in qs])
    test = None if pick.startswith("(") else qs[[f"{q['id']} — {q['boolean']}" for q in qs].index(pick)]
    bq = st.text_input("Boolean query (AND, OR, NOT, parentheses)", test["boolean"] if test else "iot AND intrusion AND NOT survey",
                       key=f"bq_{pick}")
    try:
        hits = BooleanModel(ui.index()).search(bq)
        st.write("Postfix form:", " ".join(t[1] if isinstance(t, tuple) else t for t in to_postfix(bq)))
        st.write(f"**{len(hits)}** documents satisfy the query (Rel = 1). The Boolean model gives no ranking.")
        ui.results_table([(d, 1) for d in sorted(hits)], test)
        if test:
            p, r, f = precision_recall_f1(hits, test["relevant"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{p:.2f}")
            c2.metric("Recall", f"{r:.2f}")
            c3.metric("F1", f"{f:.2f}")
    except BooleanQueryError as e:
        st.error(f"Query error: {e}")

with tabs[1]:
    query, test = ui.query_picker("w3v")
    k = st.slider("Show top k", 3, 20, 10)
    ranked = vsm().rank(query, top_k=k)
    ui.results_table(ranked, test)
    if test:
        p, r, f = precision_recall_f1([d for d, _ in ranked], test["relevant"])
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Precision@{k}", f"{p:.2f}")
        c2.metric(f"Recall@{k}", f"{r:.2f}")
        c3.metric("F1", f"{f:.2f}")

with tabs[2]:
    st.write("Reproduces the slide's computed example with binary weights.")
    toy = {"D1": "retrieval indexing ranking", "D2": "ranking search engine", "D3": "database normalization"}
    m = VectorSpaceModel(toy, weighting="binary", analyzer=str.split)
    order = ["retrieval", "indexing", "ranking", "search", "engine", "database", "normalization"]
    qv = m.query_vector("retrieval ranking")
    rows = [{"Vector": "Query", **{t: int(qv.get(t, 0)) for t in order}, "cosine": "—"}]
    for d in toy:
        rows.append({"Vector": d, **{t: int(x) for t, x in zip(order, m.dense(d, order))},
                     "cosine": f"{m.cosine(qv, d):.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.success("D1 = 0.816, D2 = 0.408, D3 = 0.000 — matches the slide (also checked in tests/test_week3.py).")

with tabs[3]:
    res = nb_cv()
    st.metric("Naive Bayes accuracy (5-fold cross-validation)", f"{res['accuracy']:.1%}")
    st.write("Confusion matrix (rows = true category, columns = predicted):")
    st.dataframe(pd.DataFrame(res["confusion"], index=res["classes"], columns=res["classes"]))
    papers = ui.corpus()
    wrong = [{"Doc": d["id"], "Title": d["title"], "True": d["category"], "Predicted": p}
             for d, p in zip(papers, res["predictions"]) if p != d["category"]]
    st.write("Misclassified papers:")
    st.dataframe(pd.DataFrame(wrong), hide_index=True, use_container_width=True)
    text = st.text_area("Classify your own text", "A transformer model detects malicious network traffic in IoT devices.")
    st.write("Predicted category:", f"**{nb_model().predict([text])[0]}**")

with tabs[4]:
    k = st.slider("Number of clusters k", 2, 10, 6)
    res = kmeans(k)
    st.metric("Purity", f"{res['purity']:.1%}")
    papers = ui.corpus()
    for c in res["clusters"]:
        with st.expander(f"Cluster {c['cluster']} — {c['size']} papers — {c['categories']}"):
            st.write("Top terms:", ", ".join(c["top_terms"]))
            st.write("\n".join(f"- {papers[i]['id']} {papers[i]['title']}" for i in c["members"]))
