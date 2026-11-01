"""Each model's own flow for one query, with the live values at every step.

A step is {title, formula, note, table: {columns, rows}}. The browser draws them all the same way.
"""
import math

import numpy as np

from combined.catalog import EDGES
from combined.collection import FIELDS
from combined.registry import MODELS
from setmodels.extended_boolean import normalize_query
from setmodels.fuzzy import band
from week1.preprocess import explain as text_steps, preprocess
from week3.boolean import to_postfix


def step(title, formula=None, note=None, columns=None, rows=None):
    return {"title": title, "formula": formula, "note": note,
            "table": {"columns": columns, "rows": rows} if columns else None}


def text_processing(query):
    s = text_steps(query)
    return step("Text processing", note="Tokenize, drop stop-words, then stem with our Porter stemmer.",
                columns=["Tokens", "Without stop-words", "Stems"],
                rows=[[" ".join(s["tokens"]), " ".join(s["without_stopwords"]), " ".join(s["stems"])]])


def index_lookup(index, terms):
    return step("Inverted index lookup", note="Each term's postings list, read from the index.",
                columns=["Term", "df", "First documents"],
                rows=[[t, index.df(t), ", ".join(str(d) for d in index.postings(t)[0][:5].tolist())] for t in terms])


def term_rows(index, terms, doc):
    tf = {}
    for t in terms:
        docs, tfs = index.postings(t)
        i = np.searchsorted(docs, doc)
        tf[t] = int(tfs[i]) if i < len(docs) and docs[i] == doc else 0
    return tf


def bm25_flow(world, track, query):
    r = world.rankers(track)
    p = world.params
    k1, b = p["k1"], p["b"]
    terms = preprocess(query)
    ranked = r.bm25(query, k1, b, depth=5).tolist()
    steps = [text_processing(query), index_lookup(r.index, terms)]
    if ranked:
        d = ranked[0]
        tf = term_rows(r.index, terms, d)
        ratio = float(r.index.doc_len[d] / r.index.avg_len)
        rows = []
        total = 0.0
        for t in terms:
            df = r.index.df(t)
            idf = r.bm25_idf(df) if df else 0.0
            part = idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * ratio)) if tf[t] else 0.0
            total += part
            rows.append([t, df, round(idf, 3), tf[t], round(part, 3)])
        steps.append(step(f"Score the top document ({d})", MODELS["BM25"]["formula"][0],
                          f"k₁ = {k1}, b = {b}, |d|/avgdl = {ratio:.2f}. Total score = {total:.3f}",
                          ["Term", "df", "IDF", "tf", "Contribution"], rows))
    return steps, ranked


def bim_flow(world, track, query):
    r = world.rankers(track)
    terms = list(dict.fromkeys(preprocess(query)))
    ranked = r.bim(query, depth=5).tolist()
    rows = [[t, r.index.df(t), round(r.bim_weight(r.index.df(t)), 3)] for t in terms if r.index.df(t)]
    steps = [text_processing(query), index_lookup(r.index, terms),
             step("Term weights", MODELS["BIM"]["formula"][1],
                  "A term in more than half the documents gets a negative weight.", ["Term", "df", "Weight"], rows)]
    if ranked:
        scores = sorted(((sum(w for t, _, w in rows if term_rows(r.index, [t], d)[t]), d) for d in ranked), reverse=True)
        steps.append(step("Score to probability", MODELS["BIM"]["formula"][2], columns=["Document", "Score s", "P"],
                          rows=[[d, round(s, 3), round(1 / (1 + math.exp(-s)), 3)] for s, d in scores]))
    return steps, ranked


def vsm_flow(world, track, query):
    r = world.rankers(track)
    terms = preprocess(query)
    weights = [[t, r.index.df(t), round(r._idf(r.index.df(t)), 3)] for t in dict.fromkeys(terms)]
    ranked = r.vsm(query, depth=5).tolist()
    return [text_processing(query), step("Query vector", MODELS["VSM cosine"]["formula"][0],
                                         columns=["Term", "df", "idf"], rows=weights),
            step("Compare direction", MODELS["VSM cosine"]["formula"][1], "Every document with a shared word is scored.",
                 columns=["Rank", "Document", "‖d‖"],
                 rows=[[i + 1, d, round(float(r.vsm_norms()[d]), 2)] for i, d in enumerate(ranked)])], ranked


def overlap_flow(world, track, query):
    r = world.rankers(track)
    terms = list(dict.fromkeys(preprocess(query)))
    ranked = r.overlap(query, depth=5).tolist()
    rows = []
    for d in ranked:
        tf = term_rows(r.index, terms, d)
        rows.append([d, sum(1 for v in tf.values() if v), sum(tf.values())])
    return [text_processing(query), index_lookup(r.index, terms),
            step("Rank by shared terms", MODELS["Term overlap"]["formula"][0], columns=["Document", "Shared terms", "Total tf"],
                 rows=rows)], ranked


def tfidf_flow(world, track, query):
    r = world.rankers(track)
    terms = preprocess(query)
    ranked = r.weighted_sum(query, depth=5).tolist()
    candidates = np.zeros(r.index.N, bool)
    for t in terms:
        candidates[r.index.postings(t)[0]] = True
    return [step("The formal system", r"D,\ Q,\ T,\ F: Q \rightarrow 2^D,\ R(q,d)",
                 columns=["|D|", "|T|", "|F(q)|"], rows=[[f"{r.N:,}", f"{len(world.collection.terms):,}", f"{int(candidates.sum()):,}"]]),
            text_processing(query),
            step("Weights", MODELS["TF-IDF sum"]["formula"][2], columns=["Term", "df", "idf"],
                 rows=[[t, r.index.df(t), round(r._idf(r.index.df(t)), 3)] for t in terms])], ranked


def boolean_query(query):
    if any(op in query.split() for op in ("AND", "OR", "NOT", "BUT")) or "(" in query:
        return query
    words = [w for w in query.replace("'", " ").split() if preprocess(w)]
    return " AND ".join(words)


def show_postfix(query):
    return " ".join(t[1] if isinstance(t, tuple) else t for t in to_postfix(query))


def boolean_flow(world, track, query):
    r = world.rankers(track)
    q = boolean_query(query)
    mask = r.boolean_mask(normalize_query(q))
    rows = [[t, r.index.df(s), ""] for t, s in ((t, (preprocess(t) or [t])[0]) for t in
                                                 (x[1] for x in to_postfix(normalize_query(q)) if isinstance(x, tuple)))]
    ranked = np.flatnonzero(mask)[:5].tolist()
    return [step("Parse the query", note=f"{q}  →  postfix: {show_postfix(normalize_query(q))}"),
            step("Term sets", columns=["Term", "Documents containing it", ""], rows=rows),
            step("Set operations", MODELS["Boolean"]["formula"][0], f"{int(mask.sum()):,} documents satisfy the query. They are not ranked.")], ranked


def field_flow(world, track, query):
    r = world.rankers(track)
    p = world.params_for(track)
    fields = r.fields
    terms = list(dict.fromkeys(preprocess(query)))
    ranked = r.field_bm25(query, p["weights"], p["k1"], p["b"], depth=5).tolist()
    steps = [text_processing(query), step("Search each field", note="Field weights: " + ", ".join(f"{f} {w}" for f, w in p["weights"].items()),
                                          columns=["Term"] + fields,
                                          rows=[[t] + [world.collection.index(f).df(t) for f in fields] for t in terms])]
    if ranked:
        d = ranked[0]
        steps.append(step(f"Matched in the top document ({d})", MODELS["Field BM25"]["formula"][0],
                          columns=["Term"] + fields,
                          rows=[[t] + [term_rows(world.collection.index(f), [t], d)[t] for f in fields] for t in terms]))
    return steps, ranked


def lists_flow(world, track, query, overlapped):
    r = world.rankers(track)
    lists = r._region_lists(query)
    ranked = (r.overlapped if overlapped else r.non_overlapped)(query, depth=5).tolist()
    name = "Overlapped lists" if overlapped else "Non-overlapped lists"
    return [text_processing(query),
            step("One list of regions per term", note="A region is one turn of the dialogue, or the narrative.",
                 columns=["Term", "Regions"], rows=[[t, f"{len(v):,}"] for t, v in lists.items()]),
            step("Combine the lists", MODELS[name]["formula"][0],
                 "All terms must share a region." if overlapped else "A document qualifies with any term, in any region.",
                 columns=["Rank", "Document"], rows=[[i + 1, d] for i, d in enumerate(ranked)])], ranked


def proximal_flow(world, track, query):
    r = world.rankers(track)
    p = world.params
    ranked = r.proximal(query, EDGES, p["lam"], p["k1"], p["b"], depth=5).tolist()
    links = sorted(f"{a}–{b}" for a, b in EDGES if a in r.fields and b in r.fields)
    terms = list(dict.fromkeys(preprocess(query)))
    steps = [text_processing(query), step("Structural links", note=f"λ = {p['lam']}. Links: " + ", ".join(links))]
    if ranked:
        d = ranked[0]
        steps.append(step(f"Where the words sit in document {d}", MODELS["Proximal nodes"]["formula"][0],
                          columns=["Term"] + r.fields,
                          rows=[[t] + [term_rows(world.collection.index(f), [t], d)[t] for f in r.fields] for t in terms]))
    return steps, ranked


def fuzzy_flow(world, track, query, use_or):
    model = world.fuzzy(track)
    q = boolean_query(query)
    if use_or:
        q = q.replace(" AND ", " OR ")
    stems = [s for w in q.replace("(", " ").replace(")", " ").split() if w not in ("AND", "OR", "NOT", "BUT")
             for s in preprocess(w)]
    rows = [[s, ", ".join(f"{v} ({m:.2f})" for v, m in model.expand(s)) or "no close word"] for s in stems]
    ranked = model.rank(q, depth=5).tolist()
    mu = model.scores(q)
    return [step("Partial matching", MODELS["Fuzzy (AND)"]["formula"][4], "Vocabulary words close to each query word, with their similarity.",
                 columns=["Query word", "Close words"], rows=rows),
            step("Membership", MODELS["Fuzzy (AND)"]["formula"][1], "tf 1 → 0.33, tf 2 → 0.67, tf 3 or more → 1.0, times the similarity."),
            step("Combine and band", MODELS["Fuzzy (OR)"]["formula"][0] if use_or else MODELS["Fuzzy (AND)"]["formula"][2],
                 f"Query: {q}", columns=["Document", "μ", "Band"],
                 rows=[[d, round(float(mu[d]), 3), band(float(mu[d]))] for d in ranked])], ranked


def extended_flow(world, track, query):
    model = world.extended(track)
    q = boolean_query(query)
    gate = world.rankers(track).boolean_mask(normalize_query(q))
    s = model.scores(q)
    ranked = model.rank(q, depth=5).tolist()
    return [step("Set logic decides what is kept", MODELS["Extended Boolean"]["formula"][0],
                 f"{q} keeps {int(gate.sum()):,} documents."),
            step("p-norm orders them", MODELS["Extended Boolean"]["formula"][2], f"p = {model.p}. This part goes beyond the slides.",
                 ["Document", "Score"], [[d, round(float(s[d]), 3)] for d in ranked])], ranked


def gvsm_flow(world, track, query):
    model = world.gvsm(track)
    terms = preprocess(query)
    related = [[t, ", ".join(f"{w} ({g:.2f})" for w, g in model.related(t)[1:6]) or "none"] for t in terms]
    ranked = model.rank(terms, depth=5).tolist()
    return [text_processing(query),
            step("Generalized term vectors", MODELS["GVSM"]["formula"][0], "Correlated words from co-occurrence: one row of G per query word.",
                 columns=["Term", "Correlated words"], rows=related),
            step("Similarity through G", MODELS["GVSM"]["formula"][1],
                 "The query spreads onto its correlated words, so a document can match without the original word.",
                 columns=["Rank", "Document"], rows=[[i + 1, d] for i, d in enumerate(ranked)])], ranked


def lsi_flow(world, track, query):
    model = world.lsi(track)
    z = model.fold_in(query)
    strongest = np.argsort(-np.abs(z))[:5]
    ranked = model.rank(query, depth=5).tolist()
    return [text_processing(query),
            step("The term-document matrix", r"A \in \mathbb{R}^{terms \times documents}",
                 columns=["Terms kept", "Documents", "Concepts k"], rows=[[f"{len(model.kept):,}", f"{len(model.docs):,}", model.k]]),
            step("SVD", MODELS["LSI"]["formula"][0], "Largest singular values (concept strength).",
                 columns=[f"σ{i + 1}" for i in range(6)], rows=[[round(float(x), 1) for x in model.singular[:6]]]),
            step("Fold the query into concept space", MODELS["LSI"]["formula"][1], columns=["Concept", "q̂"],
                 rows=[[int(i) + 1, round(float(z[i]), 3)] for i in strongest]),
            step("Rank by cosine", MODELS["LSI"]["formula"][2], columns=["Rank", "Document"],
                 rows=[[i + 1, d] for i, d in enumerate(ranked)])], ranked


FLOWS = {
    "Term overlap": overlap_flow, "TF-IDF sum": tfidf_flow, "Boolean": boolean_flow, "VSM cosine": vsm_flow,
    "BIM": bim_flow, "BM25": bm25_flow, "Field BM25": field_flow,
    "Overlapped lists": lambda w, t, q: lists_flow(w, t, q, True), "Non-overlapped lists": lambda w, t, q: lists_flow(w, t, q, False),
    "Proximal nodes": proximal_flow, "Fuzzy (AND)": lambda w, t, q: fuzzy_flow(w, t, q, False),
    "Fuzzy (OR)": lambda w, t, q: fuzzy_flow(w, t, q, True), "Extended Boolean": extended_flow, "GVSM": gvsm_flow, "LSI": lsi_flow,
}
