"""The array versions must rank exactly like the original Week 1–4 classes on the same documents.

Needs a built collection:  python -m combined.collection 10000
"""
from collections import defaultdict

import numpy as np
import pytest

import week4.structured as structured
from combined.catalog import EDGES
from combined.collection import FIELDS, ROOT, Collection
from combined.evaluate import build_queries
from combined.fast.rankers import Rankers
from week1.index import InvertedIndex
from week1.search import rank_overlap
from week2.framework import IRSystem, g_tfidf
from week3.boolean import BooleanModel
from week3.vsm import VectorSpaceModel
from week4.probabilistic import BIM, BM25
from week4.structured import Region, StructuredIndex

pytestmark = pytest.mark.skipif(not (ROOT / "index_10000").exists(), reason="collection not built")


def original_index(index, terms):
    """Week 1's InvertedIndex holding the same postings as a compact index."""
    inv = InvertedIndex({})
    inv.postings = defaultdict(dict)
    for t, term in enumerate(terms):
        a, b = int(index.ptr[t]), int(index.ptr[t + 1])
        if b > a:
            inv.postings[term] = dict(zip(index.docs[a:b].tolist(), index.tfs[a:b].tolist()))
    inv.doc_ids = np.flatnonzero(index.doc_len > 0).tolist()
    inv.doc_len = {i: int(index.doc_len[i]) for i in inv.doc_ids}
    inv.N = len(inv.doc_ids)
    inv.avg_len = sum(inv.doc_len.values()) / inv.N
    return inv


@pytest.fixture(scope="module")
def world():
    c = Collection("10000")
    r = Rankers(c, "all")
    inv = original_index(r.index, c.terms)
    dev, test = build_queries(c)
    return c, r, inv, (dev + test)[:60]


def same(fast, slow):
    assert np.asarray(fast).tolist() == list(slow)


def ids(rows):
    return [row[0] for row in rows]


def test_term_overlap(world):
    _, r, inv, queries = world
    for q in queries:
        same(r.overlap(q["text"], depth=None), ids(rank_overlap(inv, q["text"])))


def test_tfidf_sum(world):
    _, r, inv, queries = world
    system = IRSystem.__new__(IRSystem)
    system.index, system.g, system.Q = inv, g_tfidf, []
    for q in queries:
        same(r.weighted_sum(q["text"], depth=None), ids(system.rank(q["text"], top_k=None)))


def test_boolean(world):
    _, r, inv, queries = world
    model = BooleanModel(inv)
    for q in queries:
        assert set(r.boolean(q["boolean"]).tolist()) == model.search(q["boolean"])


def test_vsm_scores_match(world):
    c, r, inv, queries = world
    vsm = VectorSpaceModel.__new__(VectorSpaceModel)
    vsm.weighting, vsm.N, vsm.analyzer = "tf-idf", inv.N, __import__("week1.preprocess").preprocess.preprocess
    vsm.df = {t: len(p) for t, p in inv.postings.items()}
    docs = defaultdict(dict)
    for t, postings in inv.postings.items():
        idf = vsm.idf(t)
        if idf > 0:
            for d, tf in postings.items():
                docs[d][t] = tf * idf
    vsm.vectors = docs
    vsm.norms = {d: sum(w * w for w in v.values()) ** 0.5 for d, v in docs.items()}
    for q in queries:
        slow = vsm.rank(q["text"])
        assert np.asarray(r.vsm(q["text"], depth=None)).tolist()[:10] == ids(slow)[:10]


def test_bim_and_bm25(world):
    _, r, inv, queries = world
    bim, bm25 = BIM(inv), BM25(inv)
    for q in queries:
        same(r.bim(q["text"], depth=None), ids(bim.rank(q["text"])))
        same(r.bm25(q["text"], depth=None), ids(bm25.rank(q["text"])))


@pytest.fixture(scope="module")
def original_structured(world):
    c, r, inv, _ = world
    s = StructuredIndex.__new__(StructuredIndex)
    s.whole, s.k1, s.b = inv, 1.2, 0.75
    s.fields = {f: original_index(c.index(f), c.terms) for f in FIELDS}
    for f in FIELDS:                                  # like Week 4, every document counts in a field's average
        s.fields[f].avg_len = float(c.index(f).avg_len)
    s._idf = BM25(inv).idf
    s.neighbors = defaultdict(set)
    for a, b in EDGES:
        s.neighbors[a].add(b)
        s.neighbors[b].add(a)
    s.regions, s.region_postings = [], defaultdict(set)
    index = c.index("regions")
    for rid, (doc, field) in enumerate(zip(c.region_doc.tolist(), c.region_field.tolist())):
        s.regions.append(Region(rid, doc, FIELDS[field], rid, ""))
    for t, term in enumerate(c.terms):
        s.region_postings[term] = set(index.docs[int(index.ptr[t]):int(index.ptr[t + 1])].tolist())
    return s


def test_overlapped_and_non_overlapped(world, original_structured):
    _, r, _, queries = world
    for q in queries:
        same(r.overlapped(q["text"], depth=None), ids(original_structured.overlapped(q["text"])))
        same(r.non_overlapped(q["text"], depth=None), ids(original_structured.non_overlapped(q["text"])))


def test_field_bm25_and_proximal(world, original_structured, monkeypatch):
    _, r, _, queries = world
    monkeypatch.setattr(structured, "FIELDS", FIELDS)
    weights = {f: 1.0 for f in FIELDS}
    for q in queries:
        same(r.field_bm25(q["text"], weights, depth=None), ids(original_structured.field_bm25(q["text"], weights)))
        same(r.proximal(q["text"], EDGES, depth=None), ids(original_structured.proximal(q["text"])))
