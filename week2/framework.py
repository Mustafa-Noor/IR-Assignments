"""Week 2 — Formal characterization of an IR system (slides 11, 20–22).

    D = {d1, d2, ...}          document collection
    Q = {q1, q2, ...}          query set
    T = {t1, t2, ...}          index terms (vocabulary)
    w_ij = g_i(d_j)            indexing (weighting) function
    F : Q -> 2^D               retrieval function   (which documents are candidates)
    R(q, d) -> real number     ranking function     (how relevant each one is)

The weighting function is swappable, so the same pipeline can run with
binary, raw TF, log TF or TF-IDF weights — later weeks plug in their own models.
"""
import math

from week1.index import InvertedIndex
from week1.preprocess import preprocess


def g_binary(index, term, doc_id):
    return 1.0 if index.tf(term, doc_id) else 0.0


def g_tf(index, term, doc_id):
    return float(index.tf(term, doc_id))


def g_log_tf(index, term, doc_id):
    tf = index.tf(term, doc_id)
    return 1 + math.log10(tf) if tf else 0.0


def g_tfidf(index, term, doc_id):
    tf, df = index.tf(term, doc_id), index.df(term)
    return tf * math.log10(index.N / df) if tf else 0.0


WEIGHTING = {"binary": g_binary, "tf": g_tf, "log-tf": g_log_tf, "tf-idf": g_tfidf}


class IRSystem:
    """Basic Document Ranking Pipeline: User -> Documents -> Index -> Matching -> Ranking -> Results."""

    def __init__(self, docs, weighting="tf-idf"):
        self.D = list(docs)                      # document collection
        self.index = InvertedIndex(docs)
        self.T = self.index.vocabulary           # index terms
        self.Q = []                              # queries seen so far
        self.g = WEIGHTING[weighting]
        self.weighting = weighting

    def w(self, term, doc_id):
        """w_ij = g_i(d_j)"""
        return self.g(self.index, term, doc_id)

    def F(self, query):
        """Retrieval function F: Q -> 2^D — every document sharing a term with q."""
        self.Q.append(query)
        out = set()
        for t in preprocess(query):
            out |= set(self.index.lookup(t))
        return out

    def R(self, query, doc_id):
        """Ranking function R(q, d) = sum of weights of the query terms in d."""
        return sum(self.w(t, doc_id) for t in preprocess(query))

    def rank(self, query, top_k=10):
        scored = [(d, self.R(query, d)) for d in self.F(query)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
