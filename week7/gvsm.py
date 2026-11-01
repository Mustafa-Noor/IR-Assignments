"""Week 7 — Generalized Vector Space Model (slides 6–9, 16).

VSM assumes terms are unrelated (orthogonal axes). GVSM lets term vectors g_i overlap:

    d_j = Σ_i w_ij g_i          (slide 16)
    sim(q, d) = qᵀ G d / (√(qᵀ G q) · √(dᵀ G d))       G = term correlation, g_i · g_k

G is estimated from co-occurrence: two terms correlate if they share documents, G = AAᵀ with unit rows.

exact_similarity works from the full G, for small examples. Over a large collection every document's
dᵀGd cannot be computed, so rank() multiplies out G q for the query's closest related terms and keeps
VSM's ||d||: the same score, approximated where the full G is out of reach.
"""
import math

import numpy as np
import scipy.sparse as sp

from week7.lsi import term_document_matrix


def exact_similarity(q, d, G):
    """sim(q, d) with the full matrix G. With G = identity this is ordinary cosine (slide 7)."""
    q, d = np.asarray(q, float), np.asarray(d, float)
    return float(q @ G @ d / (np.sqrt(q @ G @ q) * np.sqrt(d @ G @ d)))


def correlation_matrix(A):
    """G from a term-document matrix: cosine between the term rows."""
    A = np.asarray(A, float)
    unit = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return unit @ unit.T


class GVSM:
    def __init__(self, rankers, sample=50_000, neighbours=10, seed=0):
        self.r, self.index, self.neighbours = rankers, rankers.index, neighbours
        path = rankers.c.folder / f"gvsm_{rankers.track}.npz"
        if path.exists():
            saved = np.load(path)
            self.kept = saved["kept"]
            self.Dt = sp.csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=tuple(saved["shape"]))
        else:
            At, self.kept, _ = term_document_matrix(self.index)
            rows = np.random.default_rng(seed).permutation(At.shape[0])[:sample]
            D = At[np.sort(rows)]
            norms = np.sqrt(np.asarray(D.multiply(D).sum(axis=0)).ravel())
            scale = sp.diags(1.0 / np.maximum(norms, 1e-9))
            self.Dt = (D @ scale).T.tocsr()          # unit term rows x sampled documents
            unfinished = path.with_suffix(".tmp.npz")
            np.savez(unfinished, kept=self.kept, data=self.Dt.data, indices=self.Dt.indices, indptr=self.Dt.indptr,
                     shape=np.array(self.Dt.shape))
            unfinished.replace(path)
        self.position = {int(t): i for i, t in enumerate(self.kept)}
        self.terms = rankers.c.terms

    def related(self, term):
        """Row of G for one term: its most correlated terms, [(term, g)], itself included."""
        t = self.index.vocab.get(term)
        i = None if t is None else self.position.get(t)
        if i is None:
            return [(term, 1.0)]
        row = np.asarray((self.Dt @ self.Dt[i].T).todense()).ravel()
        row[i] = 1.0
        order = np.argpartition(-row, min(self.neighbours, len(row) - 1))[:self.neighbours + 1]
        return [(self.terms[int(self.kept[j])], float(row[j])) for j in order[np.argsort(-row[order])] if row[j] > 0]

    def expanded_query(self, query_terms):
        """G q: each query word spreads its weight onto correlated words."""
        out = {}
        for term in query_terms:
            df = self.index.df(term)
            idf = math.log10(self.r.N / df) if df else 0.0
            for other, g in self.related(term):
                out[other] = out.get(other, 0.0) + g * idf
        return out

    def rank(self, query_terms, depth=1000):
        expanded = self.expanded_query(query_terms)
        q_norm = math.sqrt(sum(w * w for w in expanded.values()))
        norms = self.r.vsm_norms()
        score = np.zeros(self.index.N)
        for term, weight in expanded.items():
            docs, tfs = self.index.postings(term)
            if len(docs):
                score[docs] += weight * tfs * math.log10(self.r.N / len(docs))
        hit = score > 0
        score[hit] /= q_norm * norms[hit]
        ids = np.flatnonzero(hit)
        if len(ids) > depth:
            ids = ids[np.argpartition(-score[ids], depth - 1)[:depth]]
        return ids[np.lexsort((ids, -score[ids]))]
