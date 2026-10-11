"""Week 4 — Probabilistic retrieval (slides 3–10).

Probability Ranking Principle: rank documents by decreasing P(Relevance | Q, D).

Binary Independence Model (BIM)
    Terms are only present (1) or absent (0). Each present query term adds its
    Robertson–Sparck Jones weight:
        w_t = log( (r + 0.5)(N - df - R + r + 0.5) / ((R - r + 0.5)(df - r + 0.5)) )
    With no relevance information (R = r = 0) this reduces to
        w_t = log( (N - df + 0.5) / (df + 0.5) )      — can go negative for very common terms.

BM25 ("a smarter BIM")
    score(q, d) = Σ_t IDF(t) · tf·(k1 + 1) / (tf + k1·(1 − b + b·|d|/avgdl))
    IDF(t)      = log(1 + (N − df + 0.5)/(df + 0.5))   (always positive)
    k1 controls term-frequency saturation, b controls document-length normalization.

Score → probability (slide 7):  P = 1 / (1 + e^(−s))
"""
import math

from week1.preprocess import preprocess


def sigmoid(s):
    return 1.0 / (1.0 + math.exp(-s))


def to_probabilities(ranked, calibrate=True):
    """Map [(doc, score)] to [(doc, score, P)].

    Raw BM25 scores are large (5–30), so the slide's sigmoid alone pushes every P to ≈1.0.
    calibrate=True first standardizes scores over the result list: s' = (s − mean)/std.
    """
    if not ranked:
        return []
    scores = [s for _, s in ranked]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)) or 1.0
    return [(d, s, sigmoid((s - mean) / std if calibrate else s)) for d, s in ranked]


class BIM:
    def __init__(self, index):
        self.index = index

    def weight(self, term, R=0, r=0):
        N, df = self.index.N, self.index.df(term)
        return math.log(((r + 0.5) * (N - df - R + r + 0.5)) / ((R - r + 0.5) * (df - r + 0.5)))

    def rank(self, query, relevant_feedback=None, top_k=None):
        """relevant_feedback: optional set of doc_ids known to be relevant (relevance feedback)."""
        terms = set(preprocess(query))
        fb = set(relevant_feedback or ())
        weights = {}
        for t in terms:
            r = sum(1 for d in fb if self.index.tf(t, d))
            weights[t] = self.weight(t, R=len(fb), r=r)
        candidates = set().union(*(self.index.lookup(t) for t in terms)) if terms else set()
        scored = [(d, sum(w for t, w in weights.items() if self.index.tf(t, d))) for d in candidates]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k] if top_k else scored


class BM25:
    def __init__(self, index, k1=1.2, b=0.75):
        self.index, self.k1, self.b = index, k1, b

    def idf(self, term):
        N, df = self.index.N, self.index.df(term)
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    def term_score(self, term, doc_id):
        tf = self.index.tf(term, doc_id)
        if not tf:
            return 0.0
        norm = 1 - self.b + self.b * self.index.doc_len[doc_id] / self.index.avg_len
        return self.idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def score(self, query_terms, doc_id):
        return sum(self.term_score(t, doc_id) for t in query_terms)

    def rank(self, query, top_k=None):
        terms = preprocess(query)
        candidates = set().union(*(self.index.lookup(t) for t in terms)) if terms else set()
        scored = [(d, self.score(terms, d)) for d in candidates]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k] if top_k else scored
