"""Week 3 — Vector Space Model (slides 10–14).

TF(t, d)     = number of times t appears in d
IDF(t)       = log10(N / df(t))
TF-IDF(t, d) = TF(t, d) * IDF(t)
cos(q, d)    = (q · d) / (||q|| ||d||)
"""
import math
from collections import Counter

from week1.preprocess import preprocess


class VectorSpaceModel:
    def __init__(self, docs, weighting="tf-idf", analyzer=preprocess):
        """docs: {doc_id: text}; weighting: 'tf-idf' or 'binary' (slide 14 uses binary)."""
        self.analyzer = analyzer
        self.weighting = weighting
        tfs = {d: Counter(analyzer(text)) for d, text in docs.items()}
        self.N = len(docs)
        self.df = Counter(t for c in tfs.values() for t in c)
        self.vocabulary = sorted(self.df)
        self.vectors = {d: self._weigh(c) for d, c in tfs.items()}
        self.norms = {d: math.sqrt(sum(w * w for w in v.values())) for d, v in self.vectors.items()}

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log10(self.N / df) if df else 0.0

    def _weigh(self, counts):
        if self.weighting == "binary":
            return {t: 1.0 for t in counts}
        return {t: tf * self.idf(t) for t, tf in counts.items() if self.idf(t) > 0}

    def query_vector(self, query):
        return self._weigh(Counter(self.analyzer(query)))

    def cosine(self, qv, doc_id):
        dv = self.vectors[doc_id]
        dot = sum(w * dv.get(t, 0.0) for t, w in qv.items())
        qn = math.sqrt(sum(w * w for w in qv.values()))
        dn = self.norms[doc_id]
        return dot / (qn * dn) if qn and dn else 0.0

    def rank(self, query, top_k=None):
        """[(doc_id, cosine)] best first, documents with score 0 dropped."""
        qv = self.query_vector(query)
        scored = [(d, self.cosine(qv, d)) for d in self.vectors]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k] if top_k else scored

    def dense(self, doc_id, terms):
        """Vector as a list in the given term order (for display)."""
        v = self.vectors[doc_id]
        return [round(v.get(t, 0.0), 3) for t in terms]
