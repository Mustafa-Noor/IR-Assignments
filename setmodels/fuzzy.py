"""Week 6 — Fuzzy model (slides 5–9, 16).

Relevance is a degree, not a yes/no:   μ_A(x) ∈ [0, 1]

Membership of document d in query term t combines how close a document word v is to t (partial
matching, so a typo still finds the word) and how strongly d is about v (a trapezoid over its tf):

    μ_t(d) = max_v  sim(t, v) · trap(tf(v, d); a, b, c, d)

Operators (slide 16):
    μ(A AND B) = min(μ_A, μ_B)     μ(A OR B) = max(μ_A, μ_B)     μ(NOT A) = 1 − μ_A

Bands (slide 5):  0.8–1 highly, 0.5–0.8 moderately, 0.2–0.5 slightly relevant, below 0.2 not relevant.
"""
import re
from collections import Counter, defaultdict

import numpy as np

from setmodels.similarity import similarity
from week1.preprocess import preprocess
from week3.boolean import to_postfix

THRESHOLD = 0.2        # below this a document is "not relevant" (slide 5)
MIN_SIMILARITY = 0.7   # how alike two words must be to count as a partial match


def trapezoid(x, a=0.0, b=3.0, c=float("inf"), d=float("inf")):
    """Rises from a to b, flat from b to c, falls from c to d (slide 16)."""
    rising = (x - a) / (b - a)
    falling = (d - x) / (d - c) if d > c else np.full_like(x, 1.0, dtype=np.float64)
    return np.clip(np.minimum(np.minimum(rising, 1.0), falling), 0.0, 1.0)


def band(mu):
    return "highly" if mu >= 0.8 else "moderately" if mu >= 0.5 else "slightly" if mu >= 0.2 else "not"


def trigrams(word):
    padded = f"${word}$"
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class Fuzzy:
    def __init__(self, index, terms, threshold=THRESHOLD):
        self.index, self.terms, self.threshold = index, terms, threshold
        self.by_trigram = defaultdict(list)
        for i, term in enumerate(terms):
            if index.ptr[i + 1] - index.ptr[i] > 1:     # one-off words are mostly names and typos
                for g in trigrams(term):
                    self.by_trigram[g].append(i)

    def expand(self, term, limit=5):
        """Vocabulary words close to the term: [(word, sim)]. Trigrams shortlist before comparing."""
        votes = Counter()
        for g in trigrams(term):
            votes.update(self.by_trigram.get(g, ()))
        shortlist = sorted(votes, key=lambda i: (-votes[i], i))[:40]      # ties by term id, so every run agrees
        scored = [(self.terms[i], similarity(term, self.terms[i])) for i in shortlist]
        matches = sorted((m for m in scored if m[1] >= MIN_SIMILARITY), key=lambda m: -m[1])[:limit]
        if self.index.df(term) and all(w != term for w, _ in matches):
            matches.insert(0, (term, 1.0))
        return matches

    def membership(self, word):
        """μ_t(d) for every document. A stop-word is neutral, as in the Boolean model."""
        stems = preprocess(word)
        if not stems:
            return np.ones(self.index.N, np.float32)
        result = np.ones(self.index.N, np.float32)
        for s in stems:
            mu = np.zeros(self.index.N, np.float32)
            for v, sim in self.expand(s):
                docs, tfs = self.index.postings(v)
                mu[docs] = np.maximum(mu[docs], sim * trapezoid(tfs.astype(np.float64)))
            result = np.minimum(result, mu)
        return result

    def scores(self, query):
        """Evaluate the query's postfix form with fuzzy min, max and 1 − μ."""
        stack = []
        self.evidence = np.zeros(self.index.N, np.float32)
        for tok in to_postfix(re.sub(r"\bBUT\s+NOT\b", "AND NOT", query)):
            if isinstance(tok, tuple):
                stack.append(self.membership(tok[1]))
                self.evidence += stack[-1]
            elif tok == "NOT":
                stack.append(1.0 - stack.pop())
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(np.minimum(a, b) if tok == "AND" else np.maximum(a, b))
        return stack[0]

    def rank(self, query, depth=1000):
        """Retrieved = μ ≥ threshold, best first. min and max give many equal scores, so ties go to
        the document with more total evidence across the query's words."""
        mu = self.scores(query)
        ids = np.flatnonzero((mu >= self.threshold) & (self.index.doc_len > 0))
        return ids[np.lexsort((-self.evidence[ids], -mu[ids]))][:depth]
