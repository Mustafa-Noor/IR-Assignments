"""Week 6 — Extended Boolean model (slides 10–12, 16).

Slides:  OR = union, AND = intersection, NOT / BUT NOT = difference, for example
         (A OR B) AND (NOT C) keeps documents about A or B and removes anything about C.
         That set logic decides which documents are retrieved.

Ranking: set logic alone cannot order its results, so the documents it keeps are ranked with the
         p-norm formulas of Salton, Fox and Wu (1983), which go beyond the slides:

    OR_p (a, b) = ((w_a^p + w_b^p) / 2)^(1/p)
    AND_p(a, b) = 1 − (((1 − w_a)^p + (1 − w_b)^p) / 2)^(1/p)
    NOT(a)      = 1 − w_a

    w(t, d) = tf/(tf + 1) · idf(t)/log10(N)  in [0, 1]

p = 1 treats every term like a plain average; large p approaches fuzzy min and max.
"""
import math
import re

import numpy as np

from week1.preprocess import preprocess
from week3.boolean import to_postfix


def normalize_query(query):
    """BUT NOT means AND NOT (slide 10)."""
    return re.sub(r"\bBUT\s+NOT\b", "AND NOT", query)


def and_p(a, b, p):
    return 1.0 - (((1.0 - a) ** p + (1.0 - b) ** p) / 2.0) ** (1.0 / p)


def or_p(a, b, p):
    return ((a ** p + b ** p) / 2.0) ** (1.0 / p)


class ExtendedBoolean:
    def __init__(self, rankers, p=2.0):
        self.r, self.index, self.p = rankers, rankers.index, p

    def weights(self, word):
        out = np.zeros(self.index.N, np.float32)
        for s in preprocess(word):
            docs, tfs = self.index.postings(s)
            if len(docs):
                idf = math.log10(self.r.N / len(docs)) / math.log10(self.r.N)
                out[docs] = np.maximum(out[docs], tfs / (tfs + 1.0) * idf)
        return out

    def scores(self, query):
        stack = []
        for tok in to_postfix(normalize_query(query)):
            if isinstance(tok, tuple):
                stack.append(self.weights(tok[1]))
            elif tok == "NOT":
                stack.append(1.0 - stack.pop())
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(and_p(a, b, self.p) if tok == "AND" else or_p(a, b, self.p))
        return stack[0]

    def rank(self, query, depth=1000):
        ids = np.flatnonzero(self.r.boolean_mask(normalize_query(query)))
        s = self.scores(query)
        return ids[np.argsort(-s[ids], kind="stable")][:depth]
