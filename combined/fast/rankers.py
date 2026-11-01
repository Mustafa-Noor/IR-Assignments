"""Weeks 1–4 models on the compact index. Same formulas as the original classes, written over arrays.

Each one builds a score array over all N documents, adding one query term's postings at a time, so a
query costs milliseconds even on 1.49M dialogues. tests/test_fast_models.py checks that every model here
ranks exactly like the original class on the same documents.

Rankers return document ids best first, up to `depth`. Ties break by document id, as in Weeks 1–4.
"""
import math

import numpy as np

from combined.collection import FIELDS
from week1.preprocess import preprocess
from week3.boolean import BooleanQueryError, to_postfix

DEPTH = 1000


def top(scores, mask, depth=DEPTH):
    """Document ids with mask true, by descending score then ascending id."""
    ids = np.flatnonzero(mask)
    if depth and len(ids) > depth:
        s = scores[ids]
        cut = np.partition(s, len(s) - depth)[len(s) - depth]
        ids = ids[s >= cut]
    return ids[np.argsort(-scores[ids], kind="stable")][:depth or None]


def unique_terms(query):
    return list(dict.fromkeys(preprocess(query)))


class Rankers:
    def __init__(self, collection, track="all"):
        self.c, self.track = collection, track
        self.index = collection.index(track)
        self.N = int(np.count_nonzero(self.index.doc_len))
        self.fields = FIELDS if track == "all" else FIELDS[1:]
        self._norms = {}

    def _zeros(self):
        return np.zeros(self.index.N, np.float64), np.zeros(self.index.N, bool)

    # ---------- Week 1 ----------

    def overlap(self, query, depth=DEPTH):
        """Most shared query terms first, ties by total occurrences."""
        shared, total = np.zeros(self.index.N, np.int32), np.zeros(self.index.N, np.int64)
        for t in set(preprocess(query)):
            docs, tfs = self.index.postings(t)
            shared[docs] += 1
            total[docs] += tfs
        return top(shared * 1e7 + total, shared > 0, depth)

    # ---------- Week 2 ----------

    def weighted_sum(self, query, depth=DEPTH):
        """R(q, d) = Σ_t tf · log10(N/df), counting a repeated query term each time, as in Week 2."""
        score, hit = self._zeros()
        for t in preprocess(query):
            docs, tfs = self.index.postings(t)
            if len(docs):
                score[docs] += tfs * math.log10(self.N / len(docs))
                hit[docs] = True
        return top(score, hit, depth)

    # ---------- Week 3 ----------

    def boolean_mask(self, query):
        """Rel(D, Q) as a true/false array, evaluating the postfix form on a stack."""
        universe = self.index.doc_len > 0
        stack = []
        for tok in to_postfix(query):
            if isinstance(tok, tuple):
                stems = preprocess(tok[1])
                mask = universe.copy()
                for s in stems:
                    current = np.zeros(self.index.N, bool)
                    current[self.index.postings(s)[0]] = True
                    mask &= current
                stack.append(mask)
            elif tok == "NOT":
                if not stack:
                    raise BooleanQueryError("NOT needs an operand")
                stack.append(universe & ~stack.pop())
            else:
                if len(stack) < 2:
                    raise BooleanQueryError(f"{tok} needs two operands")
                b, a = stack.pop(), stack.pop()
                stack.append(a & b if tok == "AND" else a | b)
        if len(stack) != 1:
            raise BooleanQueryError("Malformed query")
        return stack[0]

    def boolean(self, query, depth=None):
        return np.flatnonzero(self.boolean_mask(query))[:depth]

    def _idf(self, df):
        return math.log10(self.N / df) if df else 0.0

    def vsm_norms(self):
        """||d|| for every document: Σ_t (tf · idf)², one term at a time."""
        path = self.c.folder / f"vsm_norms_{self.track}.npy"
        if "vsm" not in self._norms and path.exists():
            self._norms["vsm"] = np.load(path)
        if "vsm" not in self._norms:
            squared = np.zeros(self.index.N)
            ptr = np.asarray(self.index.ptr)
            for t in range(len(ptr) - 1):
                a, b = int(ptr[t]), int(ptr[t + 1])
                idf = self._idf(b - a)
                if idf > 0:
                    squared[self.index.docs[a:b]] += (self.index.tfs[a:b].astype(np.float64) * idf) ** 2
            self._norms["vsm"] = np.sqrt(squared)
            np.save(path, self._norms["vsm"])
        return self._norms["vsm"]

    def vsm(self, query, depth=DEPTH):
        """cos(q, d) = q·d / (||q|| ||d||) with TF-IDF weights."""
        weights = {}
        for t in preprocess(query):
            idf = self._idf(self.index.df(t))
            if idf > 0:
                weights[t] = weights.get(t, 0.0) + idf
        q_norm = math.sqrt(sum(w * w for w in weights.values()))
        score, hit = self._zeros()
        norms = self.vsm_norms()
        for t, w in weights.items():
            docs, tfs = self.index.postings(t)
            score[docs] += w * tfs * self._idf(len(docs))
            hit[docs] = True
        if q_norm:
            score[hit] /= q_norm * norms[hit]
        return top(score, hit & (score > 0), depth)

    # ---------- Week 4: probabilistic ----------

    def bim_weight(self, df, R=0, r=0):
        return math.log(((r + 0.5) * (self.N - df - R + r + 0.5)) / ((R - r + 0.5) * (df - r + 0.5)))

    def bim(self, query, depth=DEPTH):
        """Σ of Robertson–Sparck Jones weights over the query terms present. Weights may be negative."""
        score, hit = self._zeros()
        for t in set(preprocess(query)):
            docs = self.index.postings(t)[0]
            if len(docs):
                score[docs] += self.bim_weight(len(docs))
                hit[docs] = True
        return top(score, hit, depth)

    def bm25_idf(self, df):
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def bm25(self, query, k1=1.2, b=0.75, depth=DEPTH):
        """Σ_t IDF(t) · tf·(k1 + 1) / (tf + k1·(1 − b + b·|d|/avgdl))."""
        score, hit = self._zeros()
        norm = 1 - b + b * self.index.doc_len / self.index.avg_len
        for t in preprocess(query):
            docs, tfs = self.index.postings(t)
            if len(docs):
                tf = tfs.astype(np.float64)
                score[docs] += self.bm25_idf(len(docs)) * tf * (k1 + 1) / (tf + k1 * norm[docs])
                hit[docs] = True
        return top(score, hit, depth)

    # ---------- Week 4: structured ----------

    def field_bm25(self, query, weights, k1=1.2, b=0.75, depth=DEPTH):
        """ptf = Σ_f w_f · tf_f / B_f, then saturated once: IDF · ptf·(k1 + 1) / (ptf + k1)."""
        score, hit = self._zeros()
        for t in set(preprocess(query)):
            ptf = np.zeros(self.index.N)
            for f, w in weights.items():
                if w <= 0 or f not in self.fields:
                    continue
                idx = self.c.index(f)
                docs, tfs = idx.postings(t)
                if len(docs):
                    norm = 1 - b + b * idx.doc_len[docs] / (idx.avg_len or 1)
                    ptf[docs] += w * tfs / norm
            found = np.flatnonzero(ptf)
            if len(found):
                idf = self.bm25_idf(self.index.df(t))
                score[found] += idf * ptf[found] * (k1 + 1) / (ptf[found] + k1)
                hit[found] = True
        return top(score, hit, depth)

    def _dialogue_regions(self):
        if "regions" not in self._norms:
            self._norms["regions"] = np.asarray(self.c.region_field) > 0
        return self._norms["regions"]

    def _region_lists(self, query):
        keep = None if self.track == "all" else self._dialogue_regions()
        lists = {}
        region_index = self.c.index("regions")
        for t in dict.fromkeys(preprocess(query)):
            regions = region_index.postings(t)[0]
            lists[t] = regions if keep is None else regions[keep[regions]]
        return lists

    def overlapped(self, query, depth=DEPTH):
        """Regions holding ALL query terms; documents ranked by how many such regions they have."""
        lists = self._region_lists(query)
        if not lists or min(len(v) for v in lists.values()) == 0:
            return np.empty(0, np.int64)
        common = None
        for regions in sorted(lists.values(), key=len):
            common = regions if common is None else np.intersect1d(common, regions, assume_unique=True)
        count = np.bincount(np.asarray(self.c.region_doc)[common], minlength=self.index.N).astype(np.float64)
        return top(count, count > 0, depth)

    def non_overlapped(self, query, depth=DEPTH):
        """Each term searched in its own list; ranked by query-term coverage, then matched regions."""
        lists = self._region_lists(query)
        if not lists:
            return np.empty(0, np.int64)
        region_doc = np.asarray(self.c.region_doc)
        covered, matched = np.zeros(self.index.N), np.zeros(self.index.N)
        seen = []
        for regions in lists.values():
            docs = region_doc[regions]
            covered[np.unique(docs)] += 1
            seen.append(regions)
        unique = np.unique(np.concatenate(seen))
        matched = np.bincount(region_doc[unique], minlength=self.index.N)
        return top(covered * 1e6 + matched, covered > 0, depth)

    def proximal(self, query, edges, lam=0.7, k1=1.2, b=0.75, depth=DEPTH):
        """score(n) = Σ_t max(s(t, n), λ · max_{m ∈ N(n)} s(t, m)); a document scores its best anchor node."""
        fields = self.fields
        neighbours = [[fields.index(m) for m in fields if (a, m) in edges or (m, a) in edges] for a in fields]
        totals = np.zeros((self.index.N, len(fields)), np.float64)
        for t in dict.fromkeys(preprocess(query)):
            whole_df = self.index.df(t)
            if not whole_df:
                continue
            idf = self.bm25_idf(whole_df)
            s = np.zeros((self.index.N, len(fields)), np.float64)
            for i, f in enumerate(fields):
                idx = self.c.index(f)
                docs, tfs = idx.postings(t)
                if len(docs):
                    tf = tfs.astype(np.float64)
                    norm = 1 - b + b * idx.doc_len[docs] / (idx.avg_len or 1)
                    s[docs, i] = idf * tf * (k1 + 1) / (tf + k1 * norm)
            for i, around in enumerate(neighbours):
                nearby = s[:, around].max(axis=1) if around else 0.0
                totals[:, i] += np.maximum(s[:, i], lam * nearby)
        score = totals.max(axis=1)
        return top(score, score > 0, depth)
