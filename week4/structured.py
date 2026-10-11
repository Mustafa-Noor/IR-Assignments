"""Week 4 — Structured text retrieval (slides 11–17).

Concepts
    Node       a structural component of a paper: title, abstract, introduction, methodology, ...
    Region     a continuous part of text — here, a chunk of ~50 words (whole sentences) inside one node
    Match point a position where a query term matches the text

Models
    Field search          search only chosen fields (e.g. only methodology), ranked with field-weighted BM25
    Overlapped lists      a region qualifies only if ALL query terms occur together in it   -> high precision
    Non-overlapped lists  each term is looked up in its own list of regions; results are combined
                          even if the terms sit in different regions                       -> high coverage
    Proximal nodes        a term found in the anchor node counts fully, a term found in a node
                          connected to it counts λ, a term in an unconnected node does not count
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field as dc_field

from data import FIELDS
from week1.index import InvertedIndex
from week1.preprocess import preprocess
from week4.probabilistic import BM25

DEFAULT_FIELD_WEIGHTS = {"title": 3.0, "abstract": 2.0, "introduction": 1.0, "methodology": 1.0,
                         "results": 1.0, "conclusion": 1.0, "references": 0.3}

# Slide 17: structural links between sections of a research paper.
NODE_EDGES = [
    ("title", "abstract"),
    ("abstract", "introduction"), ("abstract", "methodology"),
    ("introduction", "methodology"), ("introduction", "results"),
    ("methodology", "results"), ("methodology", "references"),
    ("results", "references"), ("results", "conclusion"),
]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Region:
    rid: int
    doc_id: str
    field: str
    n: int                 # position of the region inside its field
    text: str
    terms: set = dc_field(default_factory=set)

    @property
    def label(self):
        return f"{self.doc_id}/{self.field}#{self.n + 1}"


def split_regions(text, words=50):
    """Group whole sentences into non-overlapping chunks of roughly `words` words."""
    chunks, current = [], []
    for sentence in _SENTENCE.split(text):
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


class StructuredIndex:
    def __init__(self, corpus, whole_index, region_words=50, k1=1.2, b=0.75):
        self.corpus = {d["id"]: d for d in corpus}
        self.whole = whole_index                         # used for collection-level IDF
        self.k1, self.b = k1, b
        self.fields = {f: InvertedIndex({d["id"]: d.get(f, "") for d in corpus}) for f in FIELDS}
        self._idf = BM25(whole_index).idf

        self.regions = []
        self.region_postings = defaultdict(set)          # term -> {region ids}
        for d in corpus:
            for f in FIELDS:
                for n, chunk in enumerate(split_regions(d.get(f, ""), region_words)):
                    reg = Region(len(self.regions), d["id"], f, n, chunk, set(preprocess(chunk)))
                    self.regions.append(reg)
                    for t in reg.terms:
                        self.region_postings[t].add(reg.rid)

        self.neighbors = defaultdict(set)
        for a, c in NODE_EDGES:
            self.neighbors[a].add(c)
            self.neighbors[c].add(a)

    # ---------- field-level BM25 ----------

    def node_term_score(self, term, doc_id, f):
        """BM25 contribution of one term inside one node (field)."""
        idx = self.fields[f]
        tf = idx.tf(term, doc_id)
        if not tf:
            return 0.0
        norm = 1 - self.b + self.b * idx.doc_len[doc_id] / (idx.avg_len or 1)
        return self._idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def field_bm25(self, query, weights=None, top_k=None):
        """Field-weighted BM25 (simplified BM25F): per-field normalized tf, weighted, then saturated once.

        Returns [(doc_id, score, matched_fields)]. Fields with weight 0 are not searched at all,
        which is how "search only the methodology section" works.
        """
        weights = weights or DEFAULT_FIELD_WEIGHTS
        terms = preprocess(query)
        scores, matched = defaultdict(float), defaultdict(set)
        for t in set(terms):
            pseudo_tf = defaultdict(float)
            for f, w in weights.items():
                if w <= 0:
                    continue
                idx = self.fields[f]
                for doc_id, tf in idx.lookup(t).items():
                    norm = 1 - self.b + self.b * idx.doc_len[doc_id] / (idx.avg_len or 1)
                    pseudo_tf[doc_id] += w * tf / norm
                    matched[doc_id].add(f)
            for doc_id, ptf in pseudo_tf.items():
                scores[doc_id] += self._idf(t) * ptf * (self.k1 + 1) / (ptf + self.k1)
        ranked = sorted(((d, s, sorted(matched[d], key=FIELDS.index)) for d, s in scores.items()),
                        key=lambda x: (-x[1], x[0]))
        return ranked[:top_k] if top_k else ranked

    def search_fields(self, query, fields, top_k=None):
        """Structured search restricted to the given fields (each weighted 1)."""
        return self.field_bm25(query, {f: (1.0 if f in fields else 0.0) for f in FIELDS}, top_k)

    # ---------- region models ----------

    def term_lists(self, query):
        """One list of regions per query term (the 'lists' of slides 14–15)."""
        return {t: sorted(self.region_postings.get(t, ())) for t in dict.fromkeys(preprocess(query))}

    def overlapped(self, query, top_k=None):
        """Regions containing ALL query terms; documents ranked by how many such regions they have."""
        lists = self.term_lists(query)
        if not lists:
            return []
        common = set.intersection(*(set(v) for v in lists.values()))
        by_doc = defaultdict(list)
        for rid in common:
            by_doc[self.regions[rid].doc_id].append(self.regions[rid])
        ranked = sorted(((d, float(len(regs)), sorted(regs, key=lambda r: r.rid)) for d, regs in by_doc.items()),
                        key=lambda x: (-x[1], x[0]))
        return ranked[:top_k] if top_k else ranked

    def non_overlapped(self, query, top_k=None):
        """Union of every term's region list. Documents ranked by query-term coverage
        (how many distinct query terms appear anywhere in their regions), then by matched regions."""
        lists = self.term_lists(query)
        if not lists:
            return []
        covered, regs = defaultdict(set), defaultdict(set)
        for t, rids in lists.items():
            for rid in rids:
                d = self.regions[rid].doc_id
                covered[d].add(t)
                regs[d].add(rid)
        n_terms = len(lists)
        ranked = [(d, len(covered[d]) / n_terms, [self.regions[r] for r in sorted(regs[d])]) for d in covered]
        ranked.sort(key=lambda x: (-x[1], -len(x[2]), x[0]))   # coverage first, then matched regions
        return ranked[:top_k] if top_k else ranked

    # ---------- proximal nodes ----------

    def proximal(self, query, lam=0.7, top_k=None):
        """For each anchor node n:  score(n) = Σ_t max( s(t, n), λ · max_{m ∈ N(n)} s(t, m) ).
        A document's score is its best anchor. Returns [(doc_id, score, [anchor, *contributing neighbors])]."""
        terms = list(dict.fromkeys(preprocess(query)))
        candidates = set().union(*(self.whole.lookup(t) for t in terms)) if terms else set()
        ranked = []
        for d in candidates:
            s = {(t, f): self.node_term_score(t, d, f) for t in terms for f in FIELDS}
            best = (0.0, None, set())
            for anchor in FIELDS:
                total, used = 0.0, set()
                for t in terms:
                    own = s[(t, anchor)]
                    nb_field, nb = max(((m, s[(t, m)]) for m in self.neighbors[anchor]),
                                       key=lambda x: x[1], default=(None, 0.0))
                    if own >= lam * nb:
                        total += own
                    else:
                        total += lam * nb
                        used.add(nb_field)
                if total > best[0]:
                    best = (total, anchor, used)
            if best[1]:
                ranked.append((d, best[0], [best[1]] + sorted(best[2], key=FIELDS.index)))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[:top_k] if top_k else ranked
