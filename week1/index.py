"""Week 1 — Inverted index (slide 12): term -> postings list of documents containing it."""
from collections import Counter, defaultdict

from data import full_text
from week1.preprocess import preprocess


class InvertedIndex:
    """postings[term] = {doc_id: term frequency}."""

    def __init__(self, docs):
        """docs: {doc_id: text}"""
        self.postings = defaultdict(dict)
        self.doc_len = {}
        for doc_id, text in docs.items():
            counts = Counter(preprocess(text))
            self.doc_len[doc_id] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term][doc_id] = tf
        self.doc_ids = list(docs)
        self.N = len(docs)
        self.avg_len = sum(self.doc_len.values()) / max(self.N, 1)

    @classmethod
    def from_corpus(cls, corpus):
        return cls({d["id"]: full_text(d) for d in corpus})

    @property
    def vocabulary(self):
        return sorted(self.postings)

    def df(self, term):
        return len(self.postings.get(term, {}))

    def tf(self, term, doc_id):
        return self.postings.get(term, {}).get(doc_id, 0)

    def lookup(self, term):
        """Postings for an already-preprocessed term."""
        return self.postings.get(term, {})
