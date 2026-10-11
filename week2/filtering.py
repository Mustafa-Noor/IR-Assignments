"""Week 2 — Ad-hoc retrieval vs filtering (slides 17–19).

Ad-hoc:    dynamic query, static collection, the user pulls results.
Filtering: static user profile, documents arrive as a stream, the system pushes
           any document whose score passes a threshold.
"""
import math
from collections import Counter

from week1.preprocess import preprocess


class FilteringEngine:
    """Matches a fixed profile against a stream of incoming documents.

    Document frequencies are learned incrementally from what has streamed in so far,
    because a filtering system never sees the whole collection up front.
    """

    def __init__(self, profile_text, threshold=0.15):
        self.profile = Counter(preprocess(profile_text))
        self.threshold = threshold
        self.seen = 0
        self.df = Counter()

    def score(self, tokens):
        """Cosine between the profile and the document, using IDF from the stream so far."""
        tf = Counter(tokens)
        idf = {t: math.log10((self.seen + 1) / (self.df[t] + 1)) + 1 for t in set(tf) | set(self.profile)}
        doc_vec = {t: c * idf[t] for t, c in tf.items()}
        prof_vec = {t: c * idf[t] for t, c in self.profile.items()}
        dot = sum(v * doc_vec.get(t, 0) for t, v in prof_vec.items())
        norm = math.sqrt(sum(v * v for v in doc_vec.values())) * math.sqrt(sum(v * v for v in prof_vec.values()))
        return dot / norm if norm else 0.0

    def receive(self, doc_id, text):
        """Process one arriving document; return (score, delivered?)."""
        tokens = preprocess(text)
        self.seen += 1
        self.df.update(set(tokens))
        s = self.score(tokens)
        return s, s >= self.threshold


def simulate_stream(docs_in_arrival_order, profile_text, threshold=0.15):
    """docs: [(doc_id, text)] -> [{doc_id, step, score, delivered}]"""
    engine = FilteringEngine(profile_text, threshold)
    log = []
    for step, (doc_id, text) in enumerate(docs_in_arrival_order, 1):
        s, delivered = engine.receive(doc_id, text)
        log.append({"step": step, "doc_id": doc_id, "score": round(s, 4), "delivered": delivered})
    return log
