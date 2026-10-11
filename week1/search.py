"""Week 1 — Searching, then ranking (slides 13, 14, 17).

Searching: union of postings for the query terms -> candidate set.
Ranking:   the more query terms a document shares, the higher it ranks
           (ties broken by total occurrences of those terms).
"""
from week1.preprocess import preprocess


def search(index, query):
    """Candidate documents containing at least one query term."""
    candidates = set()
    for term in preprocess(query):
        candidates |= set(index.lookup(term))
    return candidates


def rank_overlap(index, query, top_k=None):
    """[(doc_id, shared_terms, total_tf)] sorted best first."""
    terms = set(preprocess(query))
    scored = []
    for doc_id in search(index, query):
        shared = [t for t in terms if index.tf(t, doc_id)]
        total = sum(index.tf(t, doc_id) for t in shared)
        scored.append((doc_id, len(shared), total))
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored[:top_k] if top_k else scored


def pipeline_trace(index, query, top_k=5):
    """Slide 14: Original query -> preprocessing -> index lookup -> ranking -> results."""
    terms = preprocess(query)
    return {
        "original": query,
        "preprocessed": terms,
        "lookup": {t: sorted(index.lookup(t)) for t in terms},
        "ranking": rank_overlap(index, query, top_k),
    }
