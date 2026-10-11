"""Week 5 — Ranked evaluation metrics (extends Week 3's set-based precision/recall).

relevant: {doc_id: grade}, grade 2 = highly relevant, 1 = partially relevant.
"""
import math


def precision_at_k(ranking, relevant, k):
    return sum(1 for d in ranking[:k] if d in relevant) / k


def recall_at_k(ranking, relevant, k):
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant) if relevant else 0.0


def average_precision(ranking, relevant):
    """Mean of precision at each rank where a relevant doc appears; missing relevant docs count as 0."""
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def ndcg_at_k(ranking, relevant, k):
    """Graded: gain = 2^grade − 1, discounted by log2(rank + 1)."""
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranking[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0
