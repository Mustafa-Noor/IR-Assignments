"""Week 3 — Basic evaluation with precision and recall (slide 18, item 7).

Precision = |retrieved ∩ relevant| / |retrieved|
Recall    = |retrieved ∩ relevant| / |relevant|
"""


def precision_recall_f1(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    hit = len(retrieved & relevant)
    p = hit / len(retrieved) if retrieved else 0.0
    r = hit / len(relevant) if relevant else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
