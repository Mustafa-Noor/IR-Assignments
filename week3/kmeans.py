"""Week 3 — K-Means clustering and TF-IDF vectors, written from the definition (slide 17).

    minimize  J = Σ_c Σ_{x ∈ c} ||x − μ_c||²       μ_c = mean of the points in cluster c

Lloyd's algorithm alternates two steps until the assignments stop changing:
    assign   every point to its nearest centroid
    update   every centroid to the mean of its points

Starting centroids use k-means++: each new centroid is picked with probability proportional to the
squared distance from the nearest centroid so far, which spreads them out.
"""
from collections import Counter

import numpy as np


def tfidf_matrix(token_lists):
    """L2-normalized TF-IDF rows, idf = ln((1 + N)/(1 + df)) + 1. Returns (matrix, terms)."""
    counts = [Counter(tokens) for tokens in token_lists]
    terms = sorted({t for c in counts for t in c})
    index = {t: i for i, t in enumerate(terms)}
    X = np.zeros((len(counts), len(terms)))
    for row, c in enumerate(counts):
        for t, n in c.items():
            X[row, index[t]] = n
    idf = np.log((1 + len(counts)) / (1 + (X > 0).sum(axis=0))) + 1
    X *= idf
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X, terms


def squared_distances(X, centroids):
    return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) if len(X) < 256 else _in_blocks(X, centroids)


def _in_blocks(X, centroids):
    out = np.empty((len(X), len(centroids)))
    for start in range(0, len(X), 512):
        block = X[start:start + 512]
        out[start:start + 512] = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return out


def plus_plus(X, k, rng):
    """Greedy k-means++: try a few candidates per step and keep the one that lowers J the most."""
    trials = 2 + int(np.log(k))
    centroids = [X[rng.integers(len(X))]]
    nearest = ((X - centroids[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = nearest.sum()
        candidates = rng.choice(len(X), trials, p=nearest / total) if total > 0 else rng.integers(len(X), size=trials)
        options = [np.minimum(nearest, ((X - X[c]) ** 2).sum(axis=1)) for c in candidates]
        best = int(np.argmin([o.sum() for o in options]))
        centroids.append(X[candidates[best]])
        nearest = options[best]
    return np.array(centroids)


def lloyd(X, centroids, max_iter=100):
    assign = None
    for _ in range(max_iter):
        distances = squared_distances(X, centroids)
        new_assign = distances.argmin(axis=1)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(len(centroids)):
            members = X[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    inertia = squared_distances(X, centroids)[np.arange(len(X)), assign].sum()
    return assign, centroids, float(inertia)


def kmeans(X, k, n_init=10, seed=0):
    """Best of n_init runs by J. Returns (assignments, centroids, J)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        result = lloyd(X, plus_plus(X, k, rng))
        if best is None or result[2] < best[2]:
            best = result
    return best


def purity(assignments, labels):
    """Share of documents that belong to the majority label of their cluster."""
    clusters = {}
    for a, label in zip(assignments, labels):
        clusters.setdefault(a, Counter())[label] += 1
    return sum(c.most_common(1)[0][1] for c in clusters.values()) / len(labels)
