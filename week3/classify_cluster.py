"""Week 3 — Classification and clustering in IR (slides 16–17).

Classification (supervised): Multinomial Naive Bayes assigns a paper to a known category.
Clustering (unsupervised):   K-Means on TF-IDF vectors discovers groups without labels.
"""
from collections import Counter

import numpy as np

from week1.preprocess import preprocess
from week3.kmeans import kmeans, purity, tfidf_matrix
from week3.naive_bayes import NaiveBayes, cross_validate


def accuracy(labels, predictions):
    return sum(t == p for t, p in zip(labels, predictions)) / len(labels)


def confusion(labels, predictions, classes):
    """Rows = true class, columns = predicted class."""
    position = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(labels, predictions):
        matrix[position[t], position[p]] += 1
    return matrix


def classify_cv(texts, labels, folds=5, seed=0):
    """Cross-validated Naive Bayes: every paper is predicted by a model that never saw it."""
    pred = cross_validate(texts, labels, folds, seed)
    classes = sorted(set(labels))
    return {
        "accuracy": accuracy(labels, pred),
        "predictions": pred,
        "classes": classes,
        "confusion": confusion(labels, pred, classes),
    }


def train_classifier(texts, labels):
    return NaiveBayes().fit(texts, labels)


def cluster(texts, labels, k=6, seed=0):
    """K-Means on TF-IDF; purity = share of papers in their cluster's majority category."""
    X, terms = tfidf_matrix([preprocess(t) for t in texts])
    assign, centroids, _ = kmeans(X, k, n_init=50, seed=seed)
    terms = np.array(terms)
    clusters = []
    for c in range(k):
        members = [i for i, a in enumerate(assign) if a == c]
        top_terms = terms[np.argsort(centroids[c])[::-1][:8]].tolist()
        clusters.append({"cluster": c, "size": len(members), "categories": dict(Counter(labels[i] for i in members)),
                         "top_terms": top_terms, "members": members})
    return {"purity": purity(assign, labels), "assignments": assign.tolist(), "clusters": clusters}
