"""Week 3 — Classification and clustering in IR (slides 16–17).

Classification (supervised): Multinomial Naive Bayes assigns a paper to a known category.
Clustering (unsupervised):   K-Means on TF-IDF vectors discovers groups without labels.
"""
import os
from collections import Counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # avoids a noisy joblib core-count probe on Windows

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from week1.preprocess import preprocess


def _vectorizer(kind):
    cls = TfidfVectorizer if kind == "tfidf" else CountVectorizer
    return cls(analyzer=preprocess)


def classify_cv(texts, labels, folds=5, seed=0):
    """Cross-validated Naive Bayes: every paper is predicted by a model that never saw it."""
    model = make_pipeline(_vectorizer("count"), MultinomialNB())
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, texts, labels, cv=cv)
    classes = sorted(set(labels))
    return {
        "accuracy": accuracy_score(labels, pred),
        "predictions": list(pred),
        "classes": classes,
        "confusion": confusion_matrix(labels, pred, labels=classes),
    }


def train_classifier(texts, labels):
    model = make_pipeline(_vectorizer("count"), MultinomialNB())
    model.fit(texts, labels)
    return model


def cluster(texts, labels, k=6, seed=0):
    """K-Means on TF-IDF; purity = share of papers in their cluster's majority category."""
    vec = _vectorizer("tfidf")
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    assign = km.labels_
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    majority_total = 0
    for c in range(k):
        members = [i for i, a in enumerate(assign) if a == c]
        cats = Counter(labels[i] for i in members)
        majority_total += cats.most_common(1)[0][1] if cats else 0
        top_terms = terms[np.argsort(km.cluster_centers_[c])[::-1][:8]].tolist()
        clusters.append({"cluster": c, "size": len(members), "categories": dict(cats), "top_terms": top_terms,
                         "members": members})
    return {"purity": majority_total / len(texts), "assignments": assign.tolist(), "clusters": clusters}
