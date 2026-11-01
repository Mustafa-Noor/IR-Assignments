"""Week 3 — Multinomial Naive Bayes, written from the formula (slide 16).

    ĉ = argmax_c [ log P(c) + Σ_t tf(t, d) · log P(t | c) ]
    P(c)     = N_c / N                                 share of training documents in class c
    P(t | c) = (count(t, c) + α) / (Σ_v count(v, c) + α·|V|)   Laplace smoothing, α = 1

Words never seen in training are ignored at prediction time.
"""
from collections import Counter

import numpy as np

from week1.preprocess import preprocess


class NaiveBayes:
    def __init__(self, alpha=1.0, analyzer=preprocess):
        self.alpha, self.analyzer = alpha, analyzer

    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        class_id = {c: i for i, c in enumerate(self.classes)}
        counts = [Counter(self.analyzer(t)) for t in texts]
        self.vocab = {t: i for i, t in enumerate(sorted({t for c in counts for t in c}))}
        tf = np.zeros((len(self.classes), len(self.vocab)))
        priors = np.zeros(len(self.classes))
        for c, label in zip(counts, labels):
            priors[class_id[label]] += 1
            for t, n in c.items():
                tf[class_id[label], self.vocab[t]] += n
        self.log_prior = np.log(priors / priors.sum())
        smoothed = tf + self.alpha
        self.log_likelihood = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        return self

    def joint_log_likelihood(self, text):
        """log P(c) + Σ tf·log P(t|c), for every class."""
        scores = self.log_prior.copy()
        for t, n in Counter(self.analyzer(text)).items():
            i = self.vocab.get(t)
            if i is not None:
                scores += n * self.log_likelihood[:, i]
        return scores

    def predict(self, texts):
        return [self.classes[int(np.argmax(self.joint_log_likelihood(t)))] for t in texts]


def stratified_folds(labels, folds=5, seed=0):
    """Fold number per document, keeping each class spread evenly across folds."""
    labels = np.asarray(labels)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    encoded = np.unique(first, return_inverse=True)[1][inverse]      # classes in order of first appearance
    ordered = np.sort(encoded)
    allocation = np.array([np.bincount(ordered[i::folds], minlength=len(first)) for i in range(folds)])
    rng = np.random.RandomState(seed)
    assigned = np.empty(len(labels), dtype=int)
    for c in range(len(first)):
        members = np.arange(folds).repeat(allocation[:, c])
        rng.shuffle(members)
        assigned[encoded == c] = members
    return assigned


def cross_validate(texts, labels, folds=5, seed=0):
    """Every document is predicted by a model that never saw it."""
    assigned = stratified_folds(labels, folds, seed)
    predictions = [None] * len(texts)
    for f in range(folds):
        train = [i for i in range(len(texts)) if assigned[i] != f]
        test = [i for i in range(len(texts)) if assigned[i] == f]
        model = NaiveBayes().fit([texts[i] for i in train], [labels[i] for i in train])
        for i, p in zip(test, model.predict([texts[i] for i in test])):
            predictions[i] = p
    return predictions
