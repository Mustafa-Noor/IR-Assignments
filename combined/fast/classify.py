"""Week 3's Naive Bayes and K-Means over the compact index.

SODA has no topic labels, so the only predefined category is `relation` (xReact, xIntent, ...): the kind of
commonsense inference behind a dialogue. It is not a topic, so expect far lower scores than on the papers.
"""
import numpy as np

from combined.collection import RELATIONS
from week3.kmeans import kmeans, purity
from week7.lsi import LSI


def macro_scores(matrix):
    """Macro precision, recall and F1 from a confusion matrix (rows = true, columns = predicted)."""
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros_like(tp), where=matrix.sum(axis=0) > 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros_like(tp), where=matrix.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return {"accuracy": float(tp.sum() / matrix.sum()), "precision": float(precision.mean()),
            "recall": float(recall.mean()), "f1": float(f1.mean())}


def naive_bayes_relations(collection, track="all", alpha=1.0):
    """Train on SODA's train split, score on the rest of the collection."""
    index = collection.index(track)
    labels, train = collection.doc_relation, collection.doc_split == 0
    held_out = np.flatnonzero(~train)
    if not len(held_out) or not train.any():            # small subsets may miss a split
        train = np.arange(collection.N) % 5 != 0
        held_out = np.flatnonzero(~train)
    position = np.full(collection.N, -1, np.int32)
    position[held_out] = np.arange(len(held_out))

    ptr = np.asarray(index.ptr)
    classes = len(RELATIONS)
    counts = np.zeros((len(ptr) - 1, classes))
    for t in range(len(ptr) - 1):
        a, b = int(ptr[t]), int(ptr[t + 1])
        docs = index.docs[a:b]
        known = train[docs]
        counts[t] = np.bincount(labels[docs[known]], weights=index.tfs[a:b][known], minlength=classes)
    in_training = counts.sum(axis=1) > 0
    smoothed = counts[in_training] + alpha
    log_likelihood = np.zeros_like(counts)
    log_likelihood[in_training] = np.log(smoothed) - np.log(smoothed.sum(axis=0, keepdims=True))
    log_prior = np.log(np.bincount(labels[train], minlength=classes) / train.sum())

    scores = np.tile(log_prior, (len(held_out), 1))
    for t in np.flatnonzero(in_training):
        a, b = int(ptr[t]), int(ptr[t + 1])
        slots = position[index.docs[a:b]]
        chosen = slots >= 0
        scores[slots[chosen]] += np.outer(index.tfs[a:b][chosen], log_likelihood[t])
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (labels[held_out], predicted), 1)
    return {"classes": RELATIONS, "confusion": matrix.tolist(), "documents": int(len(held_out)), **macro_scores(matrix),
            "majority_baseline": float(np.bincount(labels[held_out], minlength=classes).max() / len(held_out))}


def cluster_relations(collection, k_lsi, clusters=6, sample=50_000, seed=0):
    """K-Means on LSI coordinates of a document sample, scored against `relation`."""
    lsi = LSI(collection.index("all"), collection.folder, "all", k_lsi)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.permutation(collection.N)[:sample])
    X = np.asarray(lsi.docs[rows], np.float32)
    assign, _, inertia = kmeans(X, clusters, n_init=3, seed=seed)
    labels = collection.doc_relation[rows]
    majority = {c: int(np.bincount(labels[assign == c], minlength=len(RELATIONS)).argmax()) for c in range(clusters)}
    predicted = np.array([majority[c] for c in assign])
    matrix = np.zeros((len(RELATIONS), len(RELATIONS)), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    return {"purity": purity(assign.tolist(), labels.tolist()), "inertia": inertia, "documents": len(rows),
            "clusters": clusters, **{m: v for m, v in macro_scores(matrix).items() if m != "accuracy"},
            "sizes": np.bincount(assign, minlength=clusters).tolist()}
