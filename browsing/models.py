"""Week 5 slides — Browsing models: exploring step by step instead of asking directly.

Flat browsing          read documents one after another (slides 7–8)
                           Browsing Time = Σ_i t_i
Structured guided      a hierarchy with a visible path (slides 9–11)
                           StructuredGuidedBrowsing(D, H, UI) = N
                       D = collection, H = hierarchy, UI = menu and breadcrumb, N = navigation path
                       SODA has no topic labels, so H = relation -> K-Means topic -> event -> dialogue -> section.
Hypertext              non-linear jumps over links (slides 12, 19)
                           G = (V, E)   V = dialogues, E = same event, same inference, similar content

Effort is compared as the number of items a user examines before reaching a relevant dialogue.
"""
import json
from collections import deque

import numpy as np

from combined.collection import FIELDS, RELATIONS
from week3.kmeans import lloyd, plus_plus

WORDS_PER_MINUTE = 200   # assumed reading speed
PAGE_SIZE = 20
TOPICS = 12


def browsing_time(minutes):
    """Slide 8: 30 + 10 + 45 = 85 minutes."""
    return sum(minutes)


def reading_minutes(words):
    return words / WORDS_PER_MINUTE


def flat_page(n_documents, page, size=PAGE_SIZE):
    """Document ids on one page of the plain list."""
    return list(range(page * size, min((page + 1) * size, n_documents)))


class Hierarchy:
    """H: every dialogue has a place in relation -> topic -> event."""

    def __init__(self, collection, lsi, topics=TOPICS, sample=50_000, seed=0):
        self.c, self.lsi = collection, lsi
        path = collection.folder / "browse_topics.npy"
        names = collection.folder / "browse_topics.json"
        if path.exists():
            self.topic = np.load(path)
            self.topic_names = json.loads(names.read_text(encoding="utf-8"))
            return
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.permutation(collection.N)[:sample])
        X = np.asarray(lsi.docs[rows], np.float64)
        _, centroids, _ = lloyd(X, plus_plus(X, topics, rng))
        self.topic = np.zeros(collection.N, np.int8)
        for start in range(0, collection.N, 100_000):
            block = np.asarray(lsi.docs[start:start + 100_000], np.float32)
            self.topic[start:start + 100_000] = (block @ centroids.astype(np.float32).T).argmax(axis=1)
        loadings = lsi.U @ centroids.T.astype(np.float32)        # which words describe each centroid
        self.topic_names = []
        for t in range(topics):
            top = np.argsort(-loadings[:, t])[:4]
            self.topic_names.append(", ".join(collection.terms[int(lsi.kept[i])] for i in top))
        np.save(path, self.topic)
        names.write_text(json.dumps(self.topic_names, ensure_ascii=False), encoding="utf-8")

    def menu(self):
        """Category menu: relation -> topic -> dialogue count."""
        tree = {}
        for r, relation in enumerate(RELATIONS):
            in_relation = self.c.doc_relation == r
            counts = np.bincount(self.topic[in_relation], minlength=len(self.topic_names))
            tree[relation] = [{"topic": int(t), "name": self.topic_names[t], "count": int(counts[t])}
                              for t in np.argsort(-counts) if counts[t]]
        return tree

    def events(self, relation, topic, limit=200):
        mask = (self.c.doc_relation == relation) & (self.topic == topic)
        heads, counts = np.unique(self.c.doc_head[mask], return_counts=True)
        order = np.argsort(-counts)[:limit]
        return [(int(heads[i]), int(counts[i])) for i in order]

    def dialogues(self, relation, topic, event):
        return np.flatnonzero((self.c.doc_relation == relation) & (self.topic == topic) & (self.c.doc_head == event))

    def path(self, doc, section=None):
        """N: the breadcrumb for one dialogue."""
        d = self.c.document(doc)
        topic = int(self.topic[doc])
        crumbs = [d["relation"], self.topic_names[topic], d["head"], f"Dialogue {doc}"]
        if section in FIELDS:
            crumbs.append(section)
        return crumbs


class Hypertext:
    """G = (V, E), with links worked out on demand."""

    def __init__(self, collection, lsi, per_type=5):
        self.c, self.lsi, self.per_type = collection, lsi, per_type

    def similar(self, doc):
        """Internal links to the closest dialogues in concept space."""
        scores = []
        vector = np.asarray(self.lsi.docs[doc], np.float32)
        best = np.empty(0, np.int64)
        best_scores = np.empty(0, np.float32)
        for start in range(0, self.c.N, 100_000):
            s = np.asarray(self.lsi.docs[start:start + 100_000], np.float32) @ vector
            pick = np.argpartition(-s, min(self.per_type, len(s) - 1))[:self.per_type + 1]
            best = np.r_[best, start + pick]
            best_scores = np.r_[best_scores, s[pick]]
        order = np.argsort(-best_scores)
        return [int(best[i]) for i in order if best[i] != doc][:self.per_type]

    def links(self, doc):
        same_event = np.flatnonzero(self.c.doc_head == self.c.doc_head[doc])
        return {"same event": [int(d) for d in same_event if d != doc][:self.per_type], "similar": self.similar(doc)}


def similarity_graph(vectors, neighbours=5, block=2000):
    """E for a small V: each dialogue links to its nearest neighbours, as adjacency lists."""
    n = len(vectors)
    adjacency = [set() for _ in range(n)]
    for start in range(0, n, block):
        s = vectors[start:start + block] @ vectors.T
        for i, row in enumerate(s):
            row[start + i] = -1
            for j in np.argpartition(-row, neighbours)[:neighbours]:
                adjacency[start + i].add(int(j))
                adjacency[int(j)].add(start + i)
    return adjacency


def shortest_path(adjacency, start, goals):
    """Breadth-first search: fewest link clicks from start to any goal. None if unreachable."""
    if start in goals:
        return [start]
    previous = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if other not in previous:
                previous[other] = node
                if other in goals:
                    path = [other]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])
                    return path[::-1]
                queue.append(other)
    return None
