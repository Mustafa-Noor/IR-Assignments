"""The two evaluations that are not ad-hoc ranking: filtering (Week 2) and browsing (Week 5 slides).

    python -m combined.extras 200000

Filtering   each event becomes a fixed profile watching a stream of dialogues, using the original
            Week 2 engine. Scored as delivery precision, recall and F1.
Browsing    how many items a user examines before reaching a relevant dialogue, against BM25 search.
            Hypertext is measured on a smaller graph, because its links must be computed pair by pair.
"""
import json
import statistics
import sys

import numpy as np

from browsing.models import TOPICS, Hierarchy, reading_minutes, shortest_path, similarity_graph
from combined.catalog import DEFAULTS, open_collection
from combined.evaluate import RESULTS, build_queries, yield_to_the_user
from combined.fast.rankers import Rankers
from week2.filtering import FilteringEngine
from week1.preprocess import preprocess
from week3.evaluate import precision_recall_f1
from week7.lsi import LSI

PROFILES, STREAM, BROWSE_QUERIES, HYPERTEXT_DOCS = 30, 2000, 200, 10_000


def filtering(collection, test, seed=0):
    rng = np.random.default_rng(seed)
    background = rng.permutation(collection.N)[:STREAM].tolist()
    tokens = {}

    def text(doc):
        if doc not in tokens:
            d = collection.document(doc)
            tokens[doc] = preprocess((d["narrative"] or "") + " " + " ".join(d["dialogue"]))
        return tokens[doc]

    rows = []
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.3):
        scores = []
        for q in test[:PROFILES]:
            engine = FilteringEngine(q["text"], threshold)
            pushed = []
            for doc in sorted(set(background) | set(q["relevant"])):
                tf = text(doc)
                engine.seen += 1
                engine.df.update(set(tf))
                if engine.score(tf) >= threshold:
                    pushed.append(doc)
            scores.append(precision_recall_f1(pushed, q["relevant"]))
        rows.append({"threshold": threshold, "precision": statistics.mean(s[0] for s in scores),
                     "recall": statistics.mean(s[1] for s in scores), "f1": statistics.mean(s[2] for s in scores)})
        print("  filtering", rows[-1], flush=True)
    return {"profiles": PROFILES, "stream": STREAM, "rows": rows}


def browsing(collection, test, k, params):
    r = Rankers(collection, "all")
    lsi = LSI(r.index, collection.folder, "all", k)
    hierarchy = Hierarchy(collection, lsi)
    words = float(r.index.doc_len.mean()) * 1.6          # stems kept are roughly 60% of the words read
    per_item = reading_minutes(words)
    flat, guided, search = [], [], []
    for q in test[:BROWSE_QUERIES]:
        docs = sorted(q["relevant"])
        flat.append(docs[0] + 1)                         # reading the plain list from the start
        costs = []
        for d in docs[:5]:
            relation, topic = int(collection.doc_relation[d]), int(hierarchy.topic[d])
            events = [e for e, _ in hierarchy.events(relation, topic, limit=None)]
            position = events.index(int(collection.doc_head[d])) + 1
            in_event = len(hierarchy.dialogues(relation, topic, int(collection.doc_head[d])))
            costs.append(3.5 + (TOPICS + 1) / 2 + position + (in_event + 1) / 2)   # average menu scans plus lists
        guided.append(min(costs))
        ranking = r.bm25(q["text"], params["k1"], params["b"]).tolist()
        hit = next((i + 1 for i, d in enumerate(ranking) if d in q["relevant"]), None)
        if hit:
            search.append(hit)
    rows = [{"name": "Flat browsing (read the list in order)", "items": statistics.median(flat)},
            {"name": "Structured guided browsing", "items": statistics.median(guided)},
            {"name": f"Search with BM25 ({len(search)} of {BROWSE_QUERIES} queries found a match)", "items": statistics.median(search)}]

    small = open_collection("10000")
    small_lsi = LSI(small.index("all"), small.folder, "all", params["k"])
    vectors = np.asarray(small_lsi.docs, np.float32)
    graph = similarity_graph(vectors)
    rng = np.random.default_rng(0)
    dev, small_test = build_queries(small)
    hops = []
    for q in small_test[:BROWSE_QUERIES]:
        path = shortest_path(graph, int(rng.integers(small.N)), set(q["relevant"]))
        if path:
            hops.append(len(path) - 1)
    links = statistics.mean(len(a) for a in graph)
    rows.append({"name": f"Hypertext links, best path, {small.N:,} dialogues", "items": statistics.median(hops) * links})
    for row in rows:
        row["minutes"] = row["items"] * per_item
    print("  browsing", rows, flush=True)
    return {"rows": rows, "note": f"Guided browsing assumes the user picks the right menu entries. Hypertext is the shortest "
                                  f"path, about {statistics.median(hops):.0f} clicks with {links:.0f} links per page, so it is a best case."}


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "10000"
    yield_to_the_user()
    collection = open_collection(size)
    params = {**DEFAULTS, **json.loads((RESULTS / size / "params.json").read_text())}
    _, test = build_queries(collection)
    out = {"filtering": filtering(collection, test), "browsing": browsing(collection, test, params["k"], params)}
    (RESULTS / size / "extras.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
