"""Run every model on the same SODA queries and save accuracy, precision, recall, F1 and ranked metrics.

    python -m combined.evaluate 200000          tune on dev queries, then score on test queries
    python -m combined.evaluate full 200000     reuse settings tuned on a smaller collection

Relevance:  a dialogue is relevant to an event's query if it was generated from that same event.
Rows of "retrieve nothing" are included on purpose. With millions of documents almost everything is
correctly not retrieved, so accuracy is near 1 for any system, even one that returns nothing.

Outputs (combined/results/<size>/): summary_<track>.csv, per_query_<track>.csv, sweeps.json, classify.json
"""
import csv
import ctypes
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from combined.catalog import EDGES, build_models, open_collection
from combined.collection import event_text
from combined.fast.classify import cluster_relations, naive_bayes_relations
from combined.fast.rankers import Rankers
from week1.preprocess import preprocess
from week5.metrics import average_precision, ndcg_at_k
from week7.lsi import LSI

RESULTS = Path(__file__).parent / "results"
K = 10
DEV, TEST = 500, 2000


def build_queries(collection, dev=DEV, test=TEST):
    """Events with at least 5 dialogues, in the collection's fixed random order: dev first, then test."""
    meta = np.load(collection.folder.parent / "meta.npz", allow_pickle=True)
    names = meta["events"]
    order = np.argsort(collection.doc_head, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(collection.doc_head[order])) + 1)
    members = {int(collection.doc_head[g[0]]): g for g in groups}
    queries = []
    for event in np.load(collection.folder / "events_order.npy"):
        docs = members.get(int(event))
        if docs is None or len(docs) < 5:
            continue
        text = event_text(names[event])
        words = [w for w in text.replace("'", " ").split() if preprocess(w)]
        if not words:
            continue
        queries.append({"id": f"E{event}", "text": text, "boolean": " AND ".join(words),
                        "relevant": {int(d): 1 for d in docs}})
        if len(queries) == dev + test:
            break
    cut = min(dev, len(queries) // 5)
    return queries[:cut], queries[cut:]


def confusion_scores(tp, retrieved, relevant, N):
    fp, fn = retrieved - tp, relevant - tp
    tn = N - tp - fp - fn
    precision = tp / retrieved if retrieved else 0.0
    recall = tp / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": (tp + tn) / N, "balanced_accuracy": (recall + specificity) / 2,
            "precision": precision, "recall": recall, "f1": f1}


def query_scores(ranking, relevant, N, kind, no_shared):
    """All metrics for one query. no_shared = relevant dialogues sharing no query word."""
    found = np.asarray(ranking, np.int64)
    ranking = found[:1000].tolist()                     # only a Boolean set can be longer than this
    tp_k = sum(d in relevant for d in ranking[:K])
    at_k = confusion_scores(tp_k, min(K, len(ranking)), len(relevant), N)
    row = {f"{m}@{K}": v for m, v in at_k.items()}
    row["R-precision"] = sum(d in relevant for d in ranking[:len(relevant)]) / len(relevant)
    row["MAP"] = None if kind == "set" else average_precision(ranking, relevant)     # a set has no order to score
    row[f"nDCG@{K}"] = None if kind == "set" else ndcg_at_k(ranking, relevant, K)
    row["recall@1000"] = sum(d in relevant for d in ranking[:1000]) / len(relevant)
    row["gap_recall"] = sum(d in no_shared for d in ranking[:1000]) / len(no_shared) if no_shared else None
    row["retrieved"] = len(found)
    if kind != "ranked":
        tp = int(np.isin(found, list(relevant)).sum())
        row.update({f"set_{m}": v for m, v in confusion_scores(tp, len(found), len(relevant), N).items()})
    return row


def shared_mask(index, query):
    mask = np.zeros(index.N, bool)
    for t in set(preprocess(query)):
        mask[index.postings(t)[0]] = True
    return mask


def yield_to_the_user():
    """Run below normal priority, so a long evaluation leaves the laptop usable. Windows' stricter background
    mode is no good here: it caps the process at about 32 MB, and LSI then spends its time on page faults."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS


def score_model(model, queries, gaps, N, context):
    """Metrics for every query. Rankings are scored and dropped at once, because Boolean results get huge."""
    rows, seconds = [], 0.0
    if model.name == "LSI":                             # one pass over the coordinates for all queries
        start = time.perf_counter()
        rankings = context["lsi"].rank_many([q["text"] for q in queries])
        seconds = time.perf_counter() - start
        rows = [query_scores(r, q["relevant"], N, model.kind, g) for r, q, g in zip(rankings, queries, gaps)]
    else:
        for q, g in zip(queries, gaps):
            start = time.perf_counter()
            ranking = model.run(q)
            seconds += time.perf_counter() - start
            rows.append(query_scores(ranking, q["relevant"], N, model.kind, g))
    mean = {"model": model.name, "week": model.week, "family": model.family, "kind": model.kind,
            "ms": round(seconds * 1000 / len(queries), 3)}
    for key in rows[0]:
        values = [r[key] for r in rows if r.get(key) is not None]
        mean[key] = round(statistics.mean(values), 6) if values else ""
    per_query = [{"model": model.name, "query": q["id"], "MAP": "" if r["MAP"] is None else round(r["MAP"], 4),
                  f"f1@{K}": round(r[f"f1@{K}"], 4)} for r, q in zip(rows, queries)]
    return mean, per_query


def evaluate_track(collection, track, queries, params, out):
    """Each model is saved as it finishes, so a stopped run continues where it left off."""
    models, context = build_models(collection, track, params)
    index = context["rankers"].index
    N = context["rankers"].N
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    gaps = []
    for q in queries:
        mask = shared_mask(index, q["text"])
        gaps.append({d for d in q["relevant"] if not mask[d] and index.doc_len[d] > 0})
    summary, per_query = [], []
    for model in models:
        part = parts / f"{track}__{re.sub(r'[^A-Za-z0-9]+', '_', model.name)}.json"
        if part.exists():
            mean, rows = json.loads(part.read_text())
            print(f"  {model.name:22s} already done", flush=True)
        else:
            mean, rows = score_model(model, queries, gaps, N, context)
            unfinished = part.with_suffix(".tmp")
            unfinished.write_text(json.dumps([mean, rows]))
            unfinished.replace(part)
            print(f"  {model.name:22s} MAP {mean['MAP'] or 0:.3f}  P@10 {mean['precision@10']:.3f}  R@10 {mean['recall@10']:.3f}  "
                  f"F1@10 {mean['f1@10']:.3f}  Acc@10 {mean['accuracy@10']:.6f}  {mean['ms']:8.1f} ms/q", flush=True)
        summary.append(mean)
        per_query += rows
    nothing = {"model": "Retrieve nothing", "week": 0, "family": "baseline", "kind": "set", "ms": 0}
    empty = [query_scores([], q["relevant"], N, "set", g) for q, g in zip(queries, gaps)]
    for key in empty[0]:
        values = [r[key] for r in empty if r.get(key) is not None]
        nothing[key] = round(statistics.mean(values), 6) if values else ""
    summary.append(nothing)
    columns = list(dict.fromkeys(k for row in summary for k in row))
    with (out / f"summary_{track}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(summary)
    with (out / f"per_query_{track}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)


def mean_ap(run, queries):
    return statistics.mean(average_precision(np.asarray(run(q)).tolist(), q["relevant"]) for q in queries)


def tune(collection, dev):
    """Parameters are chosen on dev queries only, on the 'all' track."""
    r = Rankers(collection, "all")
    sweeps = {}
    grid = {f"{k1},{b}": mean_ap(lambda q, k1=k1, b=b: r.bm25(q["text"], k1, b), dev)
            for k1 in (0.0, 0.5, 0.9, 1.2, 1.6, 2.0, 3.0) for b in (0.0, 0.25, 0.5, 0.75, 1.0)}
    sweeps["bm25"] = grid
    k1, b = map(float, max(grid, key=grid.get).split(","))
    sweeps["proximal_lambda"] = {str(l): mean_ap(lambda q, l=l: r.proximal(q["text"], EDGES, l, k1, b), dev)
                                 for l in (0.0, 0.3, 0.5, 0.7, 1.0)}
    lam = float(max(sweeps["proximal_lambda"], key=sweeps["proximal_lambda"].get))
    variants = {"uniform": {"narrative": 1.0, "opening": 1.0, "body": 1.0, "closing": 1.0},
                "narrative x2": {"narrative": 2.0, "opening": 1.0, "body": 1.0, "closing": 1.0},
                "narrative x0.5": {"narrative": 0.5, "opening": 1.0, "body": 1.0, "closing": 1.0},
                "dialogue only": {"narrative": 0.0, "opening": 1.0, "body": 1.0, "closing": 1.0},
                "narrative only": {"narrative": 1.0, "opening": 0.0, "body": 0.0, "closing": 0.0}}
    sweeps["field_weights"] = {n: mean_ap(lambda q, w=w: r.field_bm25(q["text"], w, k1, b), dev) for n, w in variants.items()}
    weights = variants[max(sweeps["field_weights"], key=sweeps["field_weights"].get)]
    sweeps["lsi_k"] = {}
    for k in (100, 200, 300, 500, 800):
        lsi = LSI(r.index, collection.folder, "all", k)
        rankings = lsi.rank_many([q["text"] for q in dev])
        sweeps["lsi_k"][str(k)] = statistics.mean(average_precision(x.tolist(), q["relevant"]) for x, q in zip(rankings, dev))
    best_k = int(max(sweeps["lsi_k"], key=sweeps["lsi_k"].get))
    return {"k1": k1, "b": b, "lam": lam, "weights": weights, "k": best_k}, sweeps


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "10000"
    tune_on = sys.argv[2] if len(sys.argv) > 2 else None
    yield_to_the_user()
    collection = open_collection(size)
    out = RESULTS / size
    out.mkdir(parents=True, exist_ok=True)
    dev, test = build_queries(collection)
    print(f"{collection.N:,} dialogues, {len(dev)} dev and {len(test)} test queries, "
          f"{statistics.mean(len(q['relevant']) for q in test):.1f} relevant per query", flush=True)
    if tune_on:                                         # reuse parameters tuned on a smaller collection
        params = json.loads((RESULTS / tune_on / "params.json").read_text())
        tuned = json.loads((RESULTS / tune_on / "sweeps.json").read_text())
        params["k"] = int(max(tuned["lsi_k"], key=tuned["lsi_k"].get))
        sweeps = {}
    else:
        params, sweeps = tune(collection, dev)
    (out / "params.json").write_text(json.dumps(params, indent=2))
    if sweeps:
        (out / "sweeps.json").write_text(json.dumps(sweeps, indent=2))
    print("Parameters:", params, flush=True)
    for track in ("all", "dialogue"):
        print(f"[{track}]", flush=True)
        evaluate_track(collection, track, test, params, out)
    classify = {"naive_bayes": naive_bayes_relations(collection), "kmeans": cluster_relations(collection, params["k"])}
    (out / "classify.json").write_text(json.dumps(classify, indent=2))
    print("Naive Bayes accuracy", round(classify["naive_bayes"]["accuracy"], 4),
          "K-Means purity", round(classify["kmeans"]["purity"], 4), flush=True)
    shutil.rmtree(out / "parts")


if __name__ == "__main__":
    main()
