"""Week 5 — Run every model on the same 15 queries and write results + charts.

    python -m week5.compare

Outputs (week5/results/):
    summary.csv         one row per model: mean P@5, R@10, MAP, nDCG@10, set P/R/F1, |retrieved|, ms/query
    per_query.csv       nDCG@10 and AP for every (model, query)
    sweeps.json         BM25 k1×b grid, proximal λ, field weights, region size, sigmoid calibration
    classification.json Naive Bayes + K-Means results
    *.png               charts used in ANALYSIS.md
"""
import json
import os
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import FIELDS, full_text, load_queries
from week3.classify_cluster import classify_cv, cluster
from week3.evaluate import precision_recall_f1
from week4.probabilistic import BM25, to_probabilities
from week4.structured import DEFAULT_FIELD_WEIGHTS, StructuredIndex
from week5.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from week5.systems import build_systems

OUT = Path(__file__).parent / "results"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def evaluate_run(run, queries):
    rows, per_query = [], []
    for q in queries:
        rel = q["relevant"]
        t0 = time.perf_counter()
        ranking = run(q)
        ms = (time.perf_counter() - t0) * 1000
        p, r, f = precision_recall_f1(ranking, rel)
        row = {"P@5": precision_at_k(ranking, rel, 5), "R@10": recall_at_k(ranking, rel, 10),
               "AP": average_precision(ranking, rel), "nDCG@10": ndcg_at_k(ranking, rel, 10),
               "set_P": p, "set_R": r, "set_F1": f, "retrieved": len(ranking), "ms": ms}
        rows.append(row)
        per_query.append({"query": q["id"], **row})
    mean = {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
    mean["MAP"] = mean.pop("AP")
    return mean, per_query


def map_score(run, queries):
    return statistics.mean(average_precision(run(q), q["relevant"]) for q in queries)


def sweeps(ctx, queries):
    index, corpus = ctx["index"], ctx["corpus"]
    out = {}

    k1s, bs = [0.0, 0.5, 0.9, 1.2, 1.6, 2.0, 3.0], [0.0, 0.25, 0.5, 0.75, 1.0]
    out["bm25_grid"] = {"k1": k1s, "b": bs, "MAP": [
        [map_score(lambda q, m=BM25(index, k1, b): [d for d, _ in m.rank(q["text"])], queries) for b in bs]
        for k1 in k1s]}

    st = ctx["structured"]
    out["proximal_lambda"] = {str(lam): map_score(lambda q, lam=lam: [d for d, *_ in st.proximal(q["text"], lam=lam)],
                                                  queries) for lam in [0.0, 0.3, 0.5, 0.7, 1.0]}

    variants = {
        "default (title 3, abstract 2, ..., refs 0.3)": DEFAULT_FIELD_WEIGHTS,
        "uniform (all 1)": {f: 1.0 for f in FIELDS},
        "no references": {**DEFAULT_FIELD_WEIGHTS, "references": 0.0},
        "title + abstract only": {f: (1.0 if f in ("title", "abstract") else 0.0) for f in FIELDS},
        "methodology + results only": {f: (1.0 if f in ("methodology", "results") else 0.0) for f in FIELDS},
    }
    out["field_weights"] = {name: map_score(lambda q, w=w: [d for d, *_ in st.field_bm25(q["text"], w)], queries)
                            for name, w in variants.items()}

    region = {}
    for words in [25, 50, 100, 200]:
        s = StructuredIndex(corpus, index, region_words=words)
        ov = [precision_recall_f1([d for d, *_ in s.overlapped(q["text"])], q["relevant"]) for q in queries]
        nov = [precision_recall_f1([d for d, *_ in s.non_overlapped(q["text"])], q["relevant"]) for q in queries]
        region[str(words)] = {"overlapped_P": statistics.mean(x[0] for x in ov),
                              "overlapped_R": statistics.mean(x[1] for x in ov),
                              "non_overlapped_P": statistics.mean(x[0] for x in nov),
                              "non_overlapped_R": statistics.mean(x[1] for x in nov)}
    out["region_size"] = region

    bm25 = ctx["bm25"]
    raw_spread, cal_spread = [], []
    for q in queries:
        top = bm25.rank(q["text"], top_k=10)
        raw_spread.append(statistics.pstdev(p for *_, p in to_probabilities(top, calibrate=False)))
        cal_spread.append(statistics.pstdev(p for *_, p in to_probabilities(top, calibrate=True)))
    out["sigmoid"] = {"raw_sigmoid_mean_std_top10": statistics.mean(raw_spread),
                      "calibrated_mean_std_top10": statistics.mean(cal_spread)}
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(summary, per_query, sw):
    # 1. MAP per model, best highlighted
    s = summary.sort_values("MAP")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [ORANGE if v == s["MAP"].max() else BLUE for v in s["MAP"]]
    ax.barh(s.index, s["MAP"], color=colors, height=0.6)
    for y, v in enumerate(s["MAP"]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_title("Mean Average Precision (15 queries)", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "map_by_model.png", dpi=150)
    plt.close(fig)

    # 2. Precision vs recall of the full retrieved set — the precision/coverage trade-off
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(summary["set_R"], summary["set_P"], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    groups = {}
    for name, row in summary.iterrows():   # models landing on the same point share one label
        groups.setdefault((round(row["set_R"], 3), round(row["set_P"], 3)), []).append(name)
    for (x, y), names in groups.items():
        label = names[0] if len(names) == 1 else f"{len(names)} ranked models\n(any doc sharing a term)"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(-8 if x > 0.8 else 6, 8),
                    ha="right" if x > 0.8 else "left", fontsize=8, color=INK)
    ax.set_xlabel("Recall of everything retrieved", color=MUTED)
    ax.set_ylabel("Precision of everything retrieved", color=MUTED)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision vs coverage of the retrieved set", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "precision_vs_recall.png", dpi=150)
    plt.close(fig)

    # 3. BM25 parameter grid
    g = sw["bm25_grid"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(g["MAP"], cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(g["b"])), [str(b) for b in g["b"]])
    ax.set_yticks(range(len(g["k1"])), [str(k) for k in g["k1"]])
    ax.set_xlabel("b (length normalization)", color=MUTED)
    ax.set_ylabel("k1 (tf saturation)", color=MUTED)
    vmax = max(max(r) for r in g["MAP"])
    for i, row in enumerate(g["MAP"]):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > vmax - (vmax - min(min(r) for r in g["MAP"])) * 0.4 else INK)
    ax.set_title("BM25 MAP across k1 and b", loc="left", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "bm25_grid.png", dpi=150)
    plt.close(fig)

    # 4. Per-query nDCG@10 heatmap
    pivot = per_query.pivot(index="model", columns="query", values="nDCG@10").loc[summary.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ax.set_title("nDCG@10 per query (darker = better)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "per_query_ndcg.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    queries = load_queries()
    systems, ctx = build_systems()

    summary_rows, per_query_rows = {}, []
    for sys_ in systems:
        mean, pq = evaluate_run(sys_.run, queries)
        summary_rows[sys_.name] = mean
        per_query_rows += [{"model": sys_.name, **r} for r in pq]
        print(f"{sys_.name:22s} MAP {mean['MAP']:.3f}  nDCG@10 {mean['nDCG@10']:.3f}  P@5 {mean['P@5']:.3f}  "
              f"setP {mean['set_P']:.2f} setR {mean['set_R']:.2f}  |ret| {mean['retrieved']:.1f}")

    summary = pd.DataFrame(summary_rows).T[["P@5", "R@10", "MAP", "nDCG@10", "set_P", "set_R", "set_F1",
                                             "retrieved", "ms"]].round(4)
    summary.to_csv(OUT / "summary.csv")
    per_query = pd.DataFrame(per_query_rows).round(4)
    per_query.to_csv(OUT / "per_query.csv", index=False)

    print("\nRunning parameter sweeps ...")
    sw = sweeps(ctx, queries)
    (OUT / "sweeps.json").write_text(json.dumps(sw, indent=2))

    corpus = ctx["corpus"]
    texts, labels = [full_text(d) for d in corpus], [d["category"] for d in corpus]
    nb, km = classify_cv(texts, labels), cluster(texts, labels)
    (OUT / "classification.json").write_text(json.dumps({
        "naive_bayes_accuracy": nb["accuracy"], "classes": nb["classes"],
        "confusion": nb["confusion"].tolist(),
        "misclassified": [{"doc": d["id"], "title": d["title"], "true": t, "pred": p}
                          for d, t, p in zip(corpus, labels, nb["predictions"]) if t != p],
        "kmeans_purity": km["purity"],
        "clusters": [{k: v for k, v in c.items() if k != "members"} | {"docs": [corpus[i]["id"] for i in c["members"]]}
                     for c in km["clusters"]],
    }, indent=2))

    charts(summary, per_query, sw)
    print(f"Done. Results in {OUT}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
