"""Week 3 demo:  python -m week3.run"""
from data import full_text, load_corpus, load_queries
from week1.index import InvertedIndex
from week3.boolean import BooleanModel
from week3.classify_cluster import classify_cv, cluster
from week3.evaluate import precision_recall_f1
from week3.vsm import VectorSpaceModel


def main():
    corpus = load_corpus()
    docs = {d["id"]: full_text(d) for d in corpus}
    index = InvertedIndex(docs)
    boolean, vsm = BooleanModel(index), VectorSpaceModel(docs)

    print("Query  | Boolean P / R            | VSM top-5 P / R")
    for q in load_queries():
        rel = q["relevant"]
        bp, br, _ = precision_recall_f1(boolean.search(q["boolean"]), rel)
        top5 = [d for d, _ in vsm.rank(q["text"], top_k=5)]
        vp, vr, _ = precision_recall_f1(top5, rel)
        print(f"{q['id']}    | {bp:.2f} / {br:.2f}  ({len(boolean.search(q['boolean'])):2d} hits) | {vp:.2f} / {vr:.2f}")

    texts, labels = [full_text(d) for d in corpus], [d["category"] for d in corpus]
    print(f"\nNaive Bayes 5-fold accuracy: {classify_cv(texts, labels)['accuracy']:.2%}")
    print(f"K-Means (k=6) purity:        {cluster(texts, labels)['purity']:.2%}")


if __name__ == "__main__":
    main()
