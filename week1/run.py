"""Week 1 demo:  python -m week1.run "machine learning" """
import sys

from data import load_corpus
from week1.index import InvertedIndex
from week1.search import pipeline_trace


def main():
    query = " ".join(sys.argv[1:]) or "intrusion detection"
    corpus = load_corpus()
    titles = {d["id"]: d["title"] for d in corpus}
    index = InvertedIndex.from_corpus(corpus)
    print(f"Indexed {index.N} docs, {len(index.vocabulary)} unique terms\n")

    trace = pipeline_trace(index, query)
    print("Original query :", trace["original"])
    print("Preprocessed   :", trace["preprocessed"])
    for term, docs in trace["lookup"].items():
        print(f"Index lookup   : {term!r} -> {len(docs)} docs")
    print("\nRanking (shared terms, total tf):")
    for rank, (doc_id, shared, total) in enumerate(trace["ranking"], 1):
        print(f" {rank}. {doc_id}  shared={shared} tf={total:3d}  {titles[doc_id][:70]}")


if __name__ == "__main__":
    main()
