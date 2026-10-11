"""Week 4 demo:  python -m week4.run"""
from data import load_corpus
from week1.index import InvertedIndex
from week4.probabilistic import BIM, BM25, to_probabilities
from week4.structured import StructuredIndex


def show(title, rows):
    print(f"\n{title}")
    for i, row in enumerate(rows, 1):
        print(f"  {i}. " + "  ".join(str(x) for x in row))


def main():
    corpus = load_corpus()
    whole = InvertedIndex.from_corpus(corpus)
    structured = StructuredIndex(corpus, whole)
    q = "intrusion detection on imbalanced datasets"
    print(f"Query: {q!r}   ({len(structured.regions)} regions indexed)")

    show("BIM", [(d, f"{s:.2f}") for d, s in BIM(whole).rank(q, top_k=5)])
    bm25 = BM25(whole).rank(q, top_k=5)
    show("BM25  (score, raw sigmoid P, calibrated P)",
         [(d, f"{s:.2f}", f"{p_raw:.3f}", f"{p:.3f}")
          for (d, s, p), (_, _, p_raw) in zip(to_probabilities(bm25), to_probabilities(bm25, calibrate=False))])
    show("Field-weighted BM25", [(d, f"{s:.2f}", "matched in: " + ", ".join(f)) for d, s, f in structured.field_bm25(q, top_k=5)])
    show("Methodology-only search", [(d, f"{s:.2f}") for d, s, _ in structured.search_fields(q, ["methodology"], top_k=5)])
    ov = structured.overlapped(q)
    show(f"Overlapped lists ({len(ov)} docs)", [(d, f"{int(s)} regions", regs[0].label) for d, s, regs in ov[:5]])
    nov = structured.non_overlapped(q)
    show(f"Non-overlapped lists ({len(nov)} docs)", [(d, f"coverage {s:.2f}") for d, s, _ in nov[:5]])
    show("Proximal nodes", [(d, f"{s:.2f}", " + ".join(n)) for d, s, n in structured.proximal(q, top_k=5)])


if __name__ == "__main__":
    main()
