"""Week 2 demo:  python -m week2.run"""
from data import full_text, load_corpus
from week2.filtering import simulate_stream
from week2.framework import WEIGHTING, IRSystem


def main():
    corpus = load_corpus()
    docs = {d["id"]: full_text(d) for d in corpus}
    query = "federated learning for medical image segmentation"

    print(f"|D| = {len(docs)} documents")
    for scheme in WEIGHTING:
        system = IRSystem(docs, weighting=scheme)
        top = ", ".join(f"{d}({s:.2f})" for d, s in system.rank(query, top_k=5))
        print(f"{scheme:7s} |F(q)| = {len(system.F(query)):2d}   top-5: {top}")
    print(f"|T| = {len(system.T)} index terms")

    print("\nFiltering — profile 'intrusion detection network security', papers arrive oldest first:")
    stream = sorted(corpus, key=lambda d: d["arxiv_id"])
    log = simulate_stream([(d["id"], full_text(d)) for d in stream],
                          "intrusion detection network security attacks", threshold=0.15)
    for row in log:
        if row["delivered"]:
            print(f"  step {row['step']:2d}: pushed {row['doc_id']} (score {row['score']:.3f})")


if __name__ == "__main__":
    main()
