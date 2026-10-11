"""Week 5 — Every model from Weeks 1–4 behind one interface: query -> ranked list of doc_ids.

Boolean returns a set, not a ranking; it is listed in doc-id order (the order a set-based
system would show), which is why its ranked metrics suffer.
"""
from dataclasses import dataclass
from typing import Callable

from data import full_text, load_corpus
from week1.index import InvertedIndex
from week1.search import rank_overlap
from week2.framework import IRSystem
from week3.boolean import BooleanModel
from week3.vsm import VectorSpaceModel
from week4.probabilistic import BIM, BM25
from week4.structured import StructuredIndex


@dataclass
class System:
    name: str
    week: int
    ranked: bool                       # False = unordered set output
    run: Callable                      # query dict -> [doc_id]


def build_systems(corpus=None):
    corpus = corpus or load_corpus()
    docs = {d["id"]: full_text(d) for d in corpus}
    index = InvertedIndex(docs)
    framework = IRSystem(docs, weighting="tf-idf")
    boolean = BooleanModel(index)
    vsm = VectorSpaceModel(docs)
    bim, bm25 = BIM(index), BM25(index)
    structured = StructuredIndex(corpus, index)

    ids = lambda rows: [r[0] for r in rows]
    systems = [
        System("W1 Term overlap", 1, True, lambda q: ids(rank_overlap(index, q["text"]))),
        System("W2 TF-IDF sum", 2, True, lambda q: ids(framework.rank(q["text"], top_k=None))),
        System("W3 Boolean", 3, False, lambda q: sorted(boolean.search(q["boolean"]))),
        System("W3 VSM cosine", 3, True, lambda q: ids(vsm.rank(q["text"]))),
        System("W4 BIM", 4, True, lambda q: ids(bim.rank(q["text"]))),
        System("W4 BM25", 4, True, lambda q: ids(bm25.rank(q["text"]))),
        System("W4 Field BM25", 4, True, lambda q: ids(structured.field_bm25(q["text"]))),
        System("W4 Overlapped lists", 4, True, lambda q: ids(structured.overlapped(q["text"]))),
        System("W4 Non-overlapped", 4, True, lambda q: ids(structured.non_overlapped(q["text"]))),
        System("W4 Proximal nodes", 4, True, lambda q: ids(structured.proximal(q["text"]))),
    ]
    return systems, {"corpus": corpus, "docs": docs, "index": index, "bm25": bm25, "structured": structured}
