"""Every model behind one interface, grouped the way the slides group them.

family follows the slides' taxonomy: classical, probabilistic, structured, set-theoretic, algebraic,
plus the search-engine foundations, browsing and classification.
kind:  "ranked" returns a ranking, "set" returns an unordered set, "graded" returns a ranked set.
"""
from dataclasses import dataclass
from typing import Callable

from combined.collection import Collection
from combined.fast.rankers import Rankers
from setmodels.extended_boolean import ExtendedBoolean
from setmodels.fuzzy import Fuzzy
from week1.preprocess import preprocess
from week7.gvsm import GVSM
from week7.lsi import LSI

# Structural links between a dialogue's nodes (the role of slide 17's links between paper sections).
EDGES = {("narrative", "opening"), ("narrative", "body"), ("narrative", "closing"),
         ("opening", "body"), ("body", "closing")}
DEFAULTS = {"k1": 1.2, "b": 0.75, "lam": 0.7, "theta": 0.2, "p": 2.0, "k": 800,
            "weights": {"narrative": 1.0, "opening": 1.0, "body": 1.0, "closing": 1.0}}


@dataclass
class Model:
    name: str
    week: int
    family: str
    kind: str
    run: Callable          # query dict -> document ids


def build_models(collection, track="all", params=None, with_algebraic=True):
    p = {**DEFAULTS, **(params or {})}
    if track == "dialogue":                      # field weights are tuned with the narrative, which this track lacks
        p["weights"] = {f: w for f, w in DEFAULTS["weights"].items() if f != "narrative"}
    r = Rankers(collection, track)
    fuzzy = Fuzzy(r.index, collection.terms, p["theta"])
    extended = ExtendedBoolean(r, p["p"])
    models = [
        Model("Term overlap", 1, "foundations", "ranked", lambda q: r.overlap(q["text"])),
        Model("TF-IDF sum", 2, "foundations", "ranked", lambda q: r.weighted_sum(q["text"])),
        Model("Boolean", 3, "classical", "set", lambda q: r.boolean(q["boolean"])),
        Model("VSM cosine", 3, "classical", "ranked", lambda q: r.vsm(q["text"])),
        Model("BIM", 4, "probabilistic", "ranked", lambda q: r.bim(q["text"])),
        Model("BM25", 4, "probabilistic", "ranked", lambda q: r.bm25(q["text"], p["k1"], p["b"])),
        Model("Field BM25", 4, "structured", "ranked", lambda q: r.field_bm25(q["text"], p["weights"], p["k1"], p["b"])),
        Model("Overlapped lists", 4, "structured", "graded", lambda q: r.overlapped(q["text"])),
        Model("Non-overlapped lists", 4, "structured", "ranked", lambda q: r.non_overlapped(q["text"])),
        Model("Proximal nodes", 4, "structured", "ranked",
              lambda q: r.proximal(q["text"], EDGES, p["lam"], p["k1"], p["b"])),
        Model("Fuzzy (AND)", 6, "set-theoretic", "graded", lambda q: fuzzy.rank(q["boolean"])),
        Model("Fuzzy (OR)", 6, "set-theoretic", "graded", lambda q: fuzzy.rank(q["boolean"].replace(" AND ", " OR "))),
        Model("Extended Boolean", 6, "set-theoretic", "graded", lambda q: extended.rank(q["boolean"])),
    ]
    context = {"rankers": r, "fuzzy": fuzzy, "extended": extended, "params": p}
    if with_algebraic:
        gvsm = GVSM(r)
        lsi = LSI(r.index, collection.folder, track, p["k"])
        models.append(Model("GVSM", 7, "algebraic", "ranked", lambda q: gvsm.rank(preprocess(q["text"]))))
        models.append(Model("LSI", 7, "algebraic", "ranked", lambda q: lsi.rank(q["text"])))
        context.update(gvsm=gvsm, lsi=lsi)
    return models, context


def open_collection(size):
    return Collection(size)
