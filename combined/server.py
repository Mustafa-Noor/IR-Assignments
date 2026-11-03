"""The combined IR system: every model, its flow, and the comparison.

    python -m combined.server            then open http://127.0.0.1:5000

The Compare page shows the largest collection with saved results. Live pages (Search, Model flow, Browse) use the
largest one below the full set, because the full set is too slow to search live on a laptop; set IR_SIZE=full to
use it anyway. Models load the first time they are needed.
"""
import ast
import csv
import html
import json
import os
import re
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from browsing.models import Hierarchy, Hypertext, PAGE_SIZE, flat_page, reading_minutes
from combined.catalog import DEFAULTS, EDGES, open_collection
from combined.collection import FIELDS, RELATIONS, TRACKS, event_text
from combined.evaluate import RESULTS, build_queries
from combined.fast.rankers import Rankers
from combined.flows import FLOWS, boolean_query, step
from combined.registry import FAMILIES, MODELS, OTHERS
from setmodels.extended_boolean import ExtendedBoolean, normalize_query
from setmodels.fuzzy import Fuzzy
from week1.preprocess import preprocess, stem
from week3.boolean import BooleanQueryError
from week7.gvsm import GVSM
from week7.lsi import LSI

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).parent / "static"
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")


def available_sizes():
    sizes = [p.name for p in RESULTS.iterdir() if (p / "summary_all.csv").exists()] if RESULTS.exists() else []
    return sorted(sizes, key=lambda s: float("inf") if s == "full" else int(s))


class World:
    """The running models for one collection, created on first use."""

    def __init__(self, size):
        self.collection = open_collection(size)
        self.size = size
        path = RESULTS / size / "params.json"
        self.params = {**DEFAULTS, **(json.loads(path.read_text()) if path.exists() else {})}
        self.parts = {}
        self.events = np.load(self.collection.folder.parent / "meta.npz", allow_pickle=True)["events"]
        self.dev, self.test = build_queries(self.collection)

    def _once(self, key, make):
        if key not in self.parts:
            self.parts[key] = make()
        return self.parts[key]

    def params_for(self, track):
        p = dict(self.params)
        if track == "dialogue":
            p["weights"] = {f: w for f, w in DEFAULTS["weights"].items() if f != "narrative"}
        return p

    def rankers(self, track):
        return self._once(("r", track), lambda: Rankers(self.collection, track))

    def fuzzy(self, track):
        return self._once(("f", track), lambda: Fuzzy(self.rankers(track).index, self.collection.terms, self.params["theta"]))

    def extended(self, track):
        return self._once(("e", track), lambda: ExtendedBoolean(self.rankers(track), self.params["p"]))

    def gvsm(self, track):
        return self._once(("g", track), lambda: GVSM(self.rankers(track)))

    def lsi(self, track):
        return self._once(("l", track), lambda: LSI(self.rankers(track).index, self.collection.folder, track, self.params["k"]))

    def hierarchy(self):
        return self._once("h", lambda: Hierarchy(self.collection, self.lsi("all")))

    def hypertext(self):
        return self._once("x", lambda: Hypertext(self.collection, self.lsi("all")))


world = None


def get_world():
    global world
    if world is None:
        sizes = available_sizes()
        if not sizes:
            abort(503, "No results yet. Run python -m combined.evaluate first.")
        live = [s for s in sizes if s != "full"] or sizes
        world = World(os.environ.get("IR_SIZE", live[-1]))
    return world


def track_arg():
    track = request.args.get("track", "all")
    if track not in TRACKS:
        abort(400, "Unknown track")
    return track


def matched_terms(query):
    return set(preprocess(query.replace(" AND ", " ").replace(" OR ", " ").replace(" NOT ", " ")))


def mark(text, terms):
    """Escape the text and highlight match points (slide 12 of Week 4)."""
    out, last = [], 0
    for m in re.finditer(r"[A-Za-z0-9]+", text):
        if stem([m.group().lower()])[0] in terms:
            out.append(html.escape(text[last:m.start()]))
            out.append(f"<mark>{html.escape(m.group())}</mark>")
            last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def card(w, doc, terms=(), relevant=()):
    d = w.collection.document(doc)
    return {"id": doc, "event": event_text(d["head"]), "relation": d["relation"], "relevant": doc in relevant,
            "snippet": mark((d["narrative"] or "")[:190], set(terms))}


def find_query(w, text):
    for q in w.test + w.dev:
        if q["text"] == text:
            return q
    return None


@app.get("/")
def home():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/meta")
def meta():
    w = get_world()
    return jsonify({
        "size": w.size, "documents": w.collection.N, "terms": len(w.collection.terms), "sizes": available_sizes(),
        "families": FAMILIES, "params": w.params, "edges": sorted(EDGES),
        "models": [{"name": n, "week": m["week"], "family": m["family"], "idea": m["idea"], "slides": m["slides"]}
                   for n, m in MODELS.items()],
        "others": [{"name": n, "week": m["week"], "idea": m["idea"], "slides": m["slides"]} for n, m in OTHERS.items()],
        "samples": [q["text"] for q in w.test[:40]],
    })


def run_model(w, name, query, track):
    steps, ranked = FLOWS[name](w, track, query)
    return steps, ranked


@app.get("/api/search")
def search():
    """One query through every model, side by side."""
    w = get_world()
    query, track = request.args.get("q", "").strip(), track_arg()
    if not query:
        return jsonify({"error": "Type a query first."}), 400
    gold = find_query(w, query)
    relevant = gold["relevant"] if gold else {}
    terms = matched_terms(query)
    out = []
    for name in MODELS:
        try:
            _, ranked = run_model(w, name, query, track)
            out.append({"model": name, "family": MODELS[name]["family"], "results": [card(w, d, terms, relevant) for d in ranked]})
        except (BooleanQueryError, IndexError) as e:
            out.append({"model": name, "family": MODELS[name]["family"], "results": [], "error": str(e) or "Could not read the query."})
    return jsonify({"query": query, "judged": bool(gold), "relevant": len(relevant), "models": out})


@app.get("/api/flow")
def flow():
    w = get_world()
    name, query, track = request.args.get("model", ""), request.args.get("q", "").strip(), track_arg()
    if name not in MODELS:
        abort(404)
    if not query:
        return jsonify({"error": "Type a query first."}), 400
    try:
        steps, ranked = run_model(w, name, query, track)
    except (BooleanQueryError, IndexError) as e:
        return jsonify({"error": f"Could not read the query: {e}"}), 400
    terms = matched_terms(query)
    gold = find_query(w, query)
    steps.append(step("Results", note="No document was retrieved." if not ranked else None))
    m = MODELS[name]
    return jsonify({"model": name, "idea": m["idea"], "slides": m["slides"], "formula": m["formula"], "steps": steps,
                    "results": [card(w, d, terms, gold["relevant"] if gold else ()) for d in ranked]})


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/api/results")
def results():
    size = request.args.get("size") or get_world().size
    if size not in available_sizes():
        abort(404)
    track = track_arg()
    folder = RESULTS / size
    def load(name):
        p = folder / name
        return json.loads(p.read_text()) if p.exists() else None
    rows = read_csv(folder / f"summary_{track}.csv")
    per_query = read_csv(folder / f"per_query_{track}.csv")
    query_ids = list(dict.fromkeys(r["query"] for r in per_query))[:30]
    picked = [r for r in per_query if r["query"] in set(query_ids)]
    return jsonify({"size": size, "track": track, "summary": rows, "per_query": picked, "queries": query_ids,
                    "classify": load("classify.json"), "sweeps": load("sweeps.json"), "extras": load("extras.json"),
                    "params": load("params.json"), "scale": load("scale.json")})


@app.get("/api/doc/<int:doc>")
def document(doc):
    w = get_world()
    if not 0 <= doc < w.collection.N:
        abort(404)
    d = w.collection.document(doc)
    terms = matched_terms(request.args.get("q", ""))
    turns = [{"speaker": html.escape(s), "text": mark(t, terms)} for s, t in zip(d["speakers"], d["dialogue"])]
    words = len((d["narrative"] or "").split()) + sum(len(t.split()) for t in d["dialogue"])
    links = w.hypertext().links(doc)
    label = lambda i: {"id": i, "event": event_text(w.collection.document(i)["head"])}
    return jsonify({"id": doc, "head": d["head"], "event": event_text(d["head"]), "relation": d["relation"], "tail": d["tail"],
                    "split": d["split"], "narrative": mark(d["narrative"] or "", terms), "turns": turns, "words": words,
                    "minutes": round(reading_minutes(words), 2), "path": w.hierarchy().path(doc),
                    "topic": int(w.hierarchy().topic[doc]), "relation_id": RELATIONS.index(d["relation"]),
                    "event_id": int(w.collection.doc_head[doc]),
                    "links": {k: [label(i) for i in v] for k, v in links.items()}})


@app.get("/api/browse/menu")
def browse_menu():
    return jsonify(get_world().hierarchy().menu())


@app.get("/api/browse/events")
def browse_events():
    w = get_world()
    relation, topic = request.args.get("relation", type=int), request.args.get("topic", type=int)
    return jsonify([{"event": e, "name": event_text(w.events[e]), "count": c} for e, c in w.hierarchy().events(relation, topic)])


@app.get("/api/browse/dialogues")
def browse_dialogues():
    w = get_world()
    args = [request.args.get(k, type=int) for k in ("relation", "topic", "event")]
    return jsonify([card(w, int(d)) for d in w.hierarchy().dialogues(*args)[:50]])


@app.get("/api/browse/flat")
def browse_flat():
    w = get_world()
    page = max(request.args.get("page", 0, type=int), 0)
    relation = request.args.get("relation", default=None, type=int)
    split = request.args.get("split", default=None, type=int)
    if relation is None and split is None:
        ids = flat_page(w.collection.N, page)
        total = w.collection.N
    else:
        mask = np.ones(w.collection.N, bool)
        if relation is not None:
            mask &= w.collection.doc_relation == relation
        if split is not None:
            mask &= w.collection.doc_split == split
        found = np.flatnonzero(mask)
        total = len(found)
        ids = found[page * PAGE_SIZE:(page + 1) * PAGE_SIZE].tolist()
    return jsonify({"page": page, "pages": (total + PAGE_SIZE - 1) // PAGE_SIZE, "total": total,
                    "documents": [card(w, int(d)) for d in ids]})


def definition(path, symbol):
    """The real source of a function or class, read from the file."""
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = tree.body
    node = None
    for part in symbol.split("."):
        node = next(n for n in nodes if getattr(n, "name", None) == part)
        nodes = node.body
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]), node.lineno


@app.get("/api/source")
def source():
    name = request.args.get("model", "")
    m = MODELS.get(name) or OTHERS.get(name)
    if not m:
        abort(404)
    code = []
    for path, symbol in m["source"]:
        text, line = definition(path, symbol)
        code.append({"file": path, "symbol": symbol, "line": line, "code": text})
    return jsonify({"model": name, "idea": m["idea"], "slides": m["slides"], "formula": m["formula"], "code": code})


@app.errorhandler(503)
def not_ready(e):
    return jsonify({"error": e.description}), 503


if __name__ == "__main__":
    app.run()
