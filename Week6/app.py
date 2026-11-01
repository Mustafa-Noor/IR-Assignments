"""Browsing module for the social rehab app.

Provides scenario search (Boolean and TF-IDF vector ranking), category
browsing, breadcrumb data, related scenarios and scenario previews.
Scenario content is a draft until reviewed by a clinician.
"""
import json
import math
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "scenarios.json"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "you", "your", "i",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
QUERY_TOKEN_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+")


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_scenarios():
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)["scenarios"]


class SearchIndex:
    def __init__(self, scenarios):
        self.scenarios = {s["id"]: s for s in scenarios}
        self.all_ids = set(self.scenarios)
        self.postings = defaultdict(set)  # term -> scenario ids (Boolean)
        self.tf = {}
        self.df = Counter()

        for s in scenarios:
            text = " ".join([
                s["title"], s["summary"], s["category"],
                s["subcategory"], " ".join(s["tags"]),
            ])
            counts = Counter(tokenize(text))
            self.tf[s["id"]] = counts
            for term in counts:
                self.postings[term].add(s["id"])
                self.df[term] += 1

        n = len(scenarios)
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in self.df.items()}
        self.vectors = {sid: self._vectorize(c) for sid, c in self.tf.items()}

    def _vectorize(self, counts):
        vec = {
            t: (1 + math.log(c)) * self.idf[t]
            for t, c in counts.items() if t in self.idf
        }
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    # Boolean retrieval: AND, OR, NOT (capitals) and parentheses.
    # NOT binds tightest, then AND, then OR.
    def boolean_search(self, query):
        tokens = QUERY_TOKEN_RE.findall(query)
        if not tokens:
            return set(self.all_ids)
        pos = 0

        def peek():
            return tokens[pos] if pos < len(tokens) else None

        def take():
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("the query ends unexpectedly")
            tok = tokens[pos]
            pos += 1
            return tok

        def parse_or():
            result = parse_and()
            while peek() == "OR":
                take()
                result = result | parse_and()
            return result

        def parse_and():
            result = parse_not()
            while peek() == "AND":
                take()
                result = result & parse_not()
            return result

        def parse_not():
            if peek() == "NOT":
                take()
                return self.all_ids - parse_not()
            return parse_atom()

        def parse_atom():
            tok = take()
            if tok == "(":
                result = parse_or()
                if take() != ")":
                    raise ValueError("a closing parenthesis is missing")
                return result
            if tok in {")", "AND", "OR"}:
                raise ValueError(f"'{tok}' is in the wrong place")
            terms = tokenize(tok)
            if not terms:
                return set()
            return set(self.postings.get(terms[0], set()))

        result = parse_or()
        if pos != len(tokens):
            raise ValueError("part of the query could not be read")
        return result

    # Vector retrieval: cosine similarity between TF-IDF vectors.
    def vector_scores(self, query):
        q_vec = self._vectorize(Counter(tokenize(query)))
        if not q_vec:
            return {}
        scores = {}
        for sid, d_vec in self.vectors.items():
            score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            if score > 0:
                scores[sid] = score
        return scores

    def _term_membership(self, term, scenario_id):
        """Return the strongest partial match for a term in one scenario."""
        doc_terms = self.tf[scenario_id]
        if not doc_terms:
            return 0.0
        return max(SequenceMatcher(None, term, doc_term).ratio() for doc_term in doc_terms)

    def fuzzy_scores(self, query):
        """Evaluate a fuzzy query as document membership values in [0, 1]."""
        tokens = QUERY_TOKEN_RE.findall(query)
        if not tokens:
            return {sid: 1.0 for sid in self.all_ids}

        scores = {
            sid: {term: self._term_membership(term, sid) for term in self.df}
            for sid in self.all_ids
        }
        pos = 0

        def peek():
            return tokens[pos] if pos < len(tokens) else None

        def take():
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("the query ends unexpectedly")
            token = tokens[pos]
            pos += 1
            return token

        def term_scores(token):
            terms = tokenize(token)
            if not terms:
                return {sid: 0.0 for sid in self.all_ids}
            return {sid: max(scores[sid].get(term, 0.0) for term in terms)
                    for sid in self.all_ids}

        def combine(left, right, operation):
            if operation == "AND":
                return {sid: min(left[sid], right[sid]) for sid in self.all_ids}
            return {sid: max(left[sid], right[sid]) for sid in self.all_ids}

        def parse_or():
            result = parse_and()
            while peek() == "OR":
                take()
                result = combine(result, parse_and(), "OR")
            return result

        def parse_and():
            result = parse_not()
            while peek() == "AND" or (peek() and peek() not in {"OR", ")"}):
                if peek() == "AND":
                    take()
                result = combine(result, parse_not(), "AND")
            return result

        def parse_not():
            if peek() == "NOT":
                take()
                return {sid: 1.0 - value for sid, value in parse_not().items()}
            return parse_atom()

        def parse_atom():
            token = take()
            if token == "(":
                result = parse_or()
                if take() != ")":
                    raise ValueError("a closing parenthesis is missing")
                return result
            if token in {")", "AND", "OR"}:
                raise ValueError(f"'{token}' is in the wrong place")
            return term_scores(token)

        result = parse_or()
        if pos != len(tokens):
            raise ValueError("part of the query could not be read")
        return {sid: round(score, 4) for sid, score in result.items() if score > 0}

    def related(self, sid, limit=3):
        base = self.vectors[sid]
        sims = []
        for other, vec in self.vectors.items():
            if other == sid:
                continue
            score = sum(w * vec.get(t, 0.0) for t, w in base.items())
            if score > 0:
                sims.append((score, other))
        sims.sort(reverse=True)
        return [
            {"id": o, "title": self.scenarios[o]["title"], "score": round(s, 4)}
            for s, o in sims[:limit]
        ]

    def search(self, query, mode, category="", subcategory="", difficulty=""):
        if mode == "boolean":
            scored = {sid: None for sid in self.boolean_search(query)}
        elif mode == "fuzzy":
            scored = self.fuzzy_scores(query)
        elif query.strip():
            scored = self.vector_scores(query)
        else:
            scored = {sid: None for sid in self.all_ids}

        results = []
        for sid, score in scored.items():
            s = self.scenarios[sid]
            if category and s["category"] != category:
                continue
            if subcategory and s["subcategory"] != subcategory:
                continue
            if difficulty and s["difficulty"] != difficulty:
                continue
            results.append({
                "id": sid,
                "title": s["title"],
                "category": s["category"],
                "subcategory": s["subcategory"],
                "difficulty": s["difficulty"],
                "summary": s["summary"],
                "score": round(score, 4) if score is not None else None,
            })
        results.sort(key=lambda r: (-(r["score"] or 0), r["title"]))
        return results

    # ---- Model comparison ---------------------------------------------------
    def _filter_ids(self, ids, category, subcategory, difficulty):
        """Keep only the ids that pass the active category/difficulty filters."""
        kept = set()
        for sid in ids:
            s = self.scenarios[sid]
            if category and s["category"] != category:
                continue
            if subcategory and s["subcategory"] != subcategory:
                continue
            if difficulty and s["difficulty"] != difficulty:
                continue
            kept.add(sid)
        return kept

    def _normalize_query(self, query):
        """Insert implicit ANDs so a plain multi-word query parses for both.

        The Fuzzy parser already treats adjacent terms as AND; the Boolean
        parser requires the operator spelled out. Making the implicit ANDs
        explicit lets both models answer the same plain query (e.g. "family
        gathering" -> "family AND gathering") and does not change Fuzzy's
        result, so the comparison stays fair.
        """
        tokens = QUERY_TOKEN_RE.findall(query)

        def ends_operand(t):
            return t == ")" or t not in {"(", "AND", "OR", "NOT"}

        def starts_operand(t):
            return t == "(" or t == "NOT" or t not in {")", "AND", "OR"}

        out = []
        for tok in tokens:
            if out and ends_operand(out[-1]) and starts_operand(tok):
                out.append("AND")
            out.append(tok)
        return " ".join(out)

    def _correct_term(self, term):
        """Map a query word to its closest corpus word (fixes typos)."""
        if term in self.idf:
            return term
        best, best_ratio = term, 0.0
        for vocab in self.df:
            ratio = SequenceMatcher(None, term, vocab).ratio()
            if ratio > best_ratio:
                best, best_ratio = vocab, ratio
        return best if best_ratio >= 0.7 else term

    def relevant_set(self, query, universe):
        """A graded 'gold' relevance set used to score both models.

        There are no human relevance judgements for this small corpus, so we
        build the reference from the query's *intent*: strip the Boolean
        operators, spell-correct each content word to its nearest corpus word,
        then rank documents by TF-IDF cosine on that corrected query. A
        scenario is 'relevant' when its score reaches 30% of the best score.
        Correcting the terms means the reference captures what the searcher
        meant even when they mistype, so neither model's mechanism defines it.
        """
        terms = [t for t in tokenize(query) if t != "not"]
        corrected = Counter(self._correct_term(t) for t in terms)
        q_vec = self._vectorize(corrected)
        if not q_vec:
            return set()
        scores = {}
        for sid in universe:
            d_vec = self.vectors[sid]
            score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            if score > 0:
                scores[sid] = score
        if not scores:
            return set()
        cutoff = 0.3 * max(scores.values())
        return {sid for sid, sc in scores.items() if sc >= cutoff}

    @staticmethod
    def _prf(retrieved, relevant, universe_size):
        """Precision, recall, F1 and accuracy for one retrieved set."""
        tp = len(retrieved & relevant)
        fp = len(retrieved - relevant)
        fn = len(relevant - retrieved)
        tn = universe_size - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        accuracy = (tp + tn) / universe_size if universe_size else 0.0
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }

    def _time_retrieval(self, fn, max_runs, budget=0.15):
        """Average time (ms) of a retrieval call, and the run count used.

        Runs the call repeatedly for a stable average but caps the total work
        at `budget` seconds so a slow model does not stall the request: one
        warm-up call estimates the per-call cost, then the number of timed runs
        is chosen to fit the budget (at least 3, at most `max_runs`).
        """
        t0 = time.perf_counter()
        fn()
        single = time.perf_counter() - t0
        if single > 0:
            runs = min(max_runs, max(3, int(budget / single)))
        else:
            runs = max_runs
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        elapsed = time.perf_counter() - start
        return (elapsed / runs) * 1000.0, runs

    def compare(self, query, category="", subcategory="", difficulty="", runs=50):
        """Answer one query with both the Boolean and Fuzzy models and compare.

        Returns per-model retrieval quality (precision/recall/F1/accuracy),
        speed, an efficiency figure (quality per millisecond) and an overall
        verdict naming the best model for this query.
        """
        # Make implicit ANDs explicit so both models accept a plain query.
        nquery = self._normalize_query(query)

        universe = self._filter_ids(
            self.all_ids, category, subcategory, difficulty)
        universe_size = len(universe)
        relevant = self.relevant_set(nquery, universe)

        # Retrieved sets (respecting the active filters).
        bool_ids = self._filter_ids(
            self.boolean_search(nquery), category, subcategory, difficulty)
        fuzzy_raw = self.fuzzy_scores(nquery)
        fuzzy_ids = self._filter_ids(
            set(fuzzy_raw), category, subcategory, difficulty)

        # Timing of the core retrieval step, averaged over several runs.
        bool_ms, bool_runs = self._time_retrieval(
            lambda: self.boolean_search(nquery), runs)
        fuzzy_ms, fuzzy_runs = self._time_retrieval(
            lambda: self.fuzzy_scores(nquery), runs)

        def preview(ids, scores=None):
            items = []
            for sid in ids:
                items.append({
                    "id": sid,
                    "title": self.scenarios[sid]["title"],
                    "score": round(scores[sid], 4) if scores else None,
                })
            items.sort(key=lambda r: (-(r["score"] or 0), r["title"]))
            return items

        models = [
            {
                "key": "boolean",
                "name": "Boolean",
                "description": "Exact term match with AND / OR / NOT set logic.",
                "time_ms": round(bool_ms, 4),
                "runs": bool_runs,
                "count": len(bool_ids),
                "results": preview(bool_ids),
                **self._prf(bool_ids, relevant, universe_size),
            },
            {
                "key": "fuzzy",
                "name": "Fuzzy",
                "description": "Partial / approximate matching with fuzzy AND, OR, NOT.",
                "time_ms": round(fuzzy_ms, 4),
                "runs": fuzzy_runs,
                "count": len(fuzzy_ids),
                "results": preview(fuzzy_ids, fuzzy_raw),
                **self._prf(fuzzy_ids, relevant, universe_size),
            },
        ]

        # Normalise F1 (quality) and speed across the two models, then combine.
        best_f1 = max(m["f1"] for m in models) or 1.0
        best_ms = min(m["time_ms"] for m in models) or 1e-9
        for m in models:
            f1_norm = m["f1"] / best_f1 if best_f1 else 0.0
            speed_norm = best_ms / m["time_ms"] if m["time_ms"] else 1.0
            # Quality per millisecond: how much F1 each model buys per unit time.
            m["efficiency"] = round(m["f1"] / m["time_ms"], 4) if m["time_ms"] else 0.0
            m["overall"] = round(0.6 * f1_norm + 0.4 * speed_norm, 4)

        winner = max(models, key=lambda m: (m["overall"], m["f1"], -m["time_ms"]))
        loser = next(m for m in models if m["key"] != winner["key"])
        for m in models:
            m["best"] = m["key"] == winner["key"]

        reasons = []
        if winner["f1"] > loser["f1"]:
            reasons.append(
                f"higher F1 ({winner['f1']} vs {loser['f1']})")
        elif winner["f1"] == loser["f1"]:
            reasons.append(f"matching F1 ({winner['f1']})")
        if winner["time_ms"] < loser["time_ms"]:
            reasons.append(
                f"faster ({winner['time_ms']} ms vs {loser['time_ms']} ms)")
        verdict = (
            f"{winner['name']} is the better model for this query - "
            + " and ".join(reasons) + "."
            if reasons else
            f"{winner['name']} is the better model for this query."
        )

        return {
            "query": query,
            "collection_size": universe_size,
            "relevant_count": len(relevant),
            "relevant_basis": "spell-corrected TF-IDF score >= 30% of the top score",
            "runs": max(bool_runs, fuzzy_runs),
            "winner": winner["key"],
            "verdict": verdict,
            "models": models,
        }


index = SearchIndex(load_scenarios())


@app.get("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/categories")
def categories():
    tree = defaultdict(Counter)
    for s in index.scenarios.values():
        tree[s["category"]][s["subcategory"]] += 1
    return jsonify({"categories": [
        {
            "name": cat,
            "count": sum(subs.values()),
            "subcategories": [
                {"name": name, "count": count}
                for name, count in sorted(subs.items())
            ],
        }
        for cat, subs in sorted(tree.items())
    ]})


@app.get("/api/search")
def search():
    query = request.args.get("q", "")
    mode = request.args.get("mode", "vector")
    if mode not in {"vector", "boolean", "fuzzy"}:
        return jsonify({"error": "Mode must be 'vector', 'fuzzy' or 'boolean'."}), 400
    try:
        results = index.search(
            query, mode,
            category=request.args.get("category", ""),
            subcategory=request.args.get("subcategory", ""),
            difficulty=request.args.get("difficulty", ""),
        )
    except ValueError as exc:
        return jsonify({
            "error": (
                f"Could not read the search: {exc}. Use AND, OR, NOT and "
                "parentheses in capitals, for example: (friend OR neighbour) AND cafe"
            ),
        }), 400
    return jsonify({
        "query": query,
        "mode": mode,
        "count": len(results),
        "results": results,
    })


@app.get("/api/compare")
def compare():
    query = request.args.get("q", "")
    if not query.strip():
        return jsonify({
            "error": "Enter a query to compare the Boolean and Fuzzy models.",
        }), 400
    try:
        report = index.compare(
            query,
            category=request.args.get("category", ""),
            subcategory=request.args.get("subcategory", ""),
            difficulty=request.args.get("difficulty", ""),
        )
    except ValueError as exc:
        return jsonify({
            "error": (
                f"Could not read the search: {exc}. Use AND, OR, NOT and "
                "parentheses in capitals, for example: (friend OR neighbour) AND cafe"
            ),
        }), 400
    return jsonify(report)


@app.get("/api/scenarios/<sid>")
def scenario(sid):
    s = index.scenarios.get(sid)
    if s is None:
        abort(404)
    return jsonify({
        "scenario": s,
        "breadcrumb": [
            {"label": "All scenarios"},
            {"label": s["category"], "category": s["category"]},
            {"label": s["subcategory"], "category": s["category"],
             "subcategory": s["subcategory"]},
            {"label": s["title"], "scenario_id": sid},
        ],
        "related": index.related(sid),
    })


if __name__ == "__main__":
    app.run()
