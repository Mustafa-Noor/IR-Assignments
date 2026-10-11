"""Shared dataset used by every week, so all models are compared on identical data."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Structured fields of each paper, in reading order.
FIELDS = ["title", "abstract", "introduction", "methodology", "results", "conclusion", "references"]


def load_corpus():
    """List of paper dicts (see data/fetch_papers.py for the schema)."""
    return json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))


def load_queries():
    """List of {id, text, boolean, relevant: {doc_id: grade}} — grade 2 = highly, 1 = partially relevant."""
    return json.loads((DATA_DIR / "queries.json").read_text(encoding="utf-8"))


def full_text(doc):
    """Whole paper as one flat string (what Weeks 1–3 search over)."""
    return "\n".join(doc.get(f, "") for f in FIELDS)
