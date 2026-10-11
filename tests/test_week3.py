"""Checks Week 3 against the worked examples in the slides.  Run: python -m pytest tests"""
import pytest

from week1.index import InvertedIndex
from week3.boolean import BooleanModel, BooleanQueryError
from week3.vsm import VectorSpaceModel

# Slide 14 — Computed Example of Vector Ranking
SLIDE14 = {
    "D1": "retrieval indexing ranking",
    "D2": "ranking search engine",
    "D3": "database normalization",
}


def test_slide14_cosine_scores():
    vsm = VectorSpaceModel(SLIDE14, weighting="binary", analyzer=str.split)
    qv = vsm.query_vector("retrieval ranking")
    assert vsm.cosine(qv, "D1") == pytest.approx(0.816, abs=1e-3)
    assert vsm.cosine(qv, "D2") == pytest.approx(0.408, abs=1e-3)
    assert vsm.cosine(qv, "D3") == 0.0


# Slide 8 — Boolean mathematical view
SLIDE8 = {
    "D1": "information retrieval indexing",
    "D2": "search engine ranking",
    "D3": "indexing ranking retrieval",
}


def test_slide8_boolean_and():
    model = BooleanModel(InvertedIndex(SLIDE8))
    assert model.search("indexing AND ranking") == {"D3"}
    assert model.rel("indexing AND ranking", "D1") == 0


def test_boolean_or_not_and_precedence():
    model = BooleanModel(InvertedIndex(SLIDE8))
    assert model.search("indexing OR engine") == {"D1", "D2", "D3"}
    assert model.search("ranking NOT retrieval") == {"D2"}           # implicit AND
    assert model.search("NOT ranking OR search") == {"D1", "D2"}      # (NOT ranking) OR search
    assert model.search("retrieval AND (search OR information)") == {"D1"}


def test_boolean_errors():
    model = BooleanModel(InvertedIndex(SLIDE8))
    with pytest.raises(BooleanQueryError):
        model.search("(ranking AND retrieval")
    with pytest.raises(BooleanQueryError):
        model.search("ranking AND")
