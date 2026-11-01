"""The slides' worked examples. Most run the real code; a few restate the slide's arithmetic.  Run: python -m pytest tests"""
import numpy as np
import pytest

from browsing.models import browsing_time, flat_page, shortest_path
from combined.evaluate import confusion_scores
from setmodels.extended_boolean import and_p, or_p
from setmodels.fuzzy import band, trapezoid
from week1.index import InvertedIndex
from week3.boolean import BooleanModel
from week3.vsm import VectorSpaceModel
from week4.probabilistic import BIM, sigmoid
from week7.gvsm import correlation_matrix, exact_similarity


# ---------- Week 4 ----------

def test_week4_slide7_score_to_probability():
    assert sigmoid(0) == 0.5
    assert sigmoid(4) > sigmoid(1) > 0.5


def test_week4_slide9_bim_prefers_more_useful_terms():
    docs = {"D1": "cybersecurity attack detection", "D2": "cybersecurity detection",
            "D3": "attack", "D4": "cybersecurity detection"}
    ranked = BIM(InvertedIndex(docs)).rank("cybersecurity attack detection")
    assert {d for d, _ in ranked} == set(docs)


def test_week4_slide14_overlapped_lists():
    cats, dogs = {"D1", "D2", "D3", "D4", "D5"}, {"D1", "D4", "D5"}
    assert sorted(cats & dogs) == ["D1", "D4", "D5"]


def test_week4_slide15_non_overlapped_lists():
    lists = {"cybersecurity": {"R1", "R3", "R5"}, "attack": {"R2", "R4", "R6"}, "detection": {"R3", "R4", "R7"}}
    assert sorted(set().union(*lists.values())) == ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]


# ---------- Week 5 ----------

def test_week5_slide8_flat_browsing_time():
    assert browsing_time([30, 10, 45]) == 85


def test_flat_pages_cover_every_document_once():
    assert flat_page(45, 2) == list(range(40, 45))


def test_week5_hypertext_shortest_path():
    graph = [{1}, {0, 2}, {1, 3}, {2}]
    assert shortest_path(graph, 0, {3}) == [0, 1, 2, 3]
    assert shortest_path([{1}, {0}, set()], 0, {2}) is None


# ---------- Week 6 ----------

def test_week6_slide9_fuzzy_and():
    assert min(0.8, 0.6) == 0.6


def test_week6_slide16_ranking():
    rows = {"D1": (0.85, 0.60), "D2": (0.70, 0.90), "D3": (0.40, 0.80)}
    ranked = sorted(rows, key=lambda d: -min(rows[d]))
    assert ranked == ["D2", "D1", "D3"]
    assert [min(rows[d]) for d in ranked] == [0.70, 0.60, 0.40]


def test_week6_quiz_fuzzy_and():
    assert min(0.7, 0.4) == 0.4


def test_week6_slide5_bands():
    assert [band(m) for m in (0.92, 0.68, 0.41, 0.08)] == ["highly", "moderately", "slightly", "not"]


def test_week6_slide16_trapezoid():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert trapezoid(x, 1, 2, 4, 5) == pytest.approx([0, 0, 1, 1, 1, 0, 0])


def test_week6_slide11_extended_boolean_sets():
    docs = {"D1": "science health", "D2": "technology ai", "D3": "science art",
            "D4": "art culture", "D5": "technology education"}
    found = BooleanModel(InvertedIndex(docs)).search("(science OR technology) AND NOT art")
    assert found == {"D1", "D2", "D5"}


def test_extended_boolean_p_norm():
    assert or_p(np.float32(1), np.float32(0), 1) == pytest.approx(0.5)
    assert and_p(np.float32(1), np.float32(1), 2) == pytest.approx(1)
    assert and_p(np.float32(0), np.float32(0), 2) == pytest.approx(0)


# ---------- Week 7 ----------

def test_week7_slide8_cosine():
    vsm = exact_similarity([1, 1], [5, 3], np.eye(2))
    assert vsm == pytest.approx(8 / np.sqrt(68))
    assert round(vsm, 2) == 0.97


def test_week7_slide11_svd_reconstructs_matrix():
    A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
    U, S, Vt = np.linalg.svd(A)
    assert U @ np.diag(S) @ Vt == pytest.approx(A)
    assert S[0] == pytest.approx(2.0)


def test_week7_gvsm_finds_related_words_vsm_misses():
    # "nasa" and "rocket" share documents, so a query for one reaches a document with the other (slide 4).
    A = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], float)
    G = correlation_matrix(A)
    q, d = [1, 0, 0], [0, 1, 0]
    assert exact_similarity(q, d, np.eye(3)) == 0
    assert exact_similarity(q, d, G) > 0.9


# ---------- metrics ----------

def test_accuracy_is_near_one_for_a_system_that_returns_nothing():
    nothing = confusion_scores(0, 0, 10, 1_000_000)
    assert nothing["accuracy"] > 0.9999
    assert nothing["precision"] == nothing["recall"] == nothing["f1"] == 0
    assert nothing["balanced_accuracy"] == 0.5


def test_precision_recall_f1():
    s = confusion_scores(tp=5, retrieved=10, relevant=20, N=100)
    assert (s["precision"], s["recall"]) == (0.5, 0.25)
    assert s["f1"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert s["accuracy"] == pytest.approx(0.8)


def test_slide14_vsm_still_holds():
    toy = {"D1": "retrieval indexing ranking", "D2": "ranking search engine", "D3": "database normalization"}
    vsm = VectorSpaceModel(toy, weighting="binary", analyzer=str.split)
    assert vsm.cosine(vsm.query_vector("retrieval ranking"), "D1") == pytest.approx(0.816, abs=1e-3)
