"""Proof that the hand-written parts agree with the standard libraries.

The libraries are imported only here. The models themselves never use them.
"""
import random
from difflib import SequenceMatcher

import numpy as np
import pytest

from data import full_text, load_corpus
from week1.porter import stem_word
from week1.preprocess import STOPWORDS, preprocess, tokenize
from week3.kmeans import kmeans, tfidf_matrix
from week3.naive_bayes import NaiveBayes, cross_validate, stratified_folds
from setmodels.similarity import similarity

CORPUS = load_corpus()
TEXTS = [full_text(d) for d in CORPUS]
LABELS = [d["category"] for d in CORPUS]
WORDS = sorted({t for text in TEXTS for t in tokenize(text) if t not in STOPWORDS})


def test_porter_matches_nltk_on_the_whole_vocabulary():
    nltk_stem = pytest.importorskip("nltk.stem").PorterStemmer().stem
    different = [(w, stem_word(w), nltk_stem(w)) for w in WORDS if stem_word(w) != nltk_stem(w)]
    assert different == []


@pytest.mark.parametrize("word, stem", [("caresses", "caress"), ("ponies", "poni"), ("hopping", "hop"),
                                        ("relational", "relat"), ("hopeful", "hope"), ("revival", "reviv"),
                                        ("controlling", "control"), ("running", "run")])
def test_porter_paper_examples(word, stem):
    assert stem_word(word) == stem


def test_folds_match_sklearn():
    sk = pytest.importorskip("sklearn.model_selection")
    ours = stratified_folds(LABELS, 5, 0)
    for f, (_, test) in enumerate(sk.StratifiedKFold(5, shuffle=True, random_state=0).split(TEXTS, LABELS)):
        assert sorted(np.flatnonzero(ours == f)) == sorted(test)


def test_naive_bayes_matches_sklearn():
    text = pytest.importorskip("sklearn.feature_extraction.text")
    nb = pytest.importorskip("sklearn.naive_bayes")
    pipe = pytest.importorskip("sklearn.pipeline")
    sk = pytest.importorskip("sklearn.model_selection")
    theirs = sk.cross_val_predict(pipe.make_pipeline(text.CountVectorizer(analyzer=preprocess), nb.MultinomialNB()),
                                  TEXTS, LABELS, cv=sk.StratifiedKFold(5, shuffle=True, random_state=0))
    assert cross_validate(TEXTS, LABELS) == list(theirs)


def test_naive_bayes_slide_16_categories():
    model = NaiveBayes(analyzer=str.split).fit(["cat dog", "dog bird", "car road", "road truck"],
                                               ["pet", "pet", "auto", "auto"])
    assert model.predict(["dog cat", "truck", "dog unseenword"]) == ["pet", "auto", "pet"]


def test_tfidf_matches_sklearn():
    text = pytest.importorskip("sklearn.feature_extraction.text")
    theirs = text.TfidfVectorizer(analyzer=preprocess).fit_transform(TEXTS).toarray()
    ours, _ = tfidf_matrix([preprocess(t) for t in TEXTS])
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_kmeans_inertia_close_to_sklearn():
    cluster = pytest.importorskip("sklearn.cluster")
    X, _ = tfidf_matrix([preprocess(t) for t in TEXTS])
    theirs = cluster.KMeans(n_clusters=6, n_init=10, random_state=0).fit(X).inertia_
    assert kmeans(X, 6, seed=0)[2] == pytest.approx(theirs, rel=0.02)


def test_similarity_matches_difflib():
    rng = random.Random(0)
    pairs = [tuple(rng.sample(WORDS, 2)) for _ in range(3000)] + [("friend", "frend"), ("", ""), ("abc", "xyz")]
    for a, b in pairs:
        assert similarity(a, b) == pytest.approx(SequenceMatcher(None, a, b, autojunk=False).ratio(), abs=1e-12)
