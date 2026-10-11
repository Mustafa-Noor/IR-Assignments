"""Week 1 — Text operations (slide 11): tokenization, stop-word removal, stemming."""
import re

from nltk.stem import PorterStemmer

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either et etc few for from further
had has have having he her here hers herself him himself his how however i if in into is it its itself
just let may me might more most must my myself no nor not of off on once only or other our ours
ourselves out over own per same she should so some such than that the their theirs them themselves
then there these they this those through thus to too under until up upon us very via was we were what
when where whether which while who whom why will with within without would yet you your yours yourself
yourselves al one two three use used using also e g ie eg
""".split())

_stemmer = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lower-case and split into alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem(tokens):
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text):
    """Full pipeline: text -> tokens -> no stop-words -> stems."""
    return stem(remove_stopwords(tokenize(text)))


def explain(text):
    """Every intermediate step, for the UI."""
    tokens = tokenize(text)
    kept = remove_stopwords(tokens)
    return {"tokens": tokens, "without_stopwords": kept, "stems": stem(kept)}
