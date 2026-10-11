# Information Retrieval — weekly assignments (IR track)

Each week's assignment from the lecture slides is built on the same dataset, so Week 5 can compare all of them fairly.

## Setup

```
pip install -r requirements.txt
streamlit run app.py              # one page per week in the sidebar
python -m pytest tests            # checks against the slides' worked examples
```

## Dataset — `data/`

| File | Contents |
|---|---|
| `fetch_papers.py` | Downloads 40 recent arXiv papers and parses Introduction, Methodology, Results, Conclusion and References from arXiv's HTML full text. Each section is capped at 300 words. |
| `corpus.json` | The 40 papers in 6 categories: Healthcare AI, Cybersecurity, Information Retrieval, Databases, Software Engineering, Computer Vision. |
| `queries.json` | 15 test queries, each with a Boolean form and hand-judged relevant papers (2 = highly, 1 = partly relevant). |

## Weeks

| Folder | Slides' assignment | Demo |
|---|---|---|
| `week1/` | Simple search engine: tokenization, stop-words, stemming, inverted index, search, term-overlap ranking | `python -m week1.run "query"` |
| `week2/` | Basic Document Ranking Pipeline: formal D, Q, T, F: Q→2^D, R(q,d), swappable weighting w = g(d); ad-hoc retrieval vs filtering | `python -m week2.run` |
| `week3/` | Boolean AND/OR/NOT parser, VSM TF-IDF + cosine, Naive Bayes classification, K-Means clustering, precision/recall | `python -m week3.run` |
| `week4/` | BIM, BM25, P = 1/(1+e^−s), structured field search, overlapped and non-overlapped lists, proximal nodes | `python -m week4.run` |
| `week5/` | All 10 models on all 15 queries: MAP, nDCG@10, P@5, parameter sweeps, failure analysis | `python -m week5.compare`, then read `week5/ANALYSIS.md` |

Later weeks import earlier ones, the same way the course builds up. For example, Week 4's BM25 uses Week 1's
inverted index, and Week 5 wraps every week's model.
