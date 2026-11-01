# Information Retrieval — weekly assignments (IR track)

Each week's assignment from the lecture slides is built on the same dataset, so Week 5 can compare all of them fairly.

## Setup

```
pip install -r requirements.txt
streamlit run app.py              # Weeks 1–5, one page per week in the sidebar
python -m combined.server         # Week 7: every model on SODA, http://127.0.0.1:5000
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

| `Week6/` | Fuzzy search with browsing on a small scenario collection | `cd Week6`, then `python app.py` |
| `combined/` | Week 7: all models together on SODA, each with its own flow, compared visually | `python -m combined.server`, then read `combined/ANALYSIS.md` |

The folder names follow the order the work was done, so they differ slightly from the slides. The slides' Week 5 is browsing,
now in `browsing/`. Their Week 6 models are in `setmodels/` and their Week 7 models in `week7/`.

## Week 7 — the combined system on SODA

| Step | Command |
|---|---|
| Download all 1,486,896 dialogues (884 MB) | `python -m data.fetch_soda` |
| Build an index (`10000`, `200000` or `full`) | `python -m combined.collection 200000` |
| Tune and score every model | `python -m combined.evaluate 200000` |
| Filtering and browsing scores | `python -m combined.extras 10000` |
| Open the app | `python -m combined.server` |

The app has six pages: Compare, Search, Model flow, Browse, Tuning, and Math & code, which shows each formula beside the
function that runs it. Saved results in `combined/results/` load without any rebuild, but Search, Flow and Browse need the index.

Scores come from 200,010 dialogues. The full index builds in a few minutes, but scoring it takes hours: use
`python -m combined.evaluate full 200000` on a free machine. It needs about 9 GB of disk.

Later weeks import earlier ones, the same way the course builds up. For example, Week 4's BM25 uses Week 1's
inverted index, and Week 5 wraps every week's model.
