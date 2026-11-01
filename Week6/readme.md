# Practice-scenario browser — Boolean vs Fuzzy comparison

A small Flask IR app over a corpus of social-rehab conversation scenarios
(`data/scenarios.json`). It offers three retrieval modes — **Relevance**
(TF-IDF vector), **Fuzzy**, and **Boolean** — plus a **model comparison module**
that answers one query with both the Boolean and Fuzzy models and reports which
one is best, with accuracy, speed and efficiency figures.

All code referenced below lives in `app.py`; the UI lives in `static/index.html`.

## Running

```bash
pip install flask
python app.py
```

Open <http://127.0.0.1:5000>, set **Search type → "Compare Boolean vs Fuzzy"**,
type a query (e.g. `friend OR cafe`, `family gathering`, `neighbour AND favour`)
and press Search.

> **Note:** `app.py` runs without the auto-reloader, so after editing the code
> you must stop the server (Ctrl+C) and run `python app.py` again — otherwise a
> stale process keeps serving the old code.

The comparison is exposed at `GET /api/compare?q=<query>` (optional
`category`, `subcategory`, `difficulty` filters). The endpoint is
`compare()` at **app.py:547**; the work is done by `SearchIndex.compare(...)`
at **app.py:387–489**.

---

## How the metrics are calculated

### 0. The query is normalised first — `_normalize_query` (app.py:281–303)

The Boolean parser needs explicit operators, while the Fuzzy parser treats
adjacent words as AND. Before anything is measured, the query is normalised so
**both models answer the same query**: implicit ANDs are made explicit
(`family gathering` → `family AND gathering`). This does not change Fuzzy's
result (it already implicit-ANDs) but lets Boolean accept plain multi-word
queries. Applied at **app.py:395**, then fed to both models at
**app.py:400, 404, 405, 411, 413**.

### 1. The "gold" relevant set — `relevant_set` (app.py:316–341)

Metrics like precision/recall need a set of documents that are *truly* relevant
to the query. This corpus has no human relevance labels, so we build a
transparent reference from the query's **intent**:

1. Tokenise the query and drop the `not` operator word (app.py:327).
2. **Spell-correct** every content word to its nearest corpus word via
   `_correct_term` (app.py:305–314), which uses `difflib.SequenceMatcher`
   and accepts a match at ratio ≥ 0.70. So `frend` → `friend`.
3. Rank every document by **TF-IDF cosine** on that corrected query
   (app.py:329–337), reusing the vector model's `_vectorize`.
4. A document is **relevant** when its score is at least **30 % of the top
   score** (`cutoff = 0.3 * max(...)`, app.py:340–341).

Because the reference is built from TF-IDF — not from Boolean or Fuzzy — neither
model under test defines its own ground truth. Reported to the client as
`relevant_basis` and `relevant_count` (app.py:483–484).

### 2. Precision, Recall, F1, Accuracy — `_prf` (app.py:343–364)

For each model, its retrieved id set is compared against the relevant set over
the filtered universe of `universe_size` documents (app.py:346–354):

| Quantity | Code | Meaning |
|---|---|---|
| TP (true positives)  | `len(retrieved & relevant)`      | app.py:346 | retrieved **and** relevant |
| FP (false positives) | `len(retrieved - relevant)`      | app.py:347 | retrieved but **not** relevant |
| FN (false negatives) | `len(relevant - retrieved)`      | app.py:348 | relevant but **missed** |
| TN (true negatives)  | `universe_size - TP - FP - FN`   | app.py:349 | correctly not retrieved |

```
Precision = TP / (TP + FP)                 # app.py:350  — how much of what it returned was relevant
Recall    = TP / (TP + FN)                 # app.py:351  — how much of the relevant set it found
F1        = 2·P·R / (P + R)                # app.py:352  — harmonic mean of precision and recall
Accuracy  = (TP + TN) / universe_size      # app.py:354  — fraction of all documents classified correctly
```

Each divides by zero safely (returns `0.0`) and the result is rounded to 4
decimals (app.py:360–363). `_prf` is spread into each model's record at
**app.py:435 (Boolean)** and **app.py:445 (Fuzzy)**.

### 3. Speed — `_time_retrieval` (app.py:366–385)

Wall-clock time of the **core retrieval call** in milliseconds, averaged over
several runs for stability:

1. One warm-up call measures the per-call cost `single` (app.py:374–376).
2. The number of timed runs is chosen to fit a **0.15 s budget**, clamped to
   `[3, max_runs]` (`runs = min(max_runs, max(3, int(budget/single)))`,
   app.py:377–380) so the slow Fuzzy model can't stall the request.
3. The timed loop runs and returns `(elapsed / runs) * 1000` ms and the run
   count (app.py:381–385).

Called for each model at **app.py:410–413**; stored as `time_ms` and `runs`
(app.py:431–432, 441–442). Lower is better.

### 4. Efficiency (app.py:456)

```
efficiency = F1 / time_ms          # quality bought per millisecond
```

How much retrieval quality each model delivers per unit of time. Higher is
better.

### 5. Overall score and the winner (app.py:449–478)

F1 (quality) and speed are normalised across the two models, then combined:

```
f1_norm    = F1 / best_F1                  # app.py:453  (best_F1  = larger F1 of the two)
speed_norm = best_ms / time_ms             # app.py:454  (best_ms  = smaller time; fastest → 1.0)
overall    = 0.6 · f1_norm + 0.4 · speed_norm   # app.py:457
```

So the verdict weights **quality 60 % / speed 40 %**. The winner is the model
with the highest `overall`, tie-broken by F1 then by speed
(`max(..., key=(overall, f1, -time_ms))`, app.py:459). The `best` flag
(app.py:461–462) and a plain-English `verdict` string explaining *why*
(app.py:464–478) are returned to the UI.

---

## What the UI shows — `static/index.html`

Selecting **"Compare Boolean vs Fuzzy"** routes rendering to
`renderComparison()`, which calls `/api/compare` and draws:

- a **verdict banner** naming the winning model and the reason;
- a **metrics table** (accuracy, precision, recall, F1, speed, efficiency,
  results returned, overall) with the winning column highlighted — built by
  `metricRow(...)`;
- the **result lists** each model actually returned.

## Interpreting the results on this corpus

With the models **as implemented**, Boolean usually wins:

- **Boolean** does exact term matching → high precision/F1 and is far faster
  (~0.01 ms vs ~150 ms).
- **Fuzzy** (`fuzzy_scores`, app.py:148) counts any partial match
  (`score > 0`) as a hit, so it over-retrieves → recall 1.0 but low precision,
  hence low F1. It also only scores query words that already exist in the
  vocabulary, so it does not rescue query-side typos (`frend` returns nothing).

The comparison module reports this faithfully rather than engineering a tie —
it measures the models as they are.
