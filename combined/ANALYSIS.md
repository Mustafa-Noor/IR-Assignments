# Combined IR system: every model on SODA

**Setup.** 15 retrieval models from Weeks 1–7, plus Naive Bayes, K-Means, filtering and three browsing models, on real
dialogues from [SODA](https://huggingface.co/datasets/allenai/soda) (Kim et al., EMNLP 2023; CC BY 4.0).

| | |
|---|---|
| Documents | 200,010 dialogues, whole events taken in a fixed random order |
| Queries | 2,000 test queries, plus 500 separate dev queries for tuning |
| Relevance | A dialogue is relevant if it came from the same event as the query, about 11 per query |
| Not indexed | `head`, `tail` and `literal`, which hold the answer, and character names |
| Tracks | Dialogue + narrative, and Dialogue only |

**Collection size.** The goal is all 1,486,896 dialogues. The full index is built, but scoring it takes hours on a laptop and
has not finished, so **no score here comes from the full set**. The 200,010 dialogues are the largest evaluated collection.
The results reflect that sample, and the full run may change them.

Regenerate:

```
python -m data.fetch_soda
python -m combined.collection 200000
python -m combined.evaluate 200000
python -m combined.extras 10000
python -m combined.evaluate full 200000     # the full set, with settings tuned on 200,000 (takes hours)
```

Runs repeat exactly: two runs of the 10,000-dialogue collection under different Python hash seeds gave identical scores.

## Scoreboard

Top 10 results per query, averaged over the 2,000 test queries. Sorted by F1@10.

| Model | Week | Dialogue + narrative: P@10 | R@10 | F1@10 | MAP | Dialogue only: F1@10 | MAP |
|---|---|---|---|---|---|---|---|
| **Field BM25** | 4 | 0.383 | 0.373 | 0.364 | 0.380 | 0.091 | 0.082 |
| Proximal nodes | 4 | 0.325 | 0.320 | 0.311 | 0.318 | 0.086 | 0.077 |
| Overlapped lists | 4 | 0.389 | 0.264 | 0.295 | 0.246 | 0.065 | 0.041 |
| **BM25** | 4 | 0.285 | 0.282 | 0.273 | 0.283 | 0.095 | 0.086 |
| Extended Boolean | 6 | 0.318 | 0.243 | 0.260 | 0.224 | 0.072 | 0.049 |
| Term overlap | 1 | 0.248 | 0.247 | 0.238 | 0.237 | 0.071 | 0.061 |
| Boolean | 3 | 0.269 | 0.196 | 0.214 | does not rank | 0.061 | does not rank |
| BIM | 4 | 0.209 | 0.210 | 0.202 | 0.211 | 0.068 | 0.063 |
| Non-overlapped lists | 4 | 0.197 | 0.198 | 0.190 | 0.191 | 0.064 | 0.055 |
| Fuzzy (AND) | 6 | 0.211 | 0.187 | 0.190 | 0.177 | 0.056 | 0.039 |
| VSM cosine | 3 | 0.162 | 0.159 | 0.154 | 0.154 | 0.069 | 0.063 |
| GVSM | 7 | 0.155 | 0.152 | 0.147 | 0.149 | 0.067 | 0.061 |
| Fuzzy (OR) | 6 | 0.133 | 0.131 | 0.127 | 0.113 | 0.038 | 0.028 |
| LSI (k = 800) | 7 | 0.093 | 0.091 | 0.089 | 0.083 | 0.024 | 0.020 |
| TF-IDF sum | 2 | 0.085 | 0.085 | 0.081 | 0.082 | 0.036 | 0.032 |

## Read this before trusting the numbers

- **Accuracy says almost nothing here.** Every model scores about 0.9999. A system that retrieves nothing scores 0.999946,
  **higher than all 15 models**, because each wrong result in the top 10 costs accuracy and almost every dialogue is
  correctly left out. The app shows accuracy as requested, beside this baseline and balanced accuracy. Use precision,
  recall and F1 to compare models.
- **The labels are strict.** Only dialogues from the exact same event count, so an on-topic dialogue from a similar event
  scores as wrong. This hurts every model and semantic models most, so LSI and GVSM scores are lower bounds.
- **The narrative inflates one track.** SODA generated each narrative from its event, so it often repeats the query's
  words. Best MAP is 0.380 with it and 0.086 without. Tuning even chose "narrative only" as the best field weighting.
  The Dialogue-only track is the fairer test.
- **Scores depend on collection size.** More dialogues mean more competition. BM25's MAP on the Dialogue + narrative track
  fell from about 0.54 at 10,000 dialogues to 0.28 at 200,000. Expect lower scores on the full set.

## What works

1. **BM25 is the best general ranker.** It leads the Dialogue-only track (F1@10 0.095), runs in about 2 ms per query, and is
   clearly ahead of BIM (0.273 vs 0.202 on the other track). Using term frequency and length pays off.
2. **Structure helps when it points at the right text.** Field BM25 and Proximal nodes top the Dialogue + narrative track
   because they can favour the narrative. On Dialogue only, where no field stands out, neither beats plain BM25.
3. **Strict matching gives precision.** Overlapped lists have the best precision@10 on both tracks (0.389 and 0.126) at the
   cost of recall, as Week 4's slides predict.
4. **Fuzzy matching handles typos.** `frend` finds `friend` at similarity 0.91, where the Boolean model finds nothing.

## What doesn't work

1. **Raw TF-IDF sum.** Long, repetitive dialogues dominate because term frequency has no ceiling (F1@10 0.081). Tuning
   agreed: the best k1 is a low 0.5, so repeating a word adds little here.
2. **Fuzzy OR.** Any one common word is enough, so it retrieves about 993 dialogues at precision 0.005.
3. **LSI at this scale.** It reached MAP 0.473 at 10,000 dialogues but 0.083 at 200,000, with k = 800 both times.
   More concepts help: on dev queries MAP rose from 0.036 at k = 300 to 0.071 at k = 800, the largest value tried, so k may
   still be too low. Even so it stays far below BM25.
4. **GVSM's gain is modest.** It sits next to VSM (0.147 vs 0.154). The full `G` cannot be computed at this size, so it uses a
   sampled correlation for each query word's closest terms.

## Do semantic models earn their place?

Share of relevant dialogues found in the top 1,000 that share **no** query word:

| Model | Dialogue + narrative | Dialogue only |
|---|---|---|
| LSI | 0.030 | 0.021 |
| GVSM | 0.032 | 0.017 |
| Fuzzy (AND / OR) | 0.007 / 0.007 | 0.003 / 0.005 |
| All word-matching models | 0 | 0 |

They find documents that nothing else can, which is the promise of Week 7's slides, but in small amounts here. The same
measure was 0.31 for LSI at 10,000 dialogues, so this gap also shrinks as the collection grows. Raising k from 300 to 800
improved LSI's ranking but lowered this measure from 0.046 to 0.030, because results lean closer to the query's own words.

## Set models

Scored on everything they retrieve, on the Dialogue + narrative track:

| Model | Average retrieved | Precision | Recall | F1 |
|---|---|---|---|---|
| Overlapped lists | 156 | 0.378 | 0.512 | 0.325 |
| Extended Boolean | 253 | 0.269 | 0.535 | 0.255 |
| Boolean | 1,451 | 0.269 | 0.576 | 0.254 |
| Fuzzy (AND) | 439 | 0.128 | 0.542 | 0.144 |
| Fuzzy (OR) | 993 | 0.005 | 0.416 | 0.009 |

Fuzzy and Extended Boolean results are cut at the top 1,000, so Boolean's average is not directly comparable.

## Classification and clustering

SODA has no topic labels. The only predefined category is `relation`, the kind of inference behind a dialogue, and that is
not a topic.

| 200,010 dialogues | Accuracy or purity | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Naive Bayes | 0.368 (guessing the commonest class: 0.255) | 0.330 | 0.330 | 0.325 |
| K-Means | 0.260 | 0.086 | 0.177 | 0.115 |

Naive Bayes does better than guessing, and it was trained on SODA's train split and scored on unseen events. K-Means was
sampled to 50,000 dialogues and did not recover `relation`, as expected, because it groups by subject matter.

## Filtering and browsing (10,005 dialogues)

These were measured on the 10,000-dialogue collection only.

**Filtering** with 30 profiles on a stream of about 2,000 dialogues: best F1 is 0.498 at θ = 0.15 (precision 0.490, recall 0.619).

**Browsing against search**, in median items examined before reaching a relevant dialogue:

| Approach | Items | Minutes at 200 words per minute |
|---|---|---|
| Search with BM25 | 1 | 0.6 |
| Structured guided browsing | 25 | 13.9 |
| Hypertext links (best path) | 37 | 20.8 |
| Flat browsing | 801 | 445.7 |

Guided browsing assumes the user always picks the right menu entry, and hypertext is the shortest possible path, so both are
best cases. Browsing still suits users who cannot name what they want, which this measure does not capture.

## Recommendation

| Need | Use |
|---|---|
| Default ranking | **BM25** |
| Documents with a strong summary field | **Field BM25**, with weights tuned on data |
| High-precision results | **Overlapped lists** or **Extended Boolean** |
| Typing mistakes | **Fuzzy (AND)** as a fallback when exact matching finds nothing |
| Avoid | Raw TF-IDF sum, Fuzzy OR, and LSI with few concepts at this scale |

## Known limits

- No score comes from the full 1,486,896 dialogues.
- The 2,000 queries are events with at least 5 dialogues, so they are not typical user searches.
- Some gaps are small. No significance test was run, so treat close scores as ties.
- Fast array models were checked against the original Week 1–4 classes on 60 queries over 10,005 dialogues. VSM was compared
  on its top 10 only.
