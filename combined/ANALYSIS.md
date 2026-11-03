# Combined IR system: every model on SODA

**Setup.** 15 retrieval models from Weeks 1–7, plus Naive Bayes, K-Means, filtering and three browsing models, on real
dialogues from [SODA](https://huggingface.co/datasets/allenai/soda) (Kim et al., EMNLP 2023; CC BY 4.0).

| | |
|---|---|
| Documents | **All 1,486,896 dialogues.** 10,005 and 200,010 subsets were also scored, to show how size changes things |
| Queries | 2,000 test queries, plus 500 separate dev queries for tuning, the same at every size |
| Relevance | A dialogue is relevant if it came from the same event as the query, about 11 per query |
| Not indexed | `head`, `tail` and `literal`, which hold the answer, and character names |
| Tracks | Dialogue + narrative, and Dialogue only |
| Settings | Tuned on the dev queries of the 200,010 collection, then used unchanged on the full set |

Regenerate (the full set needs about 9 GB of disk and an hour or two):

```
python -m data.fetch_soda
python -m combined.collection 200000
python -m combined.evaluate 200000
python -m combined.collection full
python -m combined.evaluate full 200000
python -m combined.extras full
```

Runs repeat exactly: two runs of the 10,000-dialogue collection under different Python hash seeds gave identical scores.

## Scoreboard, all 1,486,896 dialogues

Top 10 results per query, averaged over the 2,000 test queries. Sorted by F1@10.

| Model | Week | Dialogue + narrative: P@10 | R@10 | F1@10 | MAP | Dialogue only: F1@10 | MAP |
|---|---|---|---|---|---|---|---|
| **Field BM25** | 4 | 0.233 | 0.229 | 0.223 | 0.220 | 0.043 | 0.037 |
| Proximal nodes | 4 | 0.186 | 0.184 | 0.179 | 0.176 | 0.041 | 0.034 |
| Overlapped lists | 4 | 0.207 | 0.149 | 0.162 | 0.136 | 0.036 | 0.022 |
| **BM25** | 4 | 0.153 | 0.154 | 0.148 | 0.145 | 0.045 | 0.039 |
| Extended Boolean | 6 | 0.157 | 0.129 | 0.134 | 0.115 | 0.035 | 0.023 |
| Term overlap | 1 | 0.126 | 0.127 | 0.122 | 0.117 | 0.033 | 0.026 |
| Boolean | 3 | 0.121 | 0.094 | 0.100 | does not rank | 0.028 | does not rank |
| Fuzzy (AND) | 6 | 0.106 | 0.097 | 0.097 | 0.087 | 0.026 | 0.018 |
| BIM | 4 | 0.097 | 0.099 | 0.095 | 0.097 | 0.029 | 0.025 |
| Non-overlapped lists | 4 | 0.091 | 0.094 | 0.089 | 0.087 | 0.028 | 0.022 |
| VSM cosine | 3 | 0.072 | 0.073 | 0.069 | 0.065 | 0.030 | 0.025 |
| GVSM | 7 | 0.070 | 0.071 | 0.068 | 0.062 | 0.028 | 0.024 |
| Fuzzy (OR) | 6 | 0.069 | 0.068 | 0.066 | 0.056 | 0.017 | 0.012 |
| LSI | 7 | 0.030 | 0.029 | 0.028 | 0.023 | 0.006 | 0.004 |
| TF-IDF sum | 2 | 0.027 | 0.027 | 0.026 | 0.024 | 0.011 | 0.009 |

## Read this before trusting the numbers

- **Accuracy says almost nothing here.** Every model scores about 0.99999. A system that retrieves nothing scores
  0.999993, **higher than all 15 models**, because each wrong result in the top 10 costs accuracy and almost every
  dialogue is correctly left out. The app shows accuracy as requested, beside this baseline and balanced accuracy. Use
  precision, recall and F1 to compare models.
- **The labels are strict.** Only dialogues from the exact same event count, so an on-topic dialogue from a similar event
  scores as wrong. This hurts every model and semantic models most, so LSI and GVSM scores are lower bounds.
- **The narrative inflates one track.** SODA generated each narrative from its event, so it often repeats the query's
  words. Best MAP is 0.220 with it and 0.039 without. Tuning even chose "narrative only" as the best field
  weighting. The Dialogue-only track is the fairer test.
- **Scores fall as the collection grows.** More dialogues means more competition for the top 10:

| MAP | Dialogue + narrative: 10,005 | 200,010 | 1,486,896 | Dialogue only: 10,005 | 200,010 | 1,486,896 |
|---|---|---|---|---|---|---|
| Field BM25 | 0.624 | 0.380 | 0.220 | 0.220 | 0.082 | 0.037 |
| BM25 | 0.541 | 0.283 | 0.145 | 0.225 | 0.086 | 0.039 |
| GVSM | 0.416 | 0.149 | 0.062 | 0.209 | 0.061 | 0.024 |
| LSI | 0.473 | 0.083 | 0.023 | 0.188 | 0.020 | 0.004 |

  The ranking of models barely changes with size. LSI falls fastest.

## What works

1. **BM25 is the best general ranker.** It leads the Dialogue-only track (F1@10 0.045) and is clearly ahead of
   BIM (0.148 vs 0.095 on the other track). Using term frequency and length pays off.
2. **Structure helps when it points at the right text.** Field BM25 and Proximal nodes top the Dialogue + narrative track
   because they can favour the narrative. On Dialogue only, where no field stands out, neither beats plain BM25.
3. **Strict matching gives precision.** Overlapped lists have the best precision@10 on the Dialogue-only track
   (0.063) at the cost of recall, as Week 4's slides predict. On the other track Field BM25 overtakes them at full size.
4. **Fuzzy matching handles typos.** `frend` finds `friend` at similarity 0.91, where the Boolean model finds nothing.

## What doesn't work

1. **Raw TF-IDF sum.** Long, repetitive dialogues dominate because term frequency has no ceiling (F1@10
   0.026). Tuning agreed: the best k1 is a low 0.5, so repeating a word adds little here.
2. **Fuzzy OR.** Any one common word is enough, so it retrieves about 998 dialogues at precision
   0.003.
3. **LSI at this scale.** MAP 0.473 at 10,005 dialogues falls to 0.023 on the full set, all with k = 800
   concepts. On dev queries MAP still rose from k = 300 to k = 800, the largest value tried, so k may be too low for this
   many dialogues.
4. **GVSM's gain is modest.** It sits next to VSM (0.068 vs 0.069). The full `G` cannot be computed at this
   size, so it uses a sampled correlation for each query word's closest terms.

## Do semantic models earn their place?

Share of relevant dialogues found in the top 1,000 that share **no** query word:

| Model | Dialogue + narrative | Dialogue only |
|---|---|---|
| LSI | 0.006 | 0.003 |
| GVSM | 0.006 | 0.003 |
| Fuzzy (AND / OR) | 0.003 / 0.002 | 0.001 / 0.000 |
| All word-matching models | 0 | 0 |

They find documents that nothing else can, which is the promise of Week 7's slides, but very few at full size. The same
measure for LSI was 0.312 at 10,005 dialogues and 0.030 at 200,010.

## Set models

Scored on everything they retrieve, on the Dialogue + narrative track:

| Model | Average retrieved | Precision | Recall | F1 |
|---|---|---|---|---|
| Overlapped lists | 351 | 0.197 | 0.457 | 0.184 |
| Extended Boolean | 491 | 0.122 | 0.454 | 0.128 |
| Boolean | 10,760 | 0.122 | 0.576 | 0.127 |
| Fuzzy (AND) | 668 | 0.056 | 0.434 | 0.068 |
| Fuzzy (OR) | 998 | 0.003 | 0.276 | 0.006 |

Fuzzy and Extended Boolean results are cut at the top 1,000, so Boolean's average is not directly comparable.

## Classification and clustering

SODA has no topic labels. The only predefined category is `relation`, the kind of inference behind a dialogue, and that is
not a topic.

| All 1,486,896 dialogues | Accuracy or purity | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Naive Bayes | 0.370 (guessing the commonest class: 0.254) | 0.341 | 0.340 | 0.331 |
| K-Means | 0.265 | 0.087 | 0.177 | 0.112 |

Naive Bayes was trained on SODA's train split and scored on 295,314 dialogues from unseen events. It beats
guessing, but not by much. K-Means was sampled to 50,000 dialogues and did not recover `relation`, as expected,
because it groups by subject matter.

## Filtering and browsing

**Filtering** with 30 profiles on a stream of about 2,000 dialogues: best F1 is
0.501 at θ = 0.15 (precision 0.482, recall 0.612).

**Browsing against search** on the full set, in median items examined before reaching a relevant dialogue:

| Approach | Items | Hours at 200 words per minute |
|---|---|---|
| Search with BM25 (190 of 200 queries found a match) | 7 | 0.06 |
| Hypertext links (best path, on 10,005 dialogues) | 37 | 0.34 |
| Structured guided browsing | 1,758 | 16.2 |
| Flat browsing | 125,467 | 1,154 |

With 1.5 million dialogues, browsing alone stops being practical: even the guided hierarchy leaves about
1,758 items to look through, because each of its leaves holds many events. Guided browsing assumes the
user always picks the right menu entry, and hypertext is the shortest possible path, so both are best cases. Hypertext
links had to be computed pair by pair, so they were measured on 10,005 dialogues only.

## Recommendation

| Need | Use |
|---|---|
| Default ranking | **BM25** |
| Documents with a strong summary field | **Field BM25**, with weights tuned on data |
| High-precision results | **Overlapped lists** or **Extended Boolean** |
| Typing mistakes | **Fuzzy (AND)** as a fallback when exact matching finds nothing |
| Avoid | Raw TF-IDF sum, Fuzzy OR, and LSI with few concepts at this scale |

## Known limits

- The 2,000 queries are events with at least 5 dialogues, so they are not typical user searches.
- Settings were tuned on the 200,010 collection, not re-tuned on the full set.
- Some gaps are small. No significance test was run, so treat close scores as ties.
- Fast array models were checked against the original Week 1–4 classes on 60 queries over 10,005 dialogues. VSM was compared
  on its top 10 only.
- In the app, live Search, Model flow and Browse use the 200,010 collection: on the full set a search through all 15 models
  took 8–12 seconds and 4 GB of memory on this laptop. The Compare page shows the full-set scores.
