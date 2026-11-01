"""What the app shows about each model: its idea, formula, slides, and where the code lives.

family follows the taxonomy of the Week 2 slides. source lists (file, function or class) pairs; the app
reads those definitions from the real files, so the Math & code page cannot drift from the code.
"""
FAMILIES = ["foundations", "classical", "probabilistic", "structured", "set-theoretic", "algebraic"]

MODELS = {
    "Term overlap": {
        "week": 1, "family": "foundations", "slides": "Week 1, slides 13–14, 17",
        "idea": "Count how many query words a document shares.",
        "formula": [r"\text{rank by } |q \cap d|,\ \text{then } \sum_{t \in q \cap d} tf(t,d)"],
        "source": [("week1/search.py", "rank_overlap"), ("combined/fast/rankers.py", "Rankers.overlap")],
    },
    "TF-IDF sum": {
        "week": 2, "family": "foundations", "slides": "Week 2, slides 20–22",
        "idea": "The formal pipeline: F picks candidates, R scores them with a swappable weight g.",
        "formula": [r"F: Q \rightarrow 2^D", r"R(q,d) = \sum_{t \in q} w_{t,d}, \quad w_{t,d} = g_t(d)",
                    r"g_{\text{tf-idf}} = tf \cdot \log_{10}\frac{N}{df}"],
        "source": [("week2/framework.py", "IRSystem"), ("combined/fast/rankers.py", "Rankers.weighted_sum")],
    },
    "Boolean": {
        "week": 3, "family": "classical", "slides": "Week 3, slides 5–9",
        "idea": "A document either satisfies the logical query or it does not. No ranking.",
        "formula": [r"Rel(D,Q) = \begin{cases} 1 & D \text{ satisfies } Q \\ 0 & \text{otherwise} \end{cases}"],
        "source": [("week3/boolean.py", "to_postfix"), ("week3/boolean.py", "BooleanModel"),
                   ("combined/fast/rankers.py", "Rankers.boolean_mask")],
    },
    "VSM cosine": {
        "week": 3, "family": "classical", "slides": "Week 3, slides 10–14",
        "idea": "Documents and queries are vectors; closer direction means more relevant.",
        "formula": [r"TF\text{-}IDF(t,d) = tf(t,d) \cdot \log_{10}\frac{N}{df(t)}",
                    r"\cos(q,d) = \frac{q \cdot d}{\|q\|\,\|d\|}"],
        "source": [("week3/vsm.py", "VectorSpaceModel"), ("combined/fast/rankers.py", "Rankers.vsm")],
    },
    "BIM": {
        "week": 4, "family": "probabilistic", "slides": "Week 4, slides 5, 9",
        "idea": "Rank by probability of relevance, using only term presence or absence.",
        "formula": [r"w_t = \log\frac{(r+0.5)(N-df-R+r+0.5)}{(R-r+0.5)(df-r+0.5)}",
                    r"R=r=0:\quad w_t = \log\frac{N-df+0.5}{df+0.5}",
                    r"P = \frac{1}{1+e^{-s}}"],
        "source": [("week4/probabilistic.py", "BIM"), ("week4/probabilistic.py", "sigmoid"),
                   ("combined/fast/rankers.py", "Rankers.bim")],
    },
    "BM25": {
        "week": 4, "family": "probabilistic", "slides": "Week 4, slides 10, 19",
        "idea": "A smarter BIM: also uses how often a word appears and how long the document is.",
        "formula": [r"score(q,d) = \sum_{t \in q} IDF(t) \cdot \frac{tf\,(k_1+1)}{tf + k_1\left(1-b+b\frac{|d|}{avgdl}\right)}",
                    r"IDF(t) = \log\left(1+\frac{N-df+0.5}{df+0.5}\right)"],
        "source": [("week4/probabilistic.py", "BM25"), ("combined/fast/rankers.py", "Rankers.bm25")],
    },
    "Field BM25": {
        "week": 4, "family": "structured", "slides": "Week 4, slides 11–13, 19",
        "idea": "Treat a dialogue as fields, not flat text, and weight each field.",
        "formula": [r"\tilde{tf}(t,d) = \sum_{f} w_f \frac{tf_f}{1-b+b\frac{|f|}{avg_f}}",
                    r"score = \sum_t IDF(t)\frac{\tilde{tf}\,(k_1+1)}{\tilde{tf}+k_1}"],
        "source": [("week4/structured.py", "StructuredIndex.field_bm25"), ("combined/fast/rankers.py", "Rankers.field_bm25")],
    },
    "Overlapped lists": {
        "week": 4, "family": "structured", "slides": "Week 4, slides 14, 16",
        "idea": "All query words must occur together in one region (one turn). High precision.",
        "formula": [r"\text{regions}(q) = \bigcap_{t \in q} L_t", r"score(d) = |\{r \in \text{regions}(q) : r \subset d\}|"],
        "source": [("week4/structured.py", "StructuredIndex.overlapped"), ("combined/fast/rankers.py", "Rankers.overlapped")],
    },
    "Non-overlapped lists": {
        "week": 4, "family": "structured", "slides": "Week 4, slides 15–16",
        "idea": "Words may sit in different regions. Higher coverage, lower precision.",
        "formula": [r"\text{regions}(q) = \bigcup_{t \in q} L_t", r"coverage(d) = \frac{|\{t \in q : t \text{ occurs in } d\}|}{|q|}"],
        "source": [("week4/structured.py", "StructuredIndex.non_overlapped"), ("combined/fast/rankers.py", "Rankers.non_overlapped")],
    },
    "Proximal nodes": {
        "week": 4, "family": "structured", "slides": "Week 4, slide 17",
        "idea": "A word in a connected node still counts, at reduced credit λ.",
        "formula": [r"score(n) = \sum_{t} \max\left(s(t,n),\ \lambda \max_{m \in N(n)} s(t,m)\right)",
                    r"score(d) = \max_{n \in d} score(n)"],
        "source": [("week4/structured.py", "StructuredIndex.proximal"), ("combined/fast/rankers.py", "Rankers.proximal")],
    },
    "Fuzzy (AND)": {
        "week": 6, "family": "set-theoretic", "slides": "Week 6, slides 5–9, 16",
        "idea": "Relevance is a degree between 0 and 1, and typos still match.",
        "formula": [r"\mu_A(x) \in [0,1]", r"\mu_t(d) = \max_{v \in d} sim(t,v) \cdot trap(tf(v,d))",
                    r"\mu(A \text{ AND } B) = \min(\mu_A,\mu_B)", r"\mu(\text{NOT } A) = 1-\mu_A",
                    r"sim(a,b) = \frac{2M}{|a|+|b|}"],
        "source": [("setmodels/fuzzy.py", "Fuzzy"), ("setmodels/fuzzy.py", "trapezoid"), ("setmodels/similarity.py", "similarity")],
    },
    "Fuzzy (OR)": {
        "week": 6, "family": "set-theoretic", "slides": "Week 6, slides 5–9, 16",
        "idea": "The same model with every query word joined by OR, so any one word is enough.",
        "formula": [r"\mu(A \text{ OR } B) = \max(\mu_A,\mu_B)"],
        "source": [("setmodels/fuzzy.py", "Fuzzy.scores")],
    },
    "Extended Boolean": {
        "week": 6, "family": "set-theoretic", "slides": "Week 6, slides 10–12, 16",
        "idea": "Set logic decides what is kept, including BUT NOT. p-norm scores order the results (beyond the slides).",
        "formula": [r"(A \cup B) \setminus C", r"OR_p = \left(\frac{w_a^p+w_b^p}{2}\right)^{1/p}",
                    r"AND_p = 1-\left(\frac{(1-w_a)^p+(1-w_b)^p}{2}\right)^{1/p}"],
        "source": [("setmodels/extended_boolean.py", "ExtendedBoolean"), ("setmodels/extended_boolean.py", "and_p"),
                   ("setmodels/extended_boolean.py", "or_p")],
    },
    "GVSM": {
        "week": 7, "family": "algebraic", "slides": "Week 7, slides 6–9, 16",
        "idea": "Terms are no longer unrelated axes: correlated words share weight.",
        "formula": [r"\vec{d_j} = \sum_{i=1}^{n} w_{ij}\,\vec{g_i}",
                    r"sim(q,d) = \frac{q^T G\, d}{\sqrt{q^T G\, q}\,\sqrt{d^T G\, d}}"],
        "source": [("week7/gvsm.py", "exact_similarity"), ("week7/gvsm.py", "GVSM")],
    },
    "LSI": {
        "week": 7, "family": "algebraic", "slides": "Week 7, slides 10–13",
        "idea": "SVD finds hidden concepts, so a text can match without sharing a word.",
        "formula": [r"A \approx U_k \Sigma_k V_k^T", r"\hat{q} = U_k^T q", r"score(q,d) = \cos(\hat{q},\hat{d})"],
        "source": [("week7/lsi.py", "truncated_svd"), ("week7/lsi.py", "LSI"), ("week7/lsi.py", "term_document_matrix")],
    },
}

OTHERS = {
    "Filtering": {
        "week": 2, "slides": "Week 2, slides 18–19", "idea": "A fixed profile watches a stream and pushes what passes a threshold.",
        "formula": [r"\text{deliver } d \iff \cos(profile,d) \ge \theta"], "source": [("week2/filtering.py", "FilteringEngine")],
    },
    "Naive Bayes": {
        "week": 3, "slides": "Week 3, slide 16", "idea": "Supervised: assign a document to a known category.",
        "formula": [r"\hat{c} = \arg\max_c \left[\log P(c) + \sum_t tf(t,d)\log P(t|c)\right]",
                    r"P(t|c) = \frac{count(t,c)+1}{\sum_v count(v,c)+|V|}"],
        "source": [("week3/naive_bayes.py", "NaiveBayes"), ("combined/fast/classify.py", "naive_bayes_relations")],
    },
    "K-Means": {
        "week": 3, "slides": "Week 3, slide 17", "idea": "Unsupervised: discover groups with no labels.",
        "formula": [r"J = \sum_c \sum_{x \in c} \|x-\mu_c\|^2"], "source": [("week3/kmeans.py", "lloyd"), ("week3/kmeans.py", "plus_plus")],
    },
    "Flat browsing": {
        "week": 5, "slides": "Week 5, slides 7–8", "idea": "Read documents one after another.",
        "formula": [r"\text{Browsing Time} = \sum_i t_i"], "source": [("browsing/models.py", "browsing_time"), ("browsing/models.py", "flat_page")],
    },
    "Structured guided browsing": {
        "week": 5, "slides": "Week 5, slides 9–11", "idea": "A hierarchy with a visible path.",
        "formula": [r"StructuredGuidedBrowsing(D,H,UI) = N"], "source": [("browsing/models.py", "Hierarchy")],
    },
    "Hypertext": {
        "week": 5, "slides": "Week 5, slides 12, 19", "idea": "Non-linear jumps over links between documents.",
        "formula": [r"G = (V,E)"], "source": [("browsing/models.py", "Hypertext"), ("browsing/models.py", "shortest_path")],
    },
    "Porter stemmer": {
        "week": 1, "slides": "Week 1, slide 11", "idea": "Reduce words to a common stem.",
        "formula": [r"[C](VC)^m[V]"], "source": [("week1/porter.py", "measure"), ("week1/porter.py", "stem_word")],
    },
}
