"""Week 7 — Latent Semantic Indexing (slides 10–13).

    A ≈ U_k Σ_k V_kᵀ       A = term-document matrix (rows = terms, columns = documents, slide 11)

Keeping only the top k singular values maps terms and documents into k concept dimensions, so two
texts can match through a shared concept even without a shared word.

    query fold-in:  q̂ = U_kᵀ q        document:  d̂ = U_kᵀ d = Σ_k · (row of V_k)
    score(q, d)  =  cos(q̂, d̂)
"""
import numpy as np
import scipy.sparse as sp

from week1.preprocess import preprocess

BLOCK = 100_000
SCORE_BLOCK = 10_000     # documents scored at a time, so a full collection stays well under 1 GB


def term_document_matrix(index, min_df=5):
    """Weighted A as a sparse matrix: w = (1 + log10 tf) · log10(N/df). Returns (A, term ids kept, idf)."""
    ptr = np.asarray(index.ptr)
    df = np.diff(ptr)
    n = int(np.count_nonzero(index.doc_len))
    kept = np.flatnonzero((df >= min_df) & (df < n))
    idf = np.log10(n / df[kept])
    indptr = np.r_[0, np.cumsum(df[kept])]
    rows = np.empty(indptr[-1], np.int32)
    data = np.empty(indptr[-1], np.float32)
    for i, t in enumerate(kept):
        a, b = int(ptr[t]), int(ptr[t + 1])
        rows[indptr[i]:indptr[i + 1]] = index.docs[a:b]
        data[indptr[i]:indptr[i + 1]] = (1 + np.log10(index.tfs[a:b].astype(np.float32))) * idf[i]
    # stored by term, so this is Aᵀ in compressed-row form: documents x terms by row of terms
    At = sp.csc_matrix((data, rows, indptr), shape=(index.N, len(kept)))
    return At.tocsr(), kept, idf


def blocks(At):
    for start in range(0, At.shape[0], BLOCK):
        yield At[start:start + BLOCK]


def multiply_AAt(At, X):
    """A Aᵀ X, one block of documents at a time so memory stays small."""
    out = np.zeros_like(X)
    for block in blocks(At):
        out += block.T @ (block @ X)
    return out


def truncated_svd(At, k, oversample=20, power=3, seed=0):
    """Top k of A's left singular vectors and values, by subspace iteration on A Aᵀ."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((At.shape[1], k + oversample), dtype=np.float32)
    for _ in range(power):
        X, _ = np.linalg.qr(multiply_AAt(At, X))
    small = X.T @ multiply_AAt(At, X)
    values, vectors = np.linalg.eigh(small)
    order = np.argsort(values)[::-1][:k]
    return X @ vectors[:, order], np.sqrt(np.maximum(values[order], 0))


class LSI:
    """Fitted once; document coordinates are saved to disk and memory-mapped."""

    def __init__(self, index, folder, track, k):
        self.index, self.k = index, k
        base = folder / f"lsi_{track}_{k}"
        if not (base.parent / f"{base.name}_S.npy").exists():     # saved last, so it marks a finished fit
            self.fit(base)
        self.kept = np.load(f"{base}_kept.npy")
        self.idf = np.load(f"{base}_idf.npy")
        self.U = np.load(f"{base}_U.npy")
        self.singular = np.load(f"{base}_S.npy")
        self.docs = np.load(f"{base}_docs.npy", mmap_mode="r")
        self.position = {int(t): i for i, t in enumerate(self.kept)}

    def fit(self, base):
        At, kept, idf = term_document_matrix(self.index)
        U, S = truncated_svd(At, self.k)
        docs = np.lib.format.open_memmap(f"{base}_docs.npy", "w+", np.float32, (At.shape[0], self.k))
        start = 0
        for block in blocks(At):
            d = block @ U
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            docs[start:start + len(d)] = d
            start += len(d)
        docs.flush()
        np.save(f"{base}_kept.npy", kept)
        np.save(f"{base}_idf.npy", idf)
        np.save(f"{base}_U.npy", U)
        np.save(f"{base}_S.npy", S)

    def fold_in(self, query):
        """q̂ = U_kᵀ q, normalized. Zeros if no query word is in the matrix."""
        q = np.zeros(len(self.kept), np.float32)
        for term in preprocess(query):
            t = self.index.vocab.get(term)
            i = None if t is None else self.position.get(t)
            if i is not None:
                q[i] += self.idf[i]
        z = self.U.T @ q
        norm = np.linalg.norm(z)
        return z / norm if norm else z

    def rank_many(self, queries, depth=1000):
        """Rank a batch of queries in one pass over the document coordinates."""
        Z = np.stack([self.fold_in(q) for q in queries])
        best_scores = np.full((len(queries), depth), -np.inf, np.float32)
        best_docs = np.full((len(queries), depth), -1, np.int64)
        for start in range(0, len(self.docs), SCORE_BLOCK):
            count = min(SCORE_BLOCK, len(self.docs) - start)
            block = np.fromfile(self.docs.filename, np.float32, count * self.k,            # read, not mapped into RAM
                                offset=self.docs.offset + start * self.k * 4).reshape(count, self.k)
            scores = block @ Z.T
            ids = start + np.arange(scores.shape[0])
            for q in range(len(queries)):
                merged_s = np.r_[best_scores[q], scores[:, q]]
                merged_d = np.r_[best_docs[q], ids]
                pick = np.argpartition(-merged_s, depth - 1)[:depth]
                best_scores[q], best_docs[q] = merged_s[pick], merged_d[pick]
        out = []
        for q in range(len(queries)):
            order = np.lexsort((best_docs[q], -best_scores[q]))
            ids = best_docs[q][order]
            out.append(ids[(best_scores[q][order] > 0) & (ids >= 0)] if np.any(Z[q]) else np.empty(0, np.int64))
        return out

    def rank(self, query, depth=1000):
        return self.rank_many([query], depth)[0]
