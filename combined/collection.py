"""SODA as a compact on-disk collection, so all 1.49M dialogues fit in a laptop's memory.

Week 1's InvertedIndex keeps one Python dict per term, which needs far more RAM than this data allows.
This is the same inverted index stored as flat NumPy arrays that are memory-mapped from disk:

    postings of term t = doc[ptr[t] : ptr[t + 1]]  with counts  tf[ptr[t] : ptr[t + 1]]

Indexes:  narrative, opening (first 2 turns), body, closing (last 2 turns), plus two whole-document
          tracks: dialogue, and all (narrative + dialogue). "regions" indexes every turn on its own.
Not indexed: literal, head and tail hold the answer, and character names would only add noise.

Build:   python -m combined.collection 10000        (size in dialogues; "full" for everything)
"""
import json
import re
import sys
import threading
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data.fetch_soda import parquet_files
from week1.preprocess import remove_stopwords, stem, tokenize

ROOT = Path(__file__).resolve().parent.parent / "data" / "soda"
FIELDS = ["narrative", "opening", "body", "closing"]
TRACKS = {"all": FIELDS, "dialogue": FIELDS[1:]}
RELATIONS = ["xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]
SPLITS = ["train", "valid", "test"]
PLACEHOLDER = re.compile(r"Person[XYZ]('s)?")
COLUMNS = ["head", "relation", "tail", "narrative", "dialogue", "speakers", "PersonX", "PersonY", "PersonZ", "split"]
BATCH = 4000


def event_text(head):
    """The query for an event: 'PersonX helps PersonY's children' -> 'helps children'."""
    return " ".join(PLACEHOLDER.sub(" ", head).split())


def scan_metadata():
    """One row per dialogue across all files: event, relation, split. Duplicates are flagged."""
    path = ROOT / "meta.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    events, head_id, relation, split, seen, keep = {}, [], [], [], set(), []
    for f in parquet_files():
        for batch in pq.ParquetFile(f).iter_batches(1 << 16, columns=["head", "relation", "tail", "narrative", "split"]):
            cols = batch.to_pydict()
            for h, r, t, n, s in zip(cols["head"], cols["relation"], cols["tail"], cols["narrative"], cols["split"]):
                head_id.append(events.setdefault(h, len(events)))
                relation.append(RELATIONS.index(r))
                split.append(SPLITS.index(s))
                key = hash((h, r, t, n))
                keep.append(key not in seen)
                seen.add(key)
    meta = {"head_id": np.array(head_id, np.int32), "relation": np.array(relation, np.int8),
            "split": np.array(split, np.int8), "keep": np.array(keep), "events": np.array(list(events), dtype=object)}
    np.savez(path, **meta)
    return meta


def choose_rows(meta, size):
    """Whole events in a fixed random order, so smaller collections sit inside larger ones and every
    event keeps all its dialogues. Returns (rows in file order, events in the chosen order)."""
    order = np.random.default_rng(0).permutation(len(meta["events"]))
    rows = np.flatnonzero(meta["keep"])
    counts = np.bincount(meta["head_id"][rows], minlength=len(order))
    if size == "full":
        return rows, order
    taken = np.cumsum(counts[order])
    chosen = order[: int(np.searchsorted(taken, int(size))) + 1]
    mask = np.zeros(len(order), bool)
    mask[chosen] = True
    return rows[mask[meta["head_id"][rows]]], chosen


def strip_names(tokens, names):
    return [t for t in tokens if t not in names]


def analyze(row):
    """One dialogue -> {field: term counts} and the terms of each region (a turn or the narrative)."""
    names = set()
    for n in (row["PersonX"], row["PersonY"], row["PersonZ"]):
        names.update(tokenize(n or ""))
    clean = lambda text: stem(remove_stopwords(strip_names(tokenize(text), names)))
    turns = list(row["dialogue"])
    narrative = clean(row["narrative"] or "")
    per_turn = [clean(t) for t in turns]
    opening, body, closing = per_turn[:2], per_turn[2:-2], per_turn[2:][-2:]
    fields = {"narrative": Counter(narrative), "opening": Counter(t for p in opening for t in p),
              "body": Counter(t for p in body for t in p), "closing": Counter(t for p in closing for t in p)}
    regions = [(0, set(narrative))] if narrative else []
    for i, terms in enumerate(per_turn):
        field = 1 if i < 2 else (3 if i >= len(turns) - 2 and len(turns) > 2 else 2)
        if terms:
            regions.append((field, set(terms)))
    return fields, regions


def analyze_batch(job):
    """Worker: returns compact rows plus a local vocabulary, so only arrays cross processes."""
    first_doc, rows = job
    vocab, post = {}, {name: ([], [], []) for name in FIELDS + list(TRACKS)}
    reg_doc, reg_field, reg_rows, reg_terms = [], [], [], []
    for offset, row in enumerate(rows):
        doc = first_doc + offset
        fields, regions = analyze(row)
        for name, counts in fields.items():
            out = post[name]
            for t, tf in counts.items():
                out[0].append(vocab.setdefault(t, len(vocab)))
                out[1].append(doc)
                out[2].append(min(tf, 65535))
        for track, parts in TRACKS.items():
            merged = Counter()
            for p in parts:
                merged.update(fields[p])
            out = post[track]
            for t, tf in merged.items():
                out[0].append(vocab.setdefault(t, len(vocab)))
                out[1].append(doc)
                out[2].append(min(tf, 65535))
        for field, terms in regions:
            reg_doc.append(doc)
            reg_field.append(field)
            for t in terms:
                reg_rows.append(len(reg_doc) - 1)
                reg_terms.append(vocab.setdefault(t, len(vocab)))
    arrays = {n: (np.array(p[0], np.int32), np.array(p[1], np.int32), np.array(p[2], np.uint16))
              for n, p in post.items()}
    return (list(vocab), arrays, np.array(reg_doc, np.int32), np.array(reg_field, np.int8),
            np.array(reg_rows, np.int32), np.array(reg_terms, np.int32))


def read_rows(rows):
    """Yield (position, batch of row dicts) for the selected file rows, in order."""
    wanted = np.zeros(int(rows.max()) + 1, bool)
    wanted[rows] = True
    start, buffer, position = 0, [], 0
    for f in parquet_files():
        for batch in pq.ParquetFile(f).iter_batches(1 << 14, columns=COLUMNS):
            hits = np.flatnonzero(wanted[start: start + len(batch)]) if start < len(wanted) else []
            start += len(batch)
            if len(hits) == 0:
                continue
            cols = batch.take(hits).to_pydict()
            for i in range(len(hits)):
                buffer.append({c: cols[c][i] for c in COLUMNS})
                if len(buffer) == BATCH:
                    yield position, buffer
                    position += BATCH
                    buffer = []
    if buffer:
        yield position, buffer


def write_docstore(rows, handle, offsets):
    """Dialogue text as one JSON line per document, found through byte offsets."""
    for row in rows:
        offsets.append(handle.tell())
        handle.write(json.dumps({k: row[k] for k in ("head", "relation", "tail", "narrative", "dialogue", "speakers",
                                                     "split")}, ensure_ascii=False).encode("utf-8") + b"\n")


def tee_docstore(rows, out, slots):
    """Feed batches to the workers, holding back while too many are waiting, so RAM stays flat."""
    offsets = []
    with (out / "docs.jsonl").open("wb") as handle:
        for position, batch in read_rows(rows):
            write_docstore(batch, handle, offsets)
            slots.acquire()
            yield position, batch
        offsets.append(handle.tell())
    np.save(out / "doc_offsets.npy", np.array(offsets, np.int64))


def build(size, workers=5):
    meta = scan_metadata()
    rows, events = choose_rows(meta, size)
    out = ROOT / f"index_{size}"
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    N = len(rows)
    print(f"Building {N:,} dialogues into {out}", flush=True)

    # Pass 1: analyze in worker processes, map local term ids to one vocabulary, count document frequency.
    t0 = time.perf_counter()
    vocab, names = {}, FIELDS + list(TRACKS)
    df = {n: np.zeros(1 << 21, np.int64) for n in names + ["regions"]}
    region_docs, region_fields, n_regions, chunk = [], [], 0, 0
    slots = threading.Semaphore(workers * 2)
    with Pool(workers) as pool:
        for local, arrays, reg_doc, reg_field, reg_rows, reg_terms in pool.imap(analyze_batch, tee_docstore(rows, out, slots)):
            slots.release()
            mapping = np.fromiter((vocab.setdefault(t, len(vocab)) for t in local), np.int32, len(local))
            saved = {}
            for n, (terms, docs, tfs) in arrays.items():
                terms = mapping[terms]
                df[n] += np.bincount(terms, minlength=len(df[n]))
                saved[f"{n}_t"], saved[f"{n}_d"], saved[f"{n}_f"] = terms, docs, tfs
            terms = mapping[reg_terms]
            df["regions"] += np.bincount(terms, minlength=len(df["regions"]))
            saved["regions_t"], saved["regions_d"] = terms, reg_rows + n_regions
            n_regions += len(reg_doc)
            region_docs.append(reg_doc)
            region_fields.append(reg_field)
            np.savez(runs / f"{chunk:05d}.npz", **saved)
            chunk += 1
            if chunk % 25 == 0:
                print(f"  analyzed {min(chunk * BATCH, N):,} / {N:,}  {time.perf_counter() - t0:.0f}s", flush=True)
    V = len(vocab)
    print(f"Analyzed in {time.perf_counter() - t0:.0f}s: {V:,} terms, {n_regions:,} regions", flush=True)

    # Pass 2: every chunk is in document order, so appending keeps each postings list sorted.
    t1 = time.perf_counter()
    open_maps = {}
    for n in names + ["regions"]:
        ptr = np.zeros(V + 1, np.int64)
        ptr[1:] = np.cumsum(df[n][:V])
        np.save(out / f"{n}_ptr.npy", ptr)
        doc = np.lib.format.open_memmap(out / f"{n}_doc.npy", "w+", np.int32, (int(ptr[-1]),))
        tf = None if n == "regions" else np.lib.format.open_memmap(out / f"{n}_tf.npy", "w+", np.uint16, (int(ptr[-1]),))
        open_maps[n] = (ptr[:-1].copy(), doc, tf)
    lengths = {n: np.zeros(N, np.int32) for n in names}
    for i in range(chunk):
        with np.load(runs / f"{i:05d}.npz") as saved:
            for n in names + ["regions"]:
                next_slot, doc, tf = open_maps[n]
                terms, docs = saved[f"{n}_t"], saved[f"{n}_d"]
                order = np.argsort(terms, kind="stable")
                sorted_terms = terms[order]
                starts = np.r_[0, np.flatnonzero(np.diff(sorted_terms)) + 1]
                within = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
                slots = next_slot[sorted_terms] + within
                doc[slots] = docs[order]
                if tf is not None:
                    tfs = saved[f"{n}_f"]
                    tf[slots] = tfs[order]
                    lengths[n] += np.bincount(docs, weights=tfs, minlength=N).astype(np.int32)
                next_slot += np.bincount(terms, minlength=V)
        (runs / f"{i:05d}.npz").unlink()
    runs.rmdir()
    for n in names:
        np.save(out / f"{n}_len.npy", lengths[n])
    np.save(out / "region_doc.npy", np.concatenate(region_docs))
    np.save(out / "region_field.npy", np.concatenate(region_fields))
    np.save(out / "doc_head.npy", meta["head_id"][rows])
    np.save(out / "doc_relation.npy", meta["relation"][rows])
    np.save(out / "doc_split.npy", meta["split"][rows])
    np.save(out / "events_order.npy", events)
    (out / "vocab.json").write_text(json.dumps(list(vocab), ensure_ascii=False), encoding="utf-8")
    print(f"Merged in {time.perf_counter() - t1:.0f}s. Total {time.perf_counter() - t0:.0f}s", flush=True)


class Index:
    """One memory-mapped index, with the same questions as Week 1's InvertedIndex."""

    def __init__(self, folder, name, vocab):
        self.vocab = vocab
        self.ptr = np.load(folder / f"{name}_ptr.npy", mmap_mode="r")
        self.docs = np.load(folder / f"{name}_doc.npy", mmap_mode="r")
        self.tfs = np.load(folder / f"{name}_tf.npy", mmap_mode="r") if (folder / f"{name}_tf.npy").exists() else None
        self.doc_len = np.load(folder / f"{name}_len.npy").astype(np.float32) if (folder / f"{name}_len.npy").exists() else None
        self.N = len(self.doc_len) if self.doc_len is not None else 0
        self.avg_len = float(self.doc_len.sum() / max(np.count_nonzero(self.doc_len), 1)) if self.doc_len is not None else 0.0

    def df(self, term):
        t = self.vocab.get(term)
        return 0 if t is None else int(self.ptr[t + 1] - self.ptr[t])

    def postings(self, term):
        """(doc ids, term frequencies) as arrays; empty for an unknown term."""
        t = self.vocab.get(term)
        if t is None:
            return np.empty(0, np.int32), np.empty(0, np.uint16)
        a, b = int(self.ptr[t]), int(self.ptr[t + 1])
        return np.asarray(self.docs[a:b]), None if self.tfs is None else np.asarray(self.tfs[a:b])


class Collection:
    def __init__(self, size):
        self.size = size
        self.folder = ROOT / f"index_{size}"
        self.terms = json.loads((self.folder / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = {t: i for i, t in enumerate(self.terms)}
        self.indexes = {}
        self.doc_head = np.load(self.folder / "doc_head.npy")
        self.doc_relation = np.load(self.folder / "doc_relation.npy")
        self.doc_split = np.load(self.folder / "doc_split.npy")
        self.N = len(self.doc_head)
        self.region_doc = np.load(self.folder / "region_doc.npy", mmap_mode="r")
        self.region_field = np.load(self.folder / "region_field.npy", mmap_mode="r")
        self.offsets = np.load(self.folder / "doc_offsets.npy", mmap_mode="r")

    def index(self, name):
        if name not in self.indexes:
            self.indexes[name] = Index(self.folder, name, self.vocab)
        return self.indexes[name]

    def document(self, doc):
        """Full text of one dialogue, read from disk."""
        with (self.folder / "docs.jsonl").open("rb") as f:
            f.seek(int(self.offsets[doc]))
            return json.loads(f.readline())


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "10000")
