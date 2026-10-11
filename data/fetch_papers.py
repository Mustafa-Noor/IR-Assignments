"""Download ~40 real research papers from arXiv and split them into structured fields.

Source:  arXiv API (metadata + abstract) and arxiv.org/html/<id> (full-text sections).
Output:  data/corpus.json  -- one record per paper with
         id, arxiv_id, title, abstract, introduction, methodology, results,
         conclusion, references, category, year, authors

Run:     python data/fetch_papers.py
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).parent / "corpus.json"
API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# (category label, arXiv search query, how many papers to keep)
TOPICS = [
    ("Healthcare AI",        'cat:cs.LG AND abs:"medical" AND abs:"diagnosis"',      7),
    ("Cybersecurity",        'cat:cs.CR AND abs:"intrusion detection"',               7),
    ("Information Retrieval", 'cat:cs.IR AND abs:"retrieval" AND abs:"ranking"',      7),
    ("Databases",            'cat:cs.DB AND abs:"query optimization"',                7),
    ("Software Engineering", 'cat:cs.SE AND abs:"code review"',                       6),
    ("Computer Vision",      'cat:cs.CV AND abs:"image segmentation"',                6),
]

SECTION_WORDS = 300   # keep each section short so the corpus stays readable
HEADERS = {"User-Agent": "IR-course-assignment/1.0 (student project)"}

# Map a section heading to one of our structured fields.
FIELD_PATTERNS = [
    ("introduction", r"introduction|background|motivation"),
    ("methodology",  r"method|approach|framework|model|architecture|design|proposed|system"),
    ("results",      r"result|experiment|evaluation|performance|analysis|discussion"),
    ("conclusion",   r"conclusion|future work|summary|limitations"),
]


def search_arxiv(query, max_results=40):
    params = {"search_query": query, "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": max_results}
    r = requests.get(API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    for entry in root.findall("a:entry", ATOM):
        arxiv_id = entry.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1]
        yield {
            "arxiv_id": re.sub(r"v\d+$", "", arxiv_id),
            "title": " ".join(entry.find("a:title", ATOM).text.split()),
            "abstract": " ".join(entry.find("a:summary", ATOM).text.split()),
            "year": int(entry.find("a:published", ATOM).text[:4]),
            "authors": [a.find("a:name", ATOM).text for a in entry.findall("a:author", ATOM)][:5],
        }


def clip(text, n=SECTION_WORDS):
    words = text.split()
    return " ".join(words[:n])


def field_for(heading):
    h = heading.lower()
    for field, pat in FIELD_PATTERNS:
        if re.search(pat, h):
            return field
    return None


def fetch_sections(arxiv_id):
    """Return {field: text} parsed from arXiv's HTML rendering, or None if unavailable."""
    r = requests.get(f"https://arxiv.org/html/{arxiv_id}", headers=HEADERS, timeout=60)
    if r.status_code != 200 or "ltx_section" not in r.text:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    fields = {f: "" for f, _ in FIELD_PATTERNS}
    for sec in soup.select("section.ltx_section"):
        title_el = sec.find(class_="ltx_title_section")
        if not title_el:
            continue
        field = field_for(title_el.get_text(" ", strip=True))
        if not field or fields[field]:
            continue  # first matching section wins
        paras = [p.get_text(" ", strip=True) for p in sec.select("p.ltx_p")]
        fields[field] = clip(" ".join(paras))

    refs = [li.get_text(" ", strip=True) for li in soup.select("li.ltx_bibitem")][:8]
    fields["references"] = clip(" ".join(refs), 200)

    # Need at least introduction + methodology + results to be useful for structured retrieval.
    if not (fields["introduction"] and fields["methodology"] and fields["results"]):
        return None
    return fields


def main():
    corpus = []
    for category, query, keep in TOPICS:
        kept = 0
        print(f"\n== {category} ==")
        for meta in search_arxiv(query):
            if kept >= keep:
                break
            time.sleep(1)  # be polite to arXiv
            try:
                sections = fetch_sections(meta["arxiv_id"])
            except requests.RequestException as e:
                print(f"  skip {meta['arxiv_id']}: {e}")
                continue
            if not sections:
                print(f"  skip {meta['arxiv_id']}: no usable HTML sections")
                continue
            corpus.append({"id": f"D{len(corpus) + 1:02d}", **meta, "category": category, **sections})
            kept += 1
            print(f"  + {meta['arxiv_id']}  {meta['title'][:70]}")
        time.sleep(3)

    OUT.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(corpus)} papers to {OUT}")


if __name__ == "__main__":
    main()
