"""Download the full SODA dataset (Kim et al., EMNLP 2023; CC BY 4.0) as Parquet.

Source:  https://huggingface.co/datasets/allenai/soda  (1,486,896 dialogues, 884 MB)
Output:  data/soda/raw/<split>-<n>.parquet   (git-ignored; rebuild with this script)

Run:     python -m data.fetch_soda
"""
from pathlib import Path

import requests

RAW = Path(__file__).parent / "soda" / "raw"
BASE = "https://huggingface.co/datasets/allenai/soda/resolve/refs%2Fconvert%2Fparquet/default"
# (split, part, size in bytes) as listed by the Hugging Face dataset API
FILES = [("train", 0, 263013619), ("train", 1, 267625553), ("train", 2, 178818919),
         ("validation", 0, 86705363), ("test", 0, 88111284)]


def parquet_files():
    """Local Parquet paths in a fixed order, so document ids are reproducible."""
    return [RAW / f"{split}-{part}.parquet" for split, part, _ in FILES]


def download(split, part, size):
    target = RAW / f"{split}-{part}.parquet"
    if target.exists() and target.stat().st_size == size:
        print(f"  have {target.name}")
        return
    partial = target.with_suffix(".part")
    for attempt in range(20):                     # a dropped connection resumes where it stopped
        have = partial.stat().st_size if partial.exists() else 0
        if have >= size:
            break
        try:
            with requests.get(f"{BASE}/{split}/{part:04d}.parquet", stream=True, timeout=60,
                              headers={"Range": f"bytes={have}-"}) as r:
                r.raise_for_status()
                with partial.open("ab") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
        except requests.RequestException as e:
            print(f"  retrying {target.name}: {e}", flush=True)
    if partial.stat().st_size != size:
        raise IOError(f"{target.name}: expected {size} bytes, got {partial.stat().st_size}")
    partial.replace(target)
    print(f"  got  {target.name}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for split, part, size in FILES:
        download(split, part, size)
    print(f"Saved {len(FILES)} files to {RAW}")


if __name__ == "__main__":
    main()
