"""Download the PubMed 20k RCT training data into data/pubmed-rct/.   python download_data.py

train.py runs this by itself when the files are missing. About 38 MB, from
https://github.com/Franck-Dernoncourt/pubmed-rct (Dernoncourt & Lee, 2017).
Files already downloaded and intact are skipped.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data" / "pubmed-rct"
BASE = ("https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/"
        "PubMed_20k_RCT_numbers_replaced_with_at_sign/")
# SHA-256 of each file, so a broken or changed download is caught before training on it.
FILES = {
    "train.txt": "cc06d6f4252a98dfb3112b8edd44d485b18671b0838e616200e0b878f5d6540a",
    "dev.txt": "61e0b7376844ab2cacc78dbd64338a47c8c7401bd2077134024500574ffdcab7",
    "test.txt": "216ba4f84a07f033de27f1b7d3ef75281f19e03cdd162aa2acf33d973f7c3c67",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = DATA_DIR / name
        if path.exists() and sha256(path) == expected:
            print(f"  {name}: already here")
            continue
        print(f"  {name}: downloading...", flush=True)
        part = path.with_suffix(".part")   # written first, so an interrupted download never looks finished
        with requests.get(BASE + name, timeout=60, stream=True) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if sha256(part) != expected:
            part.unlink()
            raise SystemExit(f"{name} did not download correctly (checksum mismatch). Try again, "
                             f"or download it by hand from {BASE + name}")
        os.replace(part, path)
    print(f"Data ready in {DATA_DIR}")


if __name__ == "__main__":
    download()
