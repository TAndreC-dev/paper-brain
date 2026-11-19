"""Find and download a free, legal copy of a cited paper.

    python download.py        # self-check of the DOI reader (no internet needed)

The only part of Paper Brain that uses the internet. It asks OpenAlex
(openalex.org, free, no account) where an open-access PDF of a DOI lives,
then tries each place in turn. Paywalled papers are never fetched.
"""
from __future__ import annotations

import re

import requests

_HEADERS = {"User-Agent": "PaperBrain/0.1 (local research tool)"}
_MAX_BYTES = 60 * 1024 * 1024
_DOI_RE = re.compile(r"\b10\.\d{4,9}/")


def find_doi(reference: str) -> str | None:
    """The DOI inside one reference-list entry, or None.

    PDFs break long DOIs across lines ("10.1104/pp.110. 164277"), so a piece ending
    in . / - is glued to the next one unless that starts a new sentence (a capital).
    """
    m = _DOI_RE.search(reference)
    if not m:
        return None
    pieces = reference[m.start():].split()
    doi = pieces[0]
    for nxt in pieces[1:]:
        if doi[-1] in "./-" and not nxt[0].isupper() and not nxt.startswith(("[", "(", "http")):
            doi += nxt
        else:
            break
    return doi.rstrip(".,;)") or None


def open_access_pdf(doi: str) -> bytes | None:
    """The PDF of an open-access copy of this DOI, or None if there is none we can fetch."""
    r = requests.get(f"https://api.openalex.org/works/doi:{doi}",
                     params={"select": "best_oa_location,locations"}, headers=_HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    work = r.json()
    places = [work.get("best_oa_location")] + (work.get("locations") or [])
    urls = list(dict.fromkeys(p["pdf_url"] for p in places if p and p.get("is_oa") and p.get("pdf_url")))
    for url in urls:
        try:
            with requests.get(url, headers=_HEADERS, timeout=60, stream=True) as pdf:
                if pdf.status_code != 200:
                    continue
                data = pdf.raw.read(_MAX_BYTES + 1, decode_content=True)
        except requests.RequestException:
            continue
        # Publishers often answer with a web page instead of the file; only a real PDF counts.
        if data.startswith(b"%PDF") and len(data) <= _MAX_BYTES:
            return data
    return None


if __name__ == "__main__":
    assert find_doi("Plant Physiol. 155, 988–999. doi: 10.1104/pp.110. 164277") == "10.1104/pp.110.164277"
    assert find_doi("Science. 301:653–657. https://doi.org/10.1126/ science.1086391.") == "10.1126/science.1086391"
    assert find_doi("https://doi. org/10.1016/j.molcel.2022.05.021 Zhang S, Liu Y.") == "10.1016/j.molcel.2022.05.021"
    assert find_doi("Methods Mol Biol. 2398:15–32. https://doi.org/10.1007/978-1- 0716-1912-4_2.") == "10.1007/978-1-0716-1912-4_2"
    assert find_doi("Plant Biotechnol. J. doi: 10.1111/pbi.12782 [Epub ahead of print].") == "10.1111/pbi.12782"
    assert find_doi("Chae HS, Kieber JJ (2005) Eto Brute? Trends Plant Sci 10: 291–296") is None
    print("DOI OK")
