"""PDF -> clean text by page -> sentences.

    python pdf_to_sentences.py some_paper.pdf

Two-column layouts, tables and figure captions can still come out messy;
the filters below drop the worst of it.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

# A period after one of these is not the end of a sentence.
_ABBREVIATIONS = [
    "e.g", "i.e", "et al", "etc", "vs", "cf", "ca", "approx", "Fig", "Figs",
    "Eq", "Eqs", "Ref", "Refs", "Tab", "Sec", "No", "Dr", "Prof", "Mr", "Ms",
    "Mrs", "St", "Inc", "Ltd", "Co", "resp", "al",
]
_DOT = "․"  # stand-in for protected periods, swapped back after splitting
_ABBR_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.", re.IGNORECASE)
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s?[A-Z])")          # J. Smith, U.S.
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_REFERENCES_RE = re.compile(r"^\s*(\d+\.?\s+)?(references|bibliography|works cited|literature cited)\s*$", re.IGNORECASE)
# Section headings on a line of their own; dropped so they don't glue onto the next sentence.
_HEADING_RE = re.compile(
    r"^\s*(\d+(\.\d+)*\.?\s+)?(abstract|introduction|background|related work|methods?|"
    r"materials and methods|methodology|experiments?|results|results and discussion|discussion|"
    r"conclusions?|limitations|acknowledge?ments?)\s*:?\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join words hyphenated across lines
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
    text = text.replace(chr(0xFEFF), "")           # stray byte-order marks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    protected = _ABBR_RE.sub(lambda m: m.group(1) + _DOT, text)
    protected = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, protected)
    protected = re.sub(r"(\d)\.(\d)", r"\1" + _DOT + r"\2", protected)  # 3.14
    return [p.replace(_DOT, ".").strip() for p in _SPLIT_RE.split(protected) if p.strip()]


def looks_like_sentence(s: str) -> bool:
    """Drop table rows, page headers, author lists and other debris."""
    if not 30 <= len(s) <= 700:
        return False
    words = re.findall(r"[A-Za-z]{2,}", s)
    if len(words) < 5:
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / len(s) < 0.6:          # mostly numbers/symbols -> table or equation
        return False
    if not s[-1] in ".!?)\"'":          # sentences end like sentences
        return False
    return True


def guess_title(doc: fitz.Document, fallback: str) -> str:
    """PDF metadata title if it looks real, else the largest text on page 1."""
    meta = (doc.metadata or {}).get("title") or ""
    meta = clean_text(meta)
    # A real title has words; "pp164277 988..999" is a typesetter's file name.
    if len(re.findall(r"[A-Za-z]{3,}", meta)) >= 3 and not meta.lower().endswith((".pdf", ".doc", ".docx", ".tex", ".dvi")) \
            and "untitled" not in meta.lower() and "microsoft word" not in meta.lower():
        return meta
    try:
        spans = []
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        spans.append((round(span["size"], 1), span["text"].strip()))
        # Largest font size that has a reasonable amount of text.
        for size in sorted({s for s, _ in spans}, reverse=True):
            title = " ".join(t for s, t in spans if s == size)
            if 10 <= len(title) <= 300:
                return clean_text(title)
    except Exception:
        pass
    return fallback


def pdf_to_sentences(source, filename: str = "paper.pdf") -> tuple[str, list[dict]]:
    """source: a file path or the raw bytes of a PDF.

    Returns (title, sentences) where each sentence is
    {"page": 1-based page number, "position": order in the paper, "text": ...,
     "section": nearest heading above it ("" if none), "paragraph": paragraph number}.
    Stops at the References heading.
    """
    if isinstance(source, (str, Path)):
        doc = fitz.open(source)
        filename = Path(source).name
    else:
        doc = fitz.open(stream=source, filetype="pdf")

    title = guess_title(doc, fallback=Path(filename).stem)
    sentences: list[dict] = []
    seen: set[str] = set()
    done = False
    section, paragraph, pending = "", 0, ""

    def emit(text: str, page_number: int) -> None:
        nonlocal paragraph
        for s in split_sentences(text):
            if looks_like_sentence(s) and s not in seen:   # 'seen' removes repeated headers/footers
                seen.add(s)
                sentences.append({"page": page_number, "position": len(sentences), "text": s,
                                  "section": section, "paragraph": paragraph})
        paragraph += 1

    for page_number, page in enumerate(doc, start=1):
        # Blocks come back in reading order, which usually keeps columns intact.
        for block in page.get_text("blocks"):
            if block[6] != 0:           # 0 = text block, 1 = image
                continue
            kept = []
            for line in block[4].splitlines():
                if _REFERENCES_RE.match(line):   # everything after this is the reference list
                    done = True
                    break
                heading = _HEADING_RE.match(line)
                if heading:
                    section = heading.group(3).strip().title()
                else:
                    kept.append(line)
            text = clean_text(chr(10).join(kept))
            # A short block that doesn't end like a sentence is a title, heading,
            # page header or label, not prose. (Long ones are paragraphs cut by a
            # column or page break, so those stay.)
            ends = text.endswith((".", "!", "?", ".)", '."'))
            if len(text) >= 150 or ends:
                # One PDF block is one paragraph, unless it stops mid-sentence
                # (column break): then it is joined to the block that follows.
                pending = (pending + " " + text).strip()
                if ends:
                    emit(pending, page_number)
                    pending = ""
            if done:
                break
        if done:
            break
    if pending:   # a paragraph cut by a page break is carried onto the next page; this is the last one
        emit(pending, len(doc))
    doc.close()
    return title, sentences


def page_images(path) -> dict[int, list[bytes]]:
    """JPEG bytes of the pictures embedded on each page: {1-based page: [image, ...]}.

    shortcut: only embedded pictures are found; charts drawn as vector graphics are
    missed, and so are figures inside a scanned page (the whole page is one picture,
    which is skipped). Upgrade to rendering figure regions if too many are absent.
    """
    out: dict[int, list[bytes]] = {}
    seen: set[int] = set()            # a logo repeated on every page is shown once
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            # A scanned page is one page-sized picture with the text on top: that is not a figure.
            scans = {i["xref"] for i in page.get_image_info(xrefs=True) if page.get_text("words")
                     and fitz.Rect(i["bbox"]).get_area() > 0.9 * page.rect.get_area()}
            for xref in (img[0] for img in page.get_images(full=True)):
                if xref in seen or xref in scans:
                    continue
                seen.add(xref)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < 150 or pix.height < 150 or pix.colorspace is None:
                        continue      # icons, rules and masks are not figures
                    if pix.alpha:
                        pix = fitz.Pixmap(pix, 0)
                    if pix.colorspace.name != fitz.csRGB.name:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    while pix.width > 1400:
                        pix.shrink(1)  # halves the size; keeps the page light
                    out.setdefault(page_number, []).append(pix.tobytes("jpeg"))
                except Exception:
                    continue          # one unreadable image must not hide the paper
    return out


# A reference starts with "Surname, I." / "Surname AB" or a number ("12." / "[12]").
_REF_START_RE = re.compile(
    r"^(\[?\d{1,3}[.\]]\s+[A-Z]|[A-Z\u00c0-\u00dd][\w'\u2019\u00b4\u02dc\-]+(\s[A-Z][a-z]+)?,?\s([A-Z][.\-]?\s?){1,3}"
    r"(,|\.|\(|$|\s(?=[A-Z(\d]|et\b|and\b)))")
_REF_END_RE = re.compile(r"^(acknowledge?ments|conflict of interest|copyright ©|publisher.s note|supplementa\w+ (data|material))", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d\d[a-z]?\b")


def pdf_references(path) -> list[str]:
    """The papers this PDF cites, one string per entry of its reference list.

    shortcut: entries are split on author-like line starts, so odd reference styles
    can merge or split entries. Upgrade to a real citation parser if lists look wrong.
    """
    lines: list[str] = []
    found = False
    with fitz.open(path) as doc:
        # A line printed on three or more pages is a running header, not part of a reference.
        on_pages = Counter(line for page in doc for line in {clean_text(l) for l in page.get_text().splitlines()})
        for page in doc:
            height = page.rect.height
            for block in page.get_text("blocks"):
                if block[6] != 0:
                    continue
                # Running headers and page numbers live in the page margins.
                if found and (block[3] < 0.06 * height or block[1] > 0.94 * height):
                    continue
                for line in block[4].replace("­\n", "").splitlines():
                    if found:
                        lines.append(line)
                    elif _REFERENCES_RE.match(line):
                        found = True
                    elif re.match(r"1\.\s+[A-Z][\w'’\-]+,\s[A-Z]\.", line):
                        found = True          # no heading (Nature style): the list starts at "1. Surname, I."
                        lines.append(line)

    entries: list[str] = []
    current = ""
    for line in lines:
        line = clean_text(line)
        if not line or (len(line) > 12 and on_pages[line] >= 3):
            continue
        if _REF_END_RE.match(line):
            break
        # A new entry only begins once the one before it has its year, so an
        # author list that wraps onto a new line is not cut in two.
        if current and _REF_START_RE.match(line) and _YEAR_RE.search(current):
            entries.append(current)
            current = line
        else:
            current = f"{current[:-1]}{line}" if current.endswith("-") else f"{current} {line}".strip()
    entries.append(current)
    entries = [re.sub(r"^\[?\d{1,3}[.\]]\s+", "", e) for e in entries]   # the app numbers them itself
    return [e for e in entries if 40 <= len(e) <= 700 and _YEAR_RE.search(e)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python pdf_to_sentences.py paper.pdf")
    title, sents = pdf_to_sentences(sys.argv[1])
    print(f"Title: {title}\n{len(sents)} sentences\n")
    for s in sents:
        print(f"[p{s['page']:>2} para {s['paragraph']:>3} {s['section'][:12]:<12}] {s['text']}")
