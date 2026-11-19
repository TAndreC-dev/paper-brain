"""Take the labels from a colour-highlighted copy of a paper that is already in the library.

    python import_highlights.py "highlighted.pdf" PAPER_ID            show what would be saved
    python import_highlights.py "highlighted.pdf" PAPER_ID --apply    save as your own labels

Colours (the nearest one wins): green = Results, brown = Conclusions, blue = Methods,
red and purple = Background (the app no longer keeps "key" background apart). Unhighlighted sentences are left exactly as they are.
Sentences you already labelled by hand in Read are not touched.

shortcut: reads highlights drawn as coloured boxes behind the text (a "highlighted edition"
made from a web page). Highlights added with a PDF reader's highlighter are annotations and
are not read; add page.annots() here if you start using those.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter

import pymupdf as fitz

import db

COLORS = {  # pale box colour -> label
    (0.86, 0.99, 0.91): "Results", (0.86, 0.92, 1.0): "Methods", (0.94, 0.89, 0.84): "Conclusions",
    (1.0, 0.89, 0.89): "Background", (0.93, 0.91, 1.0): "Background",
    # a see-through marker drawn over the original PDF's own pages
    (0.18, 0.85, 0.3): "Results", (0.2, 0.62, 0.98): "Methods", (0.76, 0.45, 0.16): "Conclusions",
    (0.96, 0.25, 0.25): "Background",
}


def color_label(fill) -> str | None:
    best = min(COLORS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, fill)))
    return COLORS[best] if sum((a - b) ** 2 for a, b in zip(best, fill)) < 0.003 else None


assert color_label((0.86, 0.99, 0.91)) == "Results" and color_label((1, 1, 1)) is None


def squash(text: str) -> str:
    # Letters only: reference numbers and ligatures (fi, fl) differ between two copies of a paper.
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKC", text).lower())


def highlighted_text(path: str) -> tuple[str, list[str | None]]:
    """The PDF's letters as one string, and the label of the box behind each letter."""
    letters, labels = [], []
    with fitz.open(path) as doc:
        for page in doc:
            boxes = [(d["rect"], color_label(d["fill"])) for d in page.get_drawings() if d.get("fill")]
            boxes = [(r, l) for r, l in boxes if l and r.width > 3 and r.height > 3]
            for x0, y0, x1, y1, word, *_ in page.get_text("words"):
                centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
                label = next((l for r, l in boxes if centre in r), None)
                s = squash(word)
                letters.append(s)
                labels.extend([label] * len(s))
    return "".join(letters), labels


def main() -> None:
    path, paper_id = sys.argv[1], int(sys.argv[2])
    text, labels = highlighted_text(path)
    picked, missing, plain = {}, 0, 0
    for s in db.get_sentences(paper_id):
        key = squash(s["text"])
        at = text.find(key)
        if at < 0 and len(key) > 60:                  # a word differs somewhere: match on the start,
            at = text.find(key[:40])                  # or failing that on the middle
            if at < 0:
                at = text.find(key[len(key) // 2 - 20:len(key) // 2 + 20]) - (len(key) // 2 - 20)
                at = -1 if at < -1 else max(at, 0)
        if at < 0 or len(key) < 15:
            missing += 1
            continue
        label, n = Counter(labels[at:at + len(key)]).most_common(1)[0]
        if label is None or n < 0.6 * len(key):       # not highlighted, or split between two colours
            plain += 1
        elif not s["corrected"]:
            picked[s["id"]] = (label, s["label"], s["text"])
    print(f"{len(picked)} highlighted sentences matched, {plain} unhighlighted, {missing} not found in the PDF")
    print("  by colour:", dict(Counter(l for l, _, _ in picked.values())))
    print("  current label was already the same for", sum(l == old for l, old, _ in picked.values()))
    for label in dict.fromkeys(l for l, _, _ in picked.values()):
        print(f"  e.g. {label}: {next(t for l, _, t in picked.values() if l == label)[:100]}")
    if "--apply" in sys.argv:
        for label in set(l for l, _, _ in picked.values()):
            db.set_labels([i for i, (l, _, _) in picked.items() if l == label], label)
        print("Saved as your labels.", "Backup problem: " + db.backup_error if db.backup_error else "")


if __name__ == "__main__":
    main()
