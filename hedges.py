"""Hedge detector: a word list, nothing learned.

A sentence containing any cue is marked hedged. This misses subtle
uncertainty and flags some sentences that aren't really hedged
("in May", "potential energy") - a known limitation.
"""
from __future__ import annotations

import re

HEDGE_CUES = [
    "may", "might", "could", "suggest", "suggests", "suggested", "suggesting",
    "possibly", "possible", "likely", "unlikely", "appears to", "appear to",
    "seems to", "seem to", "potential", "potentially", "preliminary",
    "we hypothesize", "we hypothesized", "we speculate", "perhaps", "presumably",
    "probably", "it is unclear", "remains unclear", "remains to be",
]

# Longest cues first so "appears to" wins over a shorter overlap; \b keeps
# "may" from matching inside "mayor" or "dismay".
_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in sorted(HEDGE_CUES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def find_hedges(sentence: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of every hedge cue in the sentence."""
    spans = []
    for m in _PATTERN.finditer(sentence):
        # "May" capitalized mid-sentence is almost always the month.
        if m.group() == "May" and m.start() > 0:
            continue
        spans.append(m.span())
    return spans


def is_hedged(sentence: str) -> bool:
    return bool(find_hedges(sentence))


if __name__ == "__main__":
    tests = [
        "These results suggest that the drug may reduce mortality.",
        "Mortality was 12% lower in the treatment group.",
        "Patients were enrolled in May 2019.",
        "The mayor opened the clinic.",
        "This appears to be a potential mechanism.",
    ]
    for t in tests:
        print(f"{'HEDGED' if is_hedged(t) else 'firm  '}  {t}")
