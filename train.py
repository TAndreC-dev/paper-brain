"""Train the sentence-role classifier on PubMed 20k RCT.

    python train.py

Loads data/pubmed-rct/{train,dev,test}.txt, fits TF-IDF (words + word pairs)
+ logistic regression, tunes C on dev, reports accuracy once on test, prints
the confusion matrix, and saves models/classifier.joblib.

The app imports load_model() and label_sentences() from this file.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "pubmed-rct"
MODEL_PATH = ROOT / "models" / "classifier.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

LABELS = ["Background", "Methods", "Results", "Conclusions"]

# The dataset has five labels; OBJECTIVE is merged into Background.
LABEL_MAP = {
    "BACKGROUND": "Background",
    "OBJECTIVE": "Background",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "CONCLUSIONS": "Conclusions",
}

_NUMBER = re.compile(r"@|\d+(?:[.,]\d+)*")


def normalize(text: str) -> str:
    """Make real-paper text look like the training text.

    The dataset replaced every number with '@'. TF-IDF would throw '@' away,
    and numbers are a strong clue for Results, so both '@' and real digits
    become the word 'num'. Used for training AND for every sentence labeled
    later, so the model always sees the same kind of input.
    """
    return _NUMBER.sub(" num ", text.lower())


def load_split(path: Path) -> tuple[list[str], list[str]]:
    """Read one PubMed RCT file -> (sentences, labels).

    Format: a '###id' line starts each abstract, then one 'LABEL<tab>sentence'
    per line, with a blank line between abstracts.
    """
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("###"):
                continue
            label, _, sentence = line.partition("\t")
            if label in LABEL_MAP and sentence:
                texts.append(sentence)
                labels.append(LABEL_MAP[label])
    return texts, labels


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


# Kinds of section heading the context model knows; anything else is "other", no heading is "none".
SECTION_KINDS = ["none", "other", "abstract", "introduction", "methods", "results", "results+discussion",
                 "discussion", "conclusion"]


def section_kind(section: str | None) -> str:
    s = (section or "").lower()
    if not s:
        return "none"
    if "result" in s and "discussion" in s:
        return "results+discussion"
    for kind, words in (("abstract", ("abstract", "summary")), ("introduction", ("introduction", "background")),
                        ("methods", ("method", "experimental")), ("results", ("result",)),
                        ("discussion", ("discussion",)), ("conclusion", ("conclusion",))):
        if any(w in s for w in words):
            return kind
    return "other"


def context_features(probs, sections: list[str | None]):
    """What the text model thinks of each sentence, plus where the sentence sits in its paper.

    probs: the text model's label probabilities for every sentence of ONE paper, in reading order.
    Adds the sentence before, the sentence after, the average of the 7 around it, how far
    through the paper it is, and its kind of section.
    """
    import numpy as np
    n = len(probs)
    around = np.array([probs[max(0, i - 3):i + 4].mean(axis=0) for i in range(n)])
    kinds = np.zeros((n, len(SECTION_KINDS)))
    kinds[range(n), [SECTION_KINDS.index(section_kind(s)) for s in sections]] = 1
    return np.hstack([probs, np.vstack([probs[:1], probs[:-1]]), np.vstack([probs[1:], probs[-1:]]), around,
                      (np.arange(n) / max(n - 1, 1))[:, None], kinds])


assert section_kind("Materials and Methods") == "methods" and section_kind("Results And Discussion") == "results+discussion"

# The labelling rules in README.md, as yes/no clues the context model can weigh.
# Rule 4: someone else's finding ("Xu et al.", "(Smith, 2010)", "[12]", "previously reported") -> Background.
_CITATION = re.compile(
    r"\bet al\b|\([A-Z][\w'’\-]+[^()]{0,60}?\b(?:19|20)\d\d[a-z]?\s*[;,)]|\[\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\]"
    r"|\b(?:previous(?:ly)?|earlier|recent(?:ly)?)\s+(?:stud|report|work|show|describ|found)\w*"
    r"|\b(?:has|have) been (?:shown|reported|described|proposed|suggested)\b")
# Rule 3: figure legends and how-it-was-done notes -> Methods.
_LEGEND = re.compile(
    r"^\s*(?:\([A-L]\)|(?:Supplementa\w+ )?(?:Fig(?:ure)?|Table)\.?\s*S?\d+[A-Z]?\s*[.:|])"
    r"|\b(?:error bars|scale bars?|bars (?:represent|indicate|show)|asterisks? (?:indicate|denote)"
    r"|mean\s*±\s*(?:SD|SE|SEM)|values are means?)\b", re.IGNORECASE)
# Rule 1: a sentence that interprets -> Conclusions.
_INTERPRET = re.compile(
    r"\b(?:therefore|thus|hence|indicat\w*|suggest\w*|support\w*|impl(?:y|ies|ied|ication\w*)|conclu\w+"
    r"|consistent with|we propose|likely|may|might|could|possibl\w*|potential\w*|appears? to|seems? to)\b",
    re.IGNORECASE)


def rule_features(sentences: list[str]):
    """One row per sentence: [cites others, figure legend, interprets], each 0 or 1."""
    import numpy as np
    return np.array([[bool(p.search(s)) for p in (_CITATION, _LEGEND, _INTERPRET)] for s in sentences], dtype=float)


assert rule_features(["As shown by Xu et al., light resets the clock.", "(B) Error bars represent SD of three replicates.",
                      "These data suggest that XCT acts upstream.", "Plants were grown at 22 C."]).tolist() == \
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]


def label_sentences(model, sentences: list[str], sections: list[str | None] | None = None) -> list[tuple[str, float]]:
    """Return (label, confidence) for each sentence.

    Pass 'sections' (one per sentence, with 'sentences' being one whole paper in reading order)
    to use the context model too, when the saved model has one. Without it, text only.
    """
    if not sentences:
        return []
    text_model = model["text"] if isinstance(model, dict) else model
    probs, classes = text_model.predict_proba([normalize(s) for s in sentences]), text_model.classes_
    if isinstance(model, dict) and sections is not None:
        x = context_features(probs, sections)
        if model.get("rules"):   # models saved before rule features existed don't have the extra columns
            import numpy as np
            x = np.hstack([x, rule_features(sentences)])
        probs, classes = model["context"].predict_proba(x), model["context"].classes_
    best = probs.argmax(axis=1)
    return [(str(classes[i]), float(p[i])) for i, p in zip(best, probs)]


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    from download_data import download
    download()   # fetches the dataset the first time; afterwards only checks it is intact

    train_x, train_y = load_split(DATA_DIR / "train.txt")
    dev_x, dev_y = load_split(DATA_DIR / "dev.txt")
    test_x, test_y = load_split(DATA_DIR / "test.txt")
    print(f"Sentences: train {len(train_x):,} | dev {len(dev_x):,} | test {len(test_x):,}")

    train_n = [normalize(s) for s in train_x]
    dev_n = [normalize(s) for s in dev_x]
    test_n = [normalize(s) for s in test_x]

    # Tune the regularization strength C on dev only. Test is touched once, at the end.
    best = None
    for C in (1.0, 5.0, 20.0):
        t0 = time.time()
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ])
        model.fit(train_n, train_y)
        dev_acc = accuracy_score(dev_y, model.predict(dev_n))
        print(f"  C={C:<5} dev accuracy {dev_acc:.4f}  ({time.time() - t0:.0f}s)")
        if best is None or dev_acc > best[0]:
            best = (dev_acc, C, model)

    dev_acc, C, model = best
    pred = model.predict(test_n)
    test_acc = accuracy_score(test_y, pred)
    cm = confusion_matrix(test_y, pred, labels=LABELS)

    print(f"\nBest C = {C} (dev accuracy {dev_acc:.4f})")
    print(f"TEST ACCURACY: {test_acc:.4f}\n")
    print(classification_report(test_y, pred, labels=LABELS, digits=3))

    print("Confusion matrix (rows = true label, columns = predicted):")
    print(" " * 12 + "".join(f"{l[:11]:>12}" for l in LABELS))
    for label, row in zip(LABELS, cm):
        print(f"{label:<12}" + "".join(f"{n:>12,}" for n in row))

    # Most-confused pair: add the mistakes in both directions.
    pairs = []
    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            pairs.append((int(cm[i][j] + cm[j][i]), LABELS[i], LABELS[j]))
    n, a, b = max(pairs)
    print(f"\nMost-confused pair: {a} vs {b} ({n:,} test sentences mixed up)")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps({
        "train_sentences": len(train_x),
        "dev_sentences": len(dev_x),
        "test_sentences": len(test_x),
        "best_C": C,
        "dev_accuracy": round(dev_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "most_confused_pair": [a, b, n],
    }, indent=2))
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
