"""Retrain the sentence classifier with your own labels added to PubMed RCT.

    python retrain.py            measure only
    python retrain.py --apply    measure, and if the retrained model is better: save it and relabel brain.db

How it is scored: every paper you labelled by hand (in Read, or with import_highlights.py) is
held out in turn. A model that has not seen that paper's labels predicts them, and the
predictions are compared with yours. Four things are compared: the original PubMed-only
model, the retrained text model, the retrained text model plus context (section, place
in the paper, neighbouring sentences), and context plus rule clues (cites others, figure
legend, interprets: see rule_features in train.py). The best one is kept.

Measuring only reads the database. Set PAPER_BRAIN_DB to measure on a copy.

Training sentences: PubMed RCT, your hand labels (counted heavily), and optionally
heading-made labels (a sentence under "Materials and methods" is Methods, ...) from papers
you have not labelled by hand: set OWN_WEIGHT above 0, or pass --own-weight N to try it.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from collections import Counter

import joblib
import numpy as np

import db
from train import DATA_DIR, LABELS, MODEL_PATH, context_features, label_sentences, load_split, normalize, rule_features

ORIGINAL_PATH = MODEL_PATH.with_name("classifier.pubmed.joblib")   # the PubMed-only model, kept to compare and revert
# One heading-labelled sentence counts as this many PubMed sentences. 0 = do not use them: with
# 1,218 hand labels they lowered held-out accuracy from 67.3% to 62.6% (20 was the old value).
OWN_WEIGHT = 0
CORRECTION_WEIGHT = 100  # and one you labelled by hand as this many
if "--own-weight" in sys.argv:
    OWN_WEIGHT = float(sys.argv[sys.argv.index("--own-weight") + 1])


def section_label(section: str | None) -> str | None:
    s = (section or "").lower()
    if "discussion" in s:
        return None
    for word, label in (("method", "Methods"), ("experimental", "Methods"), ("result", "Results"),
                        ("introduction", "Background"), ("background", "Background"), ("conclusion", "Conclusions")):
        if word in s:
            return label
    return None


assert section_label("Materials and Methods") == "Methods" and section_label("Results and Discussion") is None


def fit(texts: list[str], labels: list[str], weights: list[float], C: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                      ("clf", LogisticRegression(C=C, max_iter=1000))])   # same recipe as train.py
    return model.fit([normalize(t) for t in texts], labels, clf__sample_weight=weights)


def accuracy(model, texts: list[str], labels: list[str]) -> float:
    return sum(p == y for (p, _), y in zip(label_sentences(model, texts), labels)) / max(len(labels), 1)


def main() -> None:
    from sklearn.linear_model import LogisticRegression

    if not ORIGINAL_PATH.exists():
        shutil.copy2(MODEL_PATH, ORIGINAL_PATH)
    original = joblib.load(ORIGINAL_PATH)
    C = original.named_steps["clf"].C

    conn = sqlite3.connect(db.DB_PATH.as_uri() + "?mode=ro", uri=True)   # measuring never writes
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT id, paper_id, text, label, section, corrected FROM sentences "
                                          "ORDER BY paper_id, position")]
    conn.close()

    def hand(r: dict) -> str | None:
        return r["label"] if r["corrected"] and r["label"] in LABELS else None

    paper_rows: dict[int, list[dict]] = {}
    for r in rows:
        paper_rows.setdefault(r["paper_id"], []).append(r)
    hand_papers = [p for p, rs in paper_rows.items() if sum(bool(hand(r)) for r in rs) >= 30]
    # Heading-made labels, only from papers you have not labelled by hand and whose headings were
    # found properly (a paper with everything under "Introduction" would teach wrong labels).
    headings: list[tuple[str, str]] = []
    for p, rs in paper_rows.items():
        pairs = [(r["text"], section_label(r["section"])) for r in rs if not r["corrected"]]
        pairs = [(t, l) for t, l in pairs if l]
        if OWN_WEIGHT and p not in hand_papers and len({l for _, l in pairs}) >= 2:
            headings += pairs
    print(f"Hand labels: {sum(bool(hand(r)) for r in rows)} in {len(hand_papers)} papers "
          f"{dict(Counter(hand(r) for r in rows if hand(r)))} | heading labels used: {len(headings)}")
    if len(hand_papers) < 2:
        sys.exit("Need at least 2 hand-labelled papers to test on.")

    pub_x, pub_y = load_split(DATA_DIR / "train.txt")
    test_x, test_y = load_split(DATA_DIR / "test.txt")

    def train(without: int | None = None):
        mine = [(r["text"], hand(r)) for r in rows if hand(r) and r["paper_id"] != without]
        return fit(pub_x + [t for t, _ in headings] + [t for t, _ in mine],
                   pub_y + [l for _, l in headings] + [l for _, l in mine],
                   [1.0] * len(pub_x) + [OWN_WEIGHT] * len(headings) + [CORRECTION_WEIGHT] * len(mine), C)

    # Each hand-labelled paper is predicted by a text model that never saw its labels.
    feats = {}
    for p in hand_papers:
        rs, blind = paper_rows[p], train(without=p)
        probs = blind.predict_proba([normalize(r["text"]) for r in rs])
        keep = [i for i, r in enumerate(rs) if hand(r)]
        y = [hand(rs[i]) for i in keep]
        ctx = context_features(probs, [r["section"] for r in rs])[keep]
        feats[p] = (ctx, np.hstack([ctx, rule_features([rs[i]["text"] for i in keep])]), y,
                    [str(blind.classes_[j]) for j in probs[keep].argmax(axis=1)],
                    [l for l, _ in label_sentences(original, [rs[i]["text"] for i in keep])])
        print(f"  trained without paper #{p}", flush=True)

    def stacker(papers: list[int], which: int):   # which: 0 = context, 1 = context + rules
        return LogisticRegression(max_iter=2000).fit(np.vstack([feats[q][which] for q in papers]),
                                                     [l for q in papers for l in feats[q][2]])

    # The context models learn from the other papers' predictions-plus-context, then are tested on this one.
    # shortcut: those other papers' predictions came from text models that did see the test paper,
    # a small leak in context's favour; do a fully nested test if context only wins narrowly.
    old_hits = text_hits = ctx_hits = rule_hits = n = 0
    for p in hand_papers:
        others = [q for q in hand_papers if q != p]
        ctx, ctx_rules, y, text_pred, old_pred = feats[p]
        o, t, c, r = (sum(a == b for a, b in zip(pred, y)) for pred in
                      (old_pred, text_pred, stacker(others, 0).predict(ctx), stacker(others, 1).predict(ctx_rules)))
        print(f"  paper #{p}: {len(y):>3} hand labels | old {o / len(y):.3f}  text {t / len(y):.3f}  "
              f"with context {c / len(y):.3f}  context+rules {r / len(y):.3f}")
        old_hits, text_hits, ctx_hits, rule_hits, n = old_hits + o, text_hits + t, ctx_hits + c, rule_hits + r, n + len(y)
    print(f"\nHELD-OUT HAND LABELS ({n}): old {old_hits / n:.3f} | retrained text {text_hits / n:.3f} | "
          f"with context {ctx_hits / n:.3f} | context+rules {rule_hits / n:.3f}", flush=True)

    best_hits, kind = max((text_hits, "text"), (ctx_hits, "context"), (rule_hits, "context+rules"))
    better = best_hits > old_hits + 0.01 * n
    print("VERDICT:", f"retrained model is better ({kind})" if better else "no clear improvement: keeping the old model")
    if not better or "--apply" not in sys.argv:
        return

    final = train()
    print(f"PUBMED TEST: old {accuracy(original, test_x, test_y):.3f} -> retrained {accuracy(final, test_x, test_y):.3f}")
    model = final
    if kind != "text":
        rules = kind == "context+rules"
        model = {"text": final, "context": stacker(hand_papers, int(rules)), "rules": rules}
    joblib.dump(model, MODEL_PATH, compress=3)
    # Relabel the library with the new model; your own labels are left alone.
    backup = db.DB_PATH.with_name("brain.backup.db")
    if not backup.exists():   # keep the copy from before the first --apply, so "go back" is a full undo
        src, dst = sqlite3.connect(db.DB_PATH), sqlite3.connect(backup)
        src.backup(dst); dst.close(); src.close()
    todo, labeled = [], []
    for rs in paper_rows.values():   # one paper at a time, in reading order: the context model needs that
        for r, pair in zip(rs, label_sentences(model, [r["text"] for r in rs], [r["section"] for r in rs])):
            if not r["corrected"]:
                todo.append(r)
                labeled.append(pair)
    conn = db.connect()
    with conn:
        conn.executemany("UPDATE sentences SET label = ?, confidence = ? WHERE id = ? AND corrected = 0",
                         [(label, conf, r["id"]) for r, (label, conf) in zip(todo, labeled)])
    conn.close()
    db.backup_db()
    changed = sum(r["label"] != label for r, (label, _) in zip(todo, labeled))
    print(f"Saved {MODEL_PATH.name}; relabelled {len(todo)} sentences ({changed} changed). "
          f"Old model: {ORIGINAL_PATH.name}; database before the first relabelling: brain.backup.db")


if __name__ == "__main__":
    main()
