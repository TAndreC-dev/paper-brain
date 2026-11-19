# Paper Brain

A local, searchable second brain for research papers. Every sentence of every
paper you add is labeled by what it does (Background, Methods, Results,
Conclusions), checked for hedging ("may", "suggests"), saved, and searchable later.

No API and no pretrained LLM. A sentence classifier trained here labels each
sentence, and search only ever returns real sentences from the paper with the
title and page attached. It cannot hallucinate an answer because it never writes one.

## Run it

Needs Python 3.11.

```bash
py -3.11 -m venv .venv          # Windows;  python3.11 -m venv .venv on Mac/Linux
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python train.py                 # once, about 4 minutes
streamlit run app.py
```

The first `train.py` downloads its training data (PubMed 20k RCT, about 38 MB, from
<https://github.com/Franck-Dernoncourt/pubmed-rct>) into `data/pubmed-rct/` and checks
it arrived intact. To fetch it on its own: `python download_data.py`.

After setup, start the app with a double-click on `Open Paper Brain.bat` (Windows) or
`sh run.sh` (Mac/Linux).

## Your data

Everything you add stays in this folder: `brain.db` (sentences, labels, your corrections)
and `papers/` (the PDFs). To back up, copy those two. Automatic backups are off; to turn
them on, set `BACKUP_DIR` at the top of `db.py` to a folder.

## The three tabs

| Tab | What you do | What happens |
|---|---|---|
| Add | Upload a PDF | Text is split into sentences, each labeled and checked for hedging, then saved |
| Ask | Type a search, pick filters | Top matching sentences from all saved papers, each with paper title and page |
| Read | Pick a saved paper | The paper sentence by sentence, color-coded by label, hedges underlined |

Ask filters: label, firmness (hide hedged / only hedged), and one paper or all.

## How it works

| File | Job |
|---|---|
| `train.py` | Loads PubMed 20k RCT, merges OBJECTIVE into Background, trains TF-IDF (words + word pairs) + logistic regression, tunes `C` on dev, reports test accuracy and the confusion matrix once, saves `models/classifier.joblib` |
| `download_data.py` | Downloads and checksums the PubMed 20k RCT files (`train.py` calls it) |
| `hedges.py` | Hedge word list and `is_hedged(sentence)` |
| `pdf_to_sentences.py` | PyMuPDF text by page, cleaned, split into sentences; stops at References |
| `db.py` | SQLite tables `papers` and `sentences`, `add_paper()`, `search()` |
| `app.py` | Streamlit app |

**One detail worth explaining.** The training data has every number replaced by
`@`, but real papers have real numbers. `normalize()` in `train.py` turns both
`@` and real digits into the word `num`, for training and for every sentence
labeled later, so the model always sees the same kind of input. Numbers are a
strong clue that a sentence is a Result.

**Search.** SQLite's built-in FTS5 index ranks with BM25. Filters are plain
`WHERE` clauses. Porter stemming lets "reduce" match "reduced".

## Results

Accuracy was measured on the PubMed RCT test set (abstracts of randomized controlled trials) and may differ on other kinds of papers.

| What | Value |
|---|---|
| Training sentences used | 180,040 |
| Dev / test sentences | 30,212 / 30,135 |
| Best `C` (chosen on dev) | 5.0 (dev accuracy 87.40%) |
| Test accuracy (baseline) | **86.85%** |
| Test accuracy (PyTorch, if built) | not built yet |
| Most-confused label pair | Methods vs Results (1,149 test sentences) |
| Papers loaded | fill in |
| Sentences stored | fill in |
| One thing it helped you find | fill in |

Confusion matrix on test (rows = true label, columns = predicted):

| | Background | Methods | Results | Conclusions |
|---|---:|---:|---:|---:|
| Background | 5,102 | 301 | 56 | 495 |
| Methods | 234 | 9,126 | 459 | 78 |
| Results | 53 | 690 | 8,597 | 373 |
| Conclusions | 556 | 116 | 553 | 3,346 |

Methods vs Results is the biggest mix-up by count (both are full of numbers
and measurements). Conclusions is the weakest label: 73% recall, with its
mistakes split between Background (556) and Results (553). Background vs
Conclusions totals 1,051, a close second.

## Known limitations

- Trained on medical abstracts, so labels are less accurate on full paper
  bodies and on non-medical papers.
- Two-column PDFs, tables and figure captions can come out messy. Scanned
  PDFs (images only) give no text at all.
- The hedge word list misses subtle uncertainty and flags some sentences that
  aren't really hedged.
- Search matches words, not meaning: "heart attack" won't find "myocardial
  infarction".
- The classifier sees one sentence at a time, with no idea where in the paper
  it sits.

## Stretch ideas

- A small PyTorch model (embedding, average or 1D convolution, linear) vs the baseline
- BM25 written by hand in Python instead of FTS5
- Check the hedge list against the BioScope / CoNLL-2010 hedge data

## Labelling rules

Two highlightings of the same paper agreed on 86% of sentences; these rules settle the rest.
The Read tab shows them under "Labelling rules". Highlight PDFs and correct in Read by them.

1. A sentence that interprets ("therefore", "likely", "suggests", "our data support") is
   Conclusions, wherever it appears.
2. A sentence in the abstract that states a finding is Results.
3. Inside the Results section: how something was done (setups, figure legends, sample sizes)
   is Methods; what was observed is Results.
4. Someone else's finding cited anywhere ("Xu et al. found…", "personal communication") is
   Background.
5. Headings, titles and pointers like "Refer to Table 4" are left unhighlighted.

## Retraining on your own papers

`python retrain.py` holds out each paper you labelled by hand in turn, predicts its labels
with a model that never saw them, and compares three things: the original PubMed-only
model, a model retrained with your labels, and that model plus context (section, place in
the paper, neighbouring sentences). `python retrain.py --apply` saves the best one and
relabels `brain.db` only if it beats the original. Restart the app afterwards. To go back:
copy `models/classifier.pubmed.joblib` over `models/classifier.joblib` and `brain.backup.db`
over `brain.db`.

`python import_highlights.py "highlighted.pdf" PAPER_ID --apply` saves the colours of a
highlighted copy of a paper as your own labels (see the top of that file for the colours).

Held-out accuracy on 1,218 hand labels across 8 papers: original 57.4%, retrained 66.6%,
with context 67.3%. Research papers score 64-80%; reviews and clinical guides 42-47%.
Tried and dropped: labels made from section headings (lowered the score by 5 points once
there were enough hand labels) and a small local language model (fingerprints alone scored
7 points below word counts).
