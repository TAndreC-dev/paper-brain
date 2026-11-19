"""SQLite storage and BM25 search for Paper Brain.

Everything lives in one file, brain.db: two tables (papers, sentences) plus
an FTS5 full-text index over sentences.text, which ranks with BM25.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
from datetime import date, datetime
from pathlib import Path

# PAPER_BRAIN_DB lets check_app.py point the whole app at a throwaway database.
DB_PATH = Path(os.environ.get("PAPER_BRAIN_DB") or Path(__file__).parent / "brain.db")

# Off by default. To keep a safety copy of brain.db, papers and model after every change, set a
# folder here, e.g. Path("D:/paper-brain-backup") or one that a cloud-sync app uploads.
BACKUP_DIR: Path | None = None
backup_error: str | None = None   # why the last backup failed (None = it worked); the app shows it

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id        INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    filename  TEXT NOT NULL,
    added_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sentences (
    id          INTEGER PRIMARY KEY,
    paper_id    INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    hedged      INTEGER NOT NULL,         -- 0 or 1
    section     TEXT,                     -- nearest heading above the sentence
    paragraph   INTEGER                   -- paragraph number within the paper
);
CREATE INDEX IF NOT EXISTS idx_sentences_paper ON sentences(paper_id, position);

-- Your corrections to the map: keep = 1 adds a link "a cites b", keep = 0 hides one.
CREATE TABLE IF NOT EXISTS link_edits (
    a    INTEGER NOT NULL,
    b    INTEGER NOT NULL,
    keep INTEGER NOT NULL,
    PRIMARY KEY (a, b)
);

-- Full-text index that points at sentences.text instead of copying it.
-- 'porter' stemming lets "reduce" match "reduced" and "reduces".
CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(
    text, content='sentences', content_rowid='id', tokenize='porter unicode61'
);
-- Keep the index in step with the table.
CREATE TRIGGER IF NOT EXISTS sentences_ai AFTER INSERT ON sentences BEGIN
    INSERT INTO sentences_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS sentences_ad AFTER DELETE ON sentences BEGIN
    INSERT INTO sentences_fts(sentences_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""

# Words too common to help a search.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "at", "as", "it", "that", "this", "what",
    "which", "how", "does", "do", "did", "from",
}


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # Databases made before sections/paragraphs existed get the columns added (old rows stay NULL).
    have = {row[1] for row in conn.execute("PRAGMA table_info(sentences)")}
    # 'corrected' = 1 where you set the label yourself; those rows can later retrain the classifier.
    for column, kind in (("section", "TEXT"), ("paragraph", "INTEGER"), ("corrected", "INTEGER DEFAULT 0")):
        if column not in have:
            conn.execute(f"ALTER TABLE sentences ADD COLUMN {column} {kind}")
    return conn


def backup_db(db_path: Path | str = DB_PATH, files: bool = True) -> None:
    """Copy brain.db, the PDFs and the code into BACKUP_DIR. Never raises: a failure is left in backup_error.

    Uses SQLite's backup API, so the copy is a consistent snapshot even if the app is
    writing. It is written to a temporary file first and swapped in when complete, so an
    interrupted backup cannot damage the previous one. The first backup of each day is
    also kept as brain-YYYY-MM-DD.db (newest 7), in case a bad state gets backed up.
    """
    global backup_error
    if BACKUP_DIR is None or Path(db_path) != DB_PATH:
        return   # backup is off, or: the check scripts use throwaway databases; those must not replace the real backup
    try:
        BACKUP_DIR.mkdir(exist_ok=True)   # no parents: if the drive is missing, fail instead of writing elsewhere
        latest, tmp = BACKUP_DIR / "brain.db", BACKUP_DIR / "brain.db.tmp"
        tmp.unlink(missing_ok=True)
        src, dst = sqlite3.connect(db_path), sqlite3.connect(tmp)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        os.replace(tmp, latest)
        daily = BACKUP_DIR / f"brain-{date.today()}.db"
        if not daily.exists():
            shutil.copy2(latest, daily)
            for old in sorted(BACKUP_DIR.glob("brain-????-??-??.db"))[:-7]:
                old.unlink()
        # The rest of what a restore needs: the PDFs (figures, works cited), the code and the trained
        # model. Only new or changed files are copied. Nothing is ever deleted from the backup.
        root = DB_PATH.parent
        # files=False skips this (about a second of checking) for small edits such as one corrected label.
        for pattern in ("papers/*.pdf", "*.py", "*.md", "*.txt", "*.bat", "models/*", ".streamlit/*") if files else ():
            for src in root.glob(pattern):
                dst = BACKUP_DIR / src.relative_to(root)
                if src.is_file() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src, dst)
        backup_error = None
    except (OSError, sqlite3.Error) as e:
        backup_error = f"{type(e).__name__}: {e}"


def add_paper(title: str, filename: str, sentences: list[dict], db_path: Path | str = DB_PATH) -> int:
    """Save a paper and its labeled sentences. Returns the new paper id.

    Each sentence dict needs: page, position, text, label, confidence, hedged.
    """
    conn = connect(db_path)
    with conn:  # one transaction: the paper is saved completely or not at all
        cur = conn.execute(
            "INSERT INTO papers (title, filename, added_at) VALUES (?, ?, ?)",
            (title, filename, datetime.now().isoformat(timespec="seconds")),
        )
        paper_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO sentences (paper_id, page, position, text, label, confidence, hedged, section, paragraph) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(paper_id, s["page"], s["position"], s["text"], s["label"],
              float(s["confidence"]), int(bool(s["hedged"])), s.get("section"), s.get("paragraph"))
             for s in sentences],
        )
    conn.close()
    backup_db(db_path)
    return paper_id


def delete_paper(paper_id: int, db_path: Path | str = DB_PATH) -> None:
    conn = connect(db_path)
    with conn:
        conn.execute("DELETE FROM sentences WHERE paper_id = ?", (paper_id,))
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        conn.execute("DELETE FROM link_edits WHERE a = ? OR b = ?", (paper_id, paper_id))
    conn.close()
    backup_db(db_path)


def set_link(a: int, b: int, keep: bool, db_path: Path | str = DB_PATH) -> None:
    conn = connect(db_path)
    with conn:
        conn.execute("INSERT OR REPLACE INTO link_edits (a, b, keep) VALUES (?, ?, ?)", (a, b, int(keep)))
    conn.close()
    backup_db(db_path, files=False)


def link_edits(db_path: Path | str = DB_PATH) -> dict[tuple[int, int], bool]:
    conn = connect(db_path)
    rows = conn.execute("SELECT a, b, keep FROM link_edits").fetchall()
    conn.close()
    return {(a, b): bool(keep) for a, b, keep in rows}


def list_papers(db_path: Path | str = DB_PATH) -> list[dict]:
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT p.id, p.title, p.filename, p.added_at, COUNT(s.id) AS n_sentences "
        "FROM papers p LEFT JOIN sentences s ON s.paper_id = p.id "
        "GROUP BY p.id ORDER BY p.added_at DESC, p.id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_labels(sentence_ids: list[int], label: str, db_path: Path | str = DB_PATH) -> None:
    """Your own label for these sentences; it replaces the classifier's guess."""
    conn = connect(db_path)
    with conn:
        conn.executemany("UPDATE sentences SET label = ?, confidence = 1.0, corrected = 1 WHERE id = ?",
                         [(label, int(i)) for i in sentence_ids])
    conn.close()
    backup_db(db_path, files=False)   # your corrections are the one thing that cannot be rebuilt from the PDFs


def get_sentences(paper_id: int, db_path: Path | str = DB_PATH) -> list[dict]:
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT * FROM sentences WHERE paper_id = ? ORDER BY position", (paper_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def stats(db_path: Path | str = DB_PATH) -> dict:
    conn = connect(db_path)
    papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    sentences = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    conn.close()
    return {"papers": papers, "sentences": sentences}


def query_terms(query: str) -> list[str]:
    """The words of a search that are actually used for matching."""
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if w not in _STOPWORDS] or words


def search(
    query: str,
    label: str | None = None,      # "Results", "Methods", ... or None for any
    hedged: bool | None = None,    # True = only hedged, False = hide hedged, None = both
    paper_id: int | None = None,   # one paper, or None for all
    limit: int = 20,
    db_path: Path | str = DB_PATH,
) -> list[dict]:
    """Top matching sentences, best first. Only ever returns stored sentences."""
    terms = query_terms(query)
    if not terms:
        return []
    # Quote every term so punctuation in the search can't break FTS5 syntax.
    # OR means a sentence needs any of the words; BM25 puts sentences that
    # match more (and rarer) words on top.
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

    where, params = ["sentences_fts MATCH ?"], [match]
    if label:
        where.append("s.label = ?")
        params.append(label)
    if hedged is not None:
        where.append("s.hedged = ?")
        params.append(int(hedged))
    if paper_id is not None:
        where.append("s.paper_id = ?")
        params.append(paper_id)
    params.append(limit)

    conn = connect(db_path)
    rows = conn.execute(
        "SELECT s.id, s.paper_id, s.page, s.position, s.text, s.label, s.confidence, s.hedged, "
        "       p.title AS paper_title, p.filename, "
        "       -bm25(sentences_fts) AS score "          # FTS5's bm25() is lower-is-better
        "FROM sentences_fts "
        "JOIN sentences s ON s.id = sentences_fts.rowid "
        "JOIN papers p ON p.id = s.paper_id "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY bm25(sentences_fts) LIMIT ?",
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
