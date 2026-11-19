"""Run the whole app once, in-process, and check its pages build.   python check_app.py

Uses a throwaway database with one made-up paper, never your real brain.db.
Needs a trained model (python train.py).
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
REAL_DB = HERE / "brain.db"
# Must be set before db is imported: db.py reads it once, at import.
os.environ["PAPER_BRAIN_DB"] = str(Path(tempfile.mkdtemp()) / "check.db")
sys.path.insert(0, str(HERE))

from streamlit.testing.v1 import AppTest

import db
from train import MODEL_PATH

assert db.DB_PATH != REAL_DB, "refusing to run against the real brain.db"
db.BACKUP_DIR = None   # a throwaway database must never overwrite a real backup
if not MODEL_PATH.exists():
    sys.exit("No trained model yet. Run: python train.py")

real_before = REAL_DB.stat().st_mtime if REAL_DB.exists() else None
db.add_paper("A check paper about blood pressure", "check.pdf", [
    {"page": 1, "position": i, "text": text, "label": label, "confidence": 0.9, "hedged": False,
     "section": "Results", "paragraph": 0}
    for i, (text, label) in enumerate([
        ("Mean systolic blood pressure decreased by 8 mm Hg in the low-sodium group.", "Results"),
        ("These findings suggest that a low-sodium diet lowers blood pressure.", "Conclusions"),
    ])])

at = AppTest.from_file(str(HERE / "app.py"), default_timeout=300).run()   # same process, so it uses the same db module
assert not at.exception, at.exception
html = " ".join(m.value for m in at.markdown)
assert 'class="pb-sent" data-id=' in html, "the paper text is missing"
assert at.get("iframe") or at.get("html"), "map is missing"
assert (REAL_DB.stat().st_mtime if REAL_DB.exists() else None) == real_before, "the real brain.db was touched"
print("APP OK")
