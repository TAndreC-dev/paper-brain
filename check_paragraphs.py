"""Make a 2-page PDF with separate paragraph blocks and check sections/paragraphs come out right."""
import os, sys, tempfile
sys.path.insert(0, os.getcwd())
import pymupdf
import db
from pdf_to_sentences import pdf_to_sentences

P1 = [("Introduction", None),
      ("Hypertension is a leading cause of cardiovascular disease worldwide. Previous studies reported mixed results for dietary interventions.", None),
      ("The aim of this study was to evaluate whether a low-sodium diet reduces systolic blood pressure in older adults. This paragraph is cut off by the page break and continues on the next page where the", None)]
P2 = [("remaining words of the sentence finally arrive and it ends properly here.", None),
      ("Results", None),
      ("Mean systolic blood pressure decreased by 8.4 mm Hg in the low-sodium group. Adherence was 87% in the intervention group.", None),
      ("These findings suggest that a low-sodium diet may lower blood pressure in older adults.", None)]
doc = pymupdf.open()
for blocks in (P1, P2):
    page, y = doc.new_page(), 60
    for text, _ in blocks:
        page.insert_textbox(pymupdf.Rect(60, y, 540, y + 70), text, fontsize=11)
        y += 90
data = doc.tobytes()

title, sents = pdf_to_sentences(data, filename="check.pdf")
for s in sents:
    print(s["page"], s["paragraph"], repr(s["section"]), s["text"][:70])
secs = [s["section"] for s in sents]
assert secs[0] == "Introduction" and secs[-1] == "Results", secs
assert sents[0]["paragraph"] == sents[1]["paragraph"] != sents[2]["paragraph"]
assert sents[-1]["paragraph"] != sents[-2]["paragraph"]

tmp = os.path.join(tempfile.mkdtemp(), "t.db")
for s in sents:
    s.update(label="Results", confidence=0.9, hedged=False)
pid = db.add_paper(title, "check.pdf", sents, db_path=tmp)
row = db.get_sentences(pid, tmp)[0]
assert row["section"] == "Introduction" and row["paragraph"] == 0, row
print("OK")

# figures: one real picture on page 2, one tiny icon that must be ignored
from pdf_to_sentences import page_images
big = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200)); big.clear_with(120)
tiny = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20)); tiny.clear_with(10)
doc[1].insert_image(pymupdf.Rect(60, 500, 360, 700), pixmap=big)
doc[0].insert_image(pymupdf.Rect(60, 700, 80, 720), pixmap=tiny)
scan = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 500)); scan.clear_with(200)
assert doc[0].get_text("words")
doc[0].insert_image(doc[0].rect, pixmap=scan, keep_proportion=False)   # a scanned page: must be ignored
pdf = os.path.join(tempfile.mkdtemp(), "fig.pdf"); doc.save(pdf)
imgs = page_images(pdf)
assert list(imgs) == [2] and len(imgs[2]) == 1 and imgs[2][0][:2] == b"\xff\xd8", {k: len(v) for k, v in imgs.items()}
print("FIGURES OK")

# references: two entries, one with a wrapped author list, then a notice that must be left out
from pdf_to_sentences import pdf_references
page = doc.new_page()
page.insert_textbox(pymupdf.Rect(60, 60, 540, 400), "\n".join([
    "References",
    "Anver, S., Roguev, A., Zofall, M., Krogan, N. J., Grewal, S. I., and",
    "Harmer, S. L. (2014). Yeast Xap5 functions with H2A.Z to suppress",
    "aberrant transcripts. EMBO Rep. 15, 894-902.",
    "Bolger AM, Lohse M, Usadel B. 2014. Trimmomatic: a flexible trimmer",
    "for Illumina sequence data. Bioinformatics. 30:2114-2120.",
    "Copyright © 2021 Someone. This is an open-access article from 2021.",
]), fontsize=10)
pdf = os.path.join(tempfile.mkdtemp(), "refs.pdf"); doc.save(pdf)
refs = pdf_references(pdf)
assert len(refs) == 2 and refs[0].startswith("Anver") and "Harmer" in refs[0] and refs[1].startswith("Bolger"), refs
print("REFERENCES OK")
