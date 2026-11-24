"""Paper Brain - a local, searchable second brain for research papers.

    streamlit run app.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st

import db
from download import find_doi, open_access_pdf
from hedges import find_hedges, is_hedged
from pdf_to_sentences import page_images, pdf_references, pdf_to_sentences
from train import LABELS, MODEL_PATH, label_sentences, load_model

st.set_page_config(page_title="Paper Brain", layout="centered")

# One color per label: (text/border color, soft background), chosen for a white page.
# The light theme itself is set in .streamlit/config.toml.
COLORS = {
    "Background":  ("#5b6676", "rgba(91,102,118,0.10)"),
    "Methods":     ("#1d64c0", "rgba(29,100,192,0.09)"),
    "Results":     ("#14794a", "rgba(20,121,74,0.10)"),
    "Conclusions": ("#a0600a", "rgba(160,96,10,0.11)"),
    "Unlabeled":   ("#6f6f6c", "transparent"),      # a highlight you removed by hand
}
PLAIN = ("Background", "Methods", "Unlabeled")      # shown as ordinary text in Read
# The classifier's guesses below this are not highlighted. Measured on 8 held-out hand-labelled papers:
# at 0.6 the highlights shown match your labels 80% of the time (82% on research papers), but only about
# half of the real Results/Conclusions get highlighted. Re-measure after adding many more labelled papers.
MIN_CONFIDENCE = 0.6
# Measured: papers without standard headings (reviews, short reports) get noticeably worse labels.
NO_HEADINGS = ("No standard section headings (Introduction, Methods, Results, Discussion) were found in this "
               "paper, so its labels are less reliable. This is common for reviews and short reports.")

# Click a sentence in Read (or select several) -> a small menu -> your label goes back to Python.
LABEL_MENU = st.components.v2.component(
    "pb_label_menu",
    html='<div id="menu"></div>',
    css="""
#menu {position: fixed; display: none; flex-direction: column; gap: 2px; z-index: 1001; background: #fff;
       border: 1px solid #111; border-radius: 8px; padding: 6px; box-shadow: 0 8px 30px rgba(0,0,0,0.15);}
#menu button {all: unset; cursor: pointer; padding: 6px 12px 6px 10px; border-left: 4px solid transparent;
              font: 0.8rem ui-monospace, Consolas, monospace; letter-spacing: 0.06em; color: #111;}
#menu button:hover, #menu button:focus {background: #f1f1ef;}
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const menu = parentElement.querySelector("#menu")
  const hide = () => { menu.style.display = "none" }
  let spans = []
  menu.replaceChildren()
  for (const [label, text, color, bg] of data.options) {
    const button = document.createElement("button")
    button.type = "button"
    button.textContent = text
    button.style.borderLeftColor = color
    button.onclick = () => {
      // Recolour here, at once; Python only saves the choice and does not redraw the paper.
      for (const s of spans) { s.style.background = bg; s.title = label + " \u00b7 set by you" }
      setTriggerValue("pick", { ids: spans.map((s) => Number(s.dataset.id)), label })
      hide()
    }
    menu.appendChild(button)
  }
  const onUp = (e) => {
    if (e.composedPath().includes(menu)) return
    const selection = window.getSelection()
    spans = []
    if (selection && !selection.isCollapsed) {
      spans = [...document.querySelectorAll(".pb-sent[data-id]")].filter((s) => selection.containsNode(s, true))
    } else if (e.target.closest && e.target.closest(".pb-sent[data-id]")) {
      spans = [e.target.closest(".pb-sent[data-id]")]
    }
    if (!spans.length) return hide()
    menu.style.display = "flex"
    menu.style.left = Math.min(e.clientX, window.innerWidth - 220) + "px"
    menu.style.top = Math.min(e.clientY + 14, window.innerHeight - 200) + "px"
  }
  const onKey = (e) => { if (e.key === "Escape") hide() }
  // This function runs again when the page redraws: drop the old listeners first.
  if (parentElement.pbOff) parentElement.pbOff()
  document.addEventListener("mouseup", onUp)
  document.addEventListener("keydown", onKey)
  parentElement.pbOff = () => {
    document.removeEventListener("mouseup", onUp)
    document.removeEventListener("keydown", onKey)
  }
  return parentElement.pbOff
}
""",
)

st.markdown("""
<style>
:root {--pb-mono: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
       --pb-line: #e6e6e4; --pb-muted: #6f6f6c;}
.block-container {max-width: 820px; padding-top: 4.5rem;}
header[data-testid="stHeader"] {background: transparent;}

/* Thin headline, small spaced-out mono labels */
h1 {font-weight: 200 !important; font-size: 3.2rem !important; letter-spacing: -0.02em;}
h3 {font-weight: 300 !important; font-size: 1.5rem !important; margin-top: 1.2rem;}
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p,
[data-testid="stTab"] p, .stButton button p, .pb-meta, .pb-chip, .pb-section, .pb-pageno {
    font-family: var(--pb-mono); font-size: 0.7rem !important; letter-spacing: 0.12em;
    text-transform: uppercase;}
[data-testid="stCaptionContainer"], .pb-meta {color: var(--pb-muted) !important;}

/* Flat, hairline-bordered controls */
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 1.6rem; border-bottom: 1px solid var(--pb-line);}
.stButton button {white-space: nowrap; border-radius: 999px; border: 1px solid #111; padding: 0.35rem 1.1rem; box-shadow: none;}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
    background: #fff !important; border: 1px solid var(--pb-line) !important; border-radius: 8px !important;}
[data-baseweb="base-input"], [data-baseweb="input"] input {background: #fff !important;}
hr {border-color: var(--pb-line) !important;}

.pb-chip {display:inline-block; padding:2px 10px; border-radius:999px; margin-right:6px; border:1px solid;}
.pb-hedge-chip {border:1px dashed var(--pb-muted); color: var(--pb-muted);}
.pb-result {padding:16px 0 16px 16px; margin:0; border-left:2px solid; border-bottom:1px solid var(--pb-line);
            background: transparent !important;}
.pb-result .pb-text {font-size:1.02rem; line-height:1.6; margin:8px 0 6px 0;}
.pb-read {font-size:1.0rem;}
.pb-para {line-height:1.9; margin:0 0 1.2rem 0;}
.pb-fig {display:block; max-width:100%; margin:1.4rem auto; border:1px solid var(--pb-line); border-radius:4px;}
.pb-section {margin:2.4rem 0 1rem 0; padding-top:1rem; border-top:1px solid var(--pb-line);}
.pb-ref {display:flex; gap:12px; margin:0 0 1rem 0; line-height:1.6;}
.pb-refno {color: var(--pb-muted); min-width:2.2em; text-align:right;}
.st-key-kw_panel {position:fixed; top:90px; right:24px; width:420px; max-width:92vw; max-height:75vh;
    overflow:auto; resize:both; z-index:1000; background:#fff; border:1px solid #111; border-radius:10px;
    padding:14px 16px; box-shadow:0 8px 30px rgba(0,0,0,0.12);}
.pb-drag {cursor:move; user-select:none; font-family:var(--pb-mono); font-size:0.7rem; letter-spacing:0.12em;
    text-transform:uppercase; padding:8px 0;}
.pb-pageno {color: var(--pb-muted); font-size:0.6rem !important; margin-right:4px;}
.pb-sent {padding:2px 4px; border-radius:3px; box-decoration-break:clone; -webkit-box-decoration-break:clone;}
.pb-hedge {text-decoration:underline wavy; text-decoration-thickness:1.5px; text-underline-offset:3px;}
mark.pb-hit {background:#fff1a8; color:inherit; padding:0 1px; border-radius:2px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


def chip(label: str) -> str:
    color, bg = COLORS[label]
    return f'<span class="pb-chip" style="color:{color};background:{bg};border-color:{color}">{label}</span>'


def render_text(text: str, underline_hedges: bool = True, highlight: list[str] | None = None) -> str:
    """Escape a sentence for HTML, underlining hedge cues and marking search hits.

    Text from a PDF is never trusted as HTML: every piece goes through html.escape.
    """
    marks: list[tuple[int, int, str]] = []
    if underline_hedges:
        marks += [(a, b, "hedge") for a, b in find_hedges(text)]
    if highlight:
        pattern = r"\b(?:" + "|".join(re.escape(t) for t in highlight) + r")\w*"
        marks += [(m.start(), m.end(), "hit") for m in re.finditer(pattern, text, re.IGNORECASE)]
    out, cursor = [], 0
    for start, end, kind in sorted(marks):
        if start < cursor:      # overlapping marks: keep the first
            continue
        out.append(html.escape(text[cursor:start]))
        piece = html.escape(text[start:end])
        if kind == "hedge":
            out.append(f'<span class="pb-hedge" title="hedge word">{piece}</span>')
        else:
            out.append(f'<mark class="pb-hit">{piece}</mark>')
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def result_html(r: dict, terms: list[str]) -> str:
    """One search hit: label, the exact sentence, and where it came from."""
    color, _ = COLORS[r["label"]]
    hedge = '<span class="pb-chip pb-hedge-chip">hedged</span>' if r["hedged"] else ""
    return (f'<div class="pb-result" style="border-left-color:{color}">'
            f'{chip(r["label"])}{hedge}'
            f'<div class="pb-text">{render_text(r["text"], highlight=terms)}</div>'
            f'<div class="pb-meta">{html.escape(r["paper_title"])} · page {r["page"]} · '
            f'label confidence {r["confidence"]:.0%}</div></div>')


@st.fragment   # typing a keyword redraws only the panel, not the whole paper
def keyword_panel(paper_id: int) -> None:
    """Floating panel: keywords -> exact sentences in this paper (background first), then in the other papers."""
    with st.container(key="kw_panel"):   # the key becomes the CSS class .st-key-kw_panel, which floats it
        col_handle, col_close = st.columns([3, 1])
        col_handle.markdown('<div class="pb-drag">Search keywords · drag here to move</div>', unsafe_allow_html=True)
        if col_close.button("Close", key="kw_close"):
            st.session_state["kw_open"] = False
            st.rerun()
        kw = st.text_input("Keywords", placeholder="e.g. splicing circadian", key="kw_text")
        if not kw.strip():
            return
        terms = db.query_terms(kw)
        background = db.search(kw, label="Background", paper_id=paper_id, limit=5)
        rest = [r for r in db.search(kw, paper_id=paper_id, limit=20) if r["label"] != "Background"][:6]
        others = [r for r in db.search(kw, limit=40) if r["paper_id"] != paper_id][:8]
        for heading, hits in (("Background in this paper", background),
                              ("Methods, results and conclusions in this paper", rest),
                              ("Related sentences in your other papers", others)):
            st.caption(f"{heading} ({len(hits)})")
            if hits:
                st.markdown("".join(result_html(r, terms) for r in hits), unsafe_allow_html=True)


# Dragging the panel: the new position is written as a CSS rule, so it survives Streamlit redrawing the page.
st.html("""<script>
if (!window.pbDrag) {
  window.pbDrag = true;
  let grab = null;
  document.addEventListener("mousedown", (e) => {
    const handle = e.target.closest ? e.target.closest(".pb-drag") : null;
    if (!handle) return;
    const box = handle.closest(".st-key-kw_panel").getBoundingClientRect();
    grab = {dx: e.clientX - box.left, dy: e.clientY - box.top};
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => {
    if (!grab) return;
    const rule = document.getElementById("pb-pos") ||
      document.head.appendChild(Object.assign(document.createElement("style"), {id: "pb-pos"}));
    const x = Math.max(0, Math.min(window.innerWidth - 80, e.clientX - grab.dx));
    const y = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - grab.dy));
    rule.textContent = ".st-key-kw_panel{left:" + x + "px !important;top:" + y + "px !important;right:auto !important}";
  });
  document.addEventListener("mouseup", () => { grab = null; });
}
</script>""", unsafe_allow_javascript=True)


PAPERS_DIR = db.DB_PATH.parent / "papers"   # a copy of every uploaded PDF is kept next to the database in use


@st.cache_data(show_spinner=False)
def figures(path: Path, modified: float) -> dict[int, list[bytes]]:
    return page_images(path)   # 'modified' is only a cache key: a re-added PDF is read again


@st.cache_data(show_spinner=False)
def references(path: Path, modified: float) -> list[str]:
    return pdf_references(path)


def squash(text: str) -> str:
    """Letters and digits only, so line breaks and hyphens in a reference don't hide a title."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@lru_cache(maxsize=None)
def title_pattern(title: str) -> re.Pattern:
    # The whole title, between punctuation: "The plant immune system" must not match
    # "Pivoting the plant immune system from dissection to deployment".
    return re.compile(r"(?:^|[.)\]:,])\W*" + r"\W*".join(squash(title)) + r"\W*(?:[.?!]|$)", re.IGNORECASE)


def cited_in_library(refs: list[str], others: list[dict]) -> list[dict | None]:
    """For each reference, the saved paper it points to (matched by title), or None."""
    patterns = [(title_pattern(p["title"]), p) for p in others if len(squash(p["title"])) >= 10]
    return [next((p for pattern, p in patterns if pattern.search(r)), None) for r in refs]


def library_links(papers: list[dict]) -> set[tuple[int, int]]:
    """(citing id, cited id) for every saved paper that cites another saved paper, plus manual edits."""
    links = set()
    for p in papers:
        pdf = PAPERS_DIR / Path(p["filename"]).name
        if pdf.exists():
            others = [o for o in papers if o["id"] != p["id"]]
            hits = cited_in_library(references(pdf, pdf.stat().st_mtime), others)
            links |= {(p["id"], hit["id"]) for hit in hits if hit}
    # Your own corrections win over what was found automatically.
    ids = {p["id"] for p in papers}
    for pair, keep in db.link_edits().items():
        if keep and set(pair) <= ids:
            links.add(pair)
        elif not keep:
            links.discard(pair)
    return links


def cite_pattern(ref: str) -> re.Pattern | None:
    """Matches a sentence that cites this reference (first author's surname + year)."""
    name, year = re.match(r"[^\s,]+", ref), re.search(r"\b(19|20)\d\d", ref)
    if not name or not year:
        return None
    # "Anver et al., 2014" / "Anver and Harmer (2014)": the year soon after the surname.
    return re.compile(rf"\b{re.escape(name.group())}\b[^;]{{0,40}}?{year.group()}")


def mentions(ref: str, texts: list[str]) -> int:
    """How many sentences of the paper cite this reference."""
    cite = cite_pattern(ref)
    return sum(bool(cite.search(text)) for text in texts) if cite else 0




def ref_html(number: int, ref: str, in_library: bool, count: int = 0) -> str:
    doi = find_doi(ref)
    links = (f'<a href="https://doi.org/{html.escape(doi)}" target="_blank">open</a> · ' if doi else "") + \
        f'<a href="https://scholar.google.com/scholar?q={quote_plus(ref[:200])}" target="_blank">search</a>'
    note = f" · cited in {count} sentences of this paper" if count else ""
    owned = '<span class="pb-chip">in your library</span>' if in_library else ""
    return (f'<div class="pb-ref"><span class="pb-refno">{number}.</span><div>{owned}{html.escape(ref)}'
            f'<div class="pb-meta">{links}{note}</div></div></div>')


def paper_label(p: dict) -> str:
    return f"{p['title']} (#{p['id']})"


def jump(label: str) -> None:
    st.session_state["read_paper"] = label

st.title("Paper Brain :gray[:small[· a Zettelkasten for research papers]]")
st.markdown("Every sentence labeled by what it does. Search only returns real sentences "
           "from your papers, with the source attached. Nothing is generated.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first (see README.md).")
    st.stop()

def add_pdf(data: bytes, name: str) -> tuple[str, list[dict]]:
    """Label a PDF and save it (uploads and downloads both come through here).

    Raises ValueError with a message for the user when the paper is not added.
    """
    name = Path(name).name                       # no folders from the upload
    saved = db.list_papers()
    if name in {p["filename"] for p in saved}:
        raise ValueError(f"A file named **{name}** is already saved. Rename this one and upload it again.")
    try:
        title, sentences = pdf_to_sentences(data, filename=name)
    except Exception as e:
        raise ValueError(f"Could not read **{name}**: {e}")
    if not sentences:
        raise ValueError(f"No text found in **{name}**. It may be a scanned PDF (images only).")
    if title.casefold() in {p["title"].casefold() for p in saved}:   # same paper under another file name
        raise ValueError(f"Skipped **{name}**: you already have **{title}**.")
    dup = db.find_duplicate([s["text"] for s in sentences])   # same paper, other file name and title
    if dup:
        raise ValueError(f"Skipped **{name}**: {dup['share']:.0%} of the shorter paper's sentences also appear in "
                         f"**{dup['title']}** (#{dup['id']}), already saved. It looks like the same paper.")
    labeled = label_sentences(get_model(), [s["text"] for s in sentences], [s.get("section") for s in sentences])
    for s, (label, confidence) in zip(sentences, labeled):
        s.update(label=label, confidence=confidence, hedged=is_hedged(s["text"]))
    PAPERS_DIR.mkdir(exist_ok=True)
    (PAPERS_DIR / name).write_bytes(data)        # before add_paper, so its backup already includes the PDF
    db.add_paper(title, name, sentences)
    return title, sentences


papers = db.list_papers()
if db.backup_error:   # the change itself is saved; only the safety copy is behind
    st.warning(f"Your change was saved, but the backup copy in {db.BACKUP_DIR} could not be updated "
               f"({db.backup_error}). Is that folder available?")
# A click on the map or on a saved paper reloads the page as ?paper=<id>: open that paper in Read.
opened = next((p for p in papers if str(p["id"]) == st.query_params.get("paper")), None)
if opened and "read_paper" not in st.session_state:
    st.session_state["read_paper"] = paper_label(opened)
tab_add, tab_ask, tab_read, tab_map = st.tabs(["Add", "Ask", "Read", "Map"], default="Read" if opened else None)

# ---------------------------------------------------------------- Add
with tab_add:
    st.subheader("Add a paper")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Label and save", type="primary"):
        for up in uploads:
            with st.spinner(f"Reading {up.name}..."):
                try:
                    title, sentences = add_pdf(up.getvalue(), up.name)
                except ValueError as e:
                    st.warning(str(e))
                    continue
            counts = {l: sum(s["label"] == l for s in sentences) for l in LABELS}
            n_hedged = sum(s["hedged"] for s in sentences)
            st.success(f"Saved **{title}**: {len(sentences)} sentences, {n_hedged} hedged.")
            st.markdown(" ".join(f"{chip(l)} {counts[l]}&nbsp;&nbsp;" for l in LABELS),
                        unsafe_allow_html=True)
            if not any(s.get("section") for s in sentences):
                st.warning(NO_HEADINGS)
        papers = db.list_papers()

    st.divider()
    totals = db.stats()
    st.subheader(f"Saved papers ({totals['papers']})")
    st.caption(f"{totals['sentences']:,} sentences stored in brain.db")
    if not papers:
        st.info("Nothing saved yet. Upload a PDF above.")
    for p in papers:
        col_info, col_del = st.columns([4, 1])
        col_info.markdown(f"<a href='?paper={p['id']}' target='_self'><b>{html.escape(p['title'])}</b></a>  \n"
                          f"<span class='pb-meta'>{html.escape(p['filename'])} · "
                          f"{p['n_sentences']} sentences · added {p['added_at'][:10]}</span>",
                          unsafe_allow_html=True)
        # Two steps: deleting also removes your own labels, which nothing can rebuild.
        if col_del.popover("Delete").button(f"Delete for good ({p['n_mine']} of your labels)",
                                            key=f"del{p['id']}", type="primary"):
            db.delete_paper(p["id"])
            (PAPERS_DIR / Path(p["filename"]).name).unlink(missing_ok=True)
            st.rerun()

# ---------------------------------------------------------------- Ask
with tab_ask:
    st.subheader("Ask your papers")
    if not papers:
        st.info("Add a paper first.")
    else:
        query = st.text_input("Search", placeholder="e.g. effect on blood pressure")
        c1, c2, c3 = st.columns([2, 2, 3])
        label_choice = c1.selectbox("Label", ["Any"] + LABELS)
        firm_choice = c2.selectbox("Firmness", ["Any", "Hide hedged", "Only hedged"])
        paper_options = {"All papers": None} | {f"{p['title']} (#{p['id']})": p["id"] for p in papers}
        paper_choice = c3.selectbox("Paper", list(paper_options))

        if query.strip():
            results = db.search(
                query,
                label=None if label_choice == "Any" else label_choice,
                hedged={"Any": None, "Hide hedged": False, "Only hedged": True}[firm_choice],
                paper_id=paper_options[paper_choice],
                limit=20,
            )
            if not results:
                st.warning("No sentence matches those words with these filters. Search matches "
                           "words, not meaning, so try a synonym the paper might use.")
            else:
                st.caption(f"Top {len(results)} sentences, best match first")
            terms = db.query_terms(query)
            st.markdown("".join(result_html(r, terms) for r in results), unsafe_allow_html=True)

MARKS = ("Results", "Conclusions")   # the labels that are highlighted in Read

# Decided after comparing two highlightings of the same paper: the same text in README.md.
LABELLING_RULES = """
1. A sentence that **interprets** ("therefore", "likely", "suggests", "our data support") is **Conclusions**, wherever it appears.
2. A sentence in the **abstract** that states a finding is **Results**.
3. Inside the Results section: **how** something was done (setups, figure legends, sample sizes) is **Methods**; **what was observed** is **Results**.
4. **Someone else's finding** cited anywhere ("Xu et al. found…", "personal communication") is **Background**.
5. Headings, titles and pointers like "Refer to Table 4" are **left unhighlighted** (Remove highlight).
"""


@st.fragment   # a pick reruns only this bit: the menu has already recoloured the sentence in the page
def label_menu(blind: bool = False) -> None:
    # Hiding model labels: your own picks show in colour (plain ones too), so you can see what you've done.
    pick = LABEL_MENU(key="label_menu", on_pick_change=lambda: None, data={"options": [
        [label, text, COLORS[label][0], COLORS[label][1] if blind or label not in PLAIN else ""] for label, text in
        [(l, l) for l in MARKS] + [("Methods", "Methods"), ("Background", "Background (plain)"),
                                   ("Unlabeled", "Remove highlight")]]}).pick
    if pick and pick.get("label") in COLORS and pick.get("ids"):
        db.set_labels(pick["ids"], pick["label"])


# ---------------------------------------------------------------- Read
with tab_read:
    st.subheader("Read a paper")
    if not papers:
        st.info("Add a paper first.")
    else:
        by_name = {f"{p['title']} (#{p['id']})": p["id"] for p in papers}
        picked = st.selectbox("Paper (type any word of the title to search)", list(by_name), key="read_paper")
        blind = st.checkbox("Hide model labels while I label", key="blind",
                            help="Unlabelled sentences show as plain text and tooltips hide the model's guess. "
                                 "Your own labels still show in their colours.")
        # The label filter and keyword panel would reveal the model's guesses, so they are off while hiding.
        show = st.multiselect("Show labels", LABELS, default=LABELS, disabled=blind)
        if st.button("Search keywords", disabled=blind):   # shows or hides the floating panel
            st.session_state["kw_open"] = not st.session_state.get("kw_open", False)
        if st.session_state.get("kw_open") and not blind:
            keyword_panel(by_name[picked])
        st.markdown(" ".join(chip(l) for l in MARKS) +
                    ' <span class="pb-meta">background and methods are plain text ·</span>'
                    ' <span class="pb-hedge">hedge words are underlined</span>',
                    unsafe_allow_html=True)
        st.caption("Wrong highlight? Click a sentence, or select several, and pick the right label or remove it.")
        with st.expander("Labelling rules (keep your corrections consistent)"):
            st.markdown(LABELLING_RULES)
        label_menu(blind)
        filename = next(p["filename"] for p in papers if p["id"] == by_name[picked])
        pdf = PAPERS_DIR / Path(filename).name
        figs = dict(figures(pdf, pdf.stat().st_mtime)) if pdf.exists() else {}   # page -> images still to place
        if not pdf.exists():
            st.caption("No figures: this paper was added before PDFs were kept. Re-add it to see them.")

        def place_figures(up_to_page: float) -> None:
            for n in sorted(k for k in figs if k <= up_to_page):
                for img in figs.pop(n):
                    parts.append('<img class="pb-fig" alt="Figure from page %d" '
                                 'src="data:image/jpeg;base64,%s">' % (n, base64.b64encode(img).decode()))

        pid = by_name[picked]
        refs = references(pdf, pdf.stat().st_mtime) if pdf.exists() else []
        sentences = db.get_sentences(pid)
        texts = [s["text"] for s in sentences]
        counts = [mentions(r, texts) for r in refs]
        # The sources this paper leans on most, to look up, download and upload.
        top = sorted((i for i in range(len(refs)) if counts[i] >= 2), key=lambda i: -counts[i])[:8]
        if sentences and not any(s["section"] for s in sentences):
            st.warning(NO_HEADINGS)

        parts, section, paragraph, page = [], "", None, None
        for s in sentences:
            if not blind and s["label"] in LABELS and s["label"] not in show:
                continue
            if page is not None and s["page"] != page:
                place_figures(page)         # a page's figures go after its last sentence
            if (s["section"] or "") != section:
                section = s["section"] or ""
                parts.append(f'</div><div class="pb-section">{html.escape(section)}</div>')
                paragraph = None
            # Papers saved before paragraphs existed fall back to one block per page.
            key = s["paragraph"] if s["paragraph"] is not None else f'page{s["page"]}'
            if key != paragraph:
                parts.append(('</div>' if paragraph is not None else '') + '<div class="pb-para">')
                paragraph = key
            if s["page"] != page:
                page = s["page"]
                parts.append(f'<span class="pb-pageno">p.{page}</span>')
            label = s["label"]
            if blind:   # only your own labels are shown; the model's guess stays hidden
                style = f' style="background:{COLORS[label][1]}"' if s["corrected"] else ""
                tip = f"{label} · set by you" if s["corrected"] else "not labelled by you yet"
            else:
                unsure = not s["corrected"] and s["confidence"] < MIN_CONFIDENCE
                style = "" if label in PLAIN or unsure else f' style="background:{COLORS[label][1]}"'
                tip = f'{label} · ' + ("set by you" if s["corrected"]
                                       else f'{s["confidence"]:.0%} confident' + (", too unsure to highlight" * unsure))
            parts.append(f'<span class="pb-sent" data-id="{s["id"]}"{style} title="{tip}">'
                         f'{render_text(s["text"])}</span> ')
        place_figures(float("inf"))
        st.markdown(f'<div class="pb-read"><div>{"".join(parts)}</div></div>', unsafe_allow_html=True)

        hits = cited_in_library(refs, [p for p in papers if p["id"] != pid])

        def heading(text: str) -> None:
            st.markdown(f'<div class="pb-section">{html.escape(text)}</div>', unsafe_allow_html=True)

        if top:
            heading("Key sources: cited most often in this paper")
            st.markdown("".join(ref_html(i + 1, refs[i], bool(hits[i]), counts[i]) for i in top),
                        unsafe_allow_html=True)

        links = library_links(papers)
        linked = [("Cites", o) for o in papers if (pid, o["id"]) in links] + \
                 [("Cited by", o) for o in papers if (o["id"], pid) in links]
        if linked:
            heading("Linked papers in your library")
            for word, o in linked:
                st.button(f"{word}: {o['title'][:60]}", key=f"jump-{word}-{o['id']}",
                          on_click=jump, args=(paper_label(o),))

        heading(f"Works cited ({len(refs)})")
        if not refs:
            st.caption("No reference list was found in this paper's PDF.")
        wanted = {r: find_doi(r) for hit, r in zip(hits, refs) if not hit and find_doi(r)}
        if wanted:
            with st.expander("Download free copies"):
                chosen = st.multiselect("Download free copies of", list(wanted), format_func=lambda r: r[:110],
                                        placeholder="Pick cited papers", key="cite_pick")
                if chosen and st.button("Download selected", type="primary", key="cite_go"):
                    added = 0
                    for r in chosen:
                        with st.spinner(f"Looking for {r[:60]}..."):
                            try:
                                data = open_access_pdf(wanted[r])
                            except Exception:       # no internet, or the lookup service is down
                                data = None
                            if data is None:
                                st.warning(f"No free copy could be fetched: [{wanted[r]}](https://doi.org/{wanted[r]}) "
                                           "— open the link, save the PDF, and upload it in Add.")
                                continue
                            try:
                                title, _ = add_pdf(data, re.sub(r"[^\w.-]", "_", wanted[r]) + ".pdf")
                                st.success(f"Added **{title}**.")
                                added += 1
                            except ValueError as e:
                                st.warning(str(e))
                    if added:
                        st.caption("Refresh the page to see the new papers in the lists.")
        st.markdown("".join(ref_html(i + 1, r, bool(hits[i]), counts[i]) for i, r in enumerate(refs)),
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- Map
with tab_map:
    st.subheader("Map of your papers")
    if not papers:
        st.info("No papers yet. Add one in the Add tab.")
    else:
        st.caption("Builds itself from who cites whom, and updates when you add papers. Papers that are "
                   "connected share a colour; unrelated topics drift apart. Click a paper to read it. Drag to rearrange, scroll to zoom.")
        links = library_links(papers)
        # Colour = connected group: follow links until nothing new is reached.
        group = {p["id"]: p["id"] for p in papers}
        for _ in papers:
            for a, b in links:
                group[a] = group[b] = min(group[a], group[b])
        manual = {pair for pair, keep in db.link_edits().items() if keep}
        graph = json.dumps({
            "nodes": [{"id": p["id"], "label": textwrap.fill(p["title"][:70], 24), "title": p["title"],
                       "group": group[p["id"]]} for p in papers],
            "edges": [{"from": a, "to": b, "dashes": (a, b) in manual} for a, b in sorted(links)],
        }).replace("</", "<\\/")
        # shortcut: the drawing library loads from the internet, so the map is blank offline.
        # Save vis-network.min.js next to app.py and inline it if that becomes a problem.
        st.iframe(f"""
<div id="map" style="height:600px;border:1px solid #e6e6e4;border-radius:8px"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/standalone/umd/vis-network.min.js"></script>
<script>
  const data = {graph};
  const net = new vis.Network(document.getElementById("map"), data, {{
    nodes: {{shape: "dot", size: 12, font: {{size: 12, face: "Helvetica"}}}},
    edges: {{arrows: {{to: {{scaleFactor: 0.5}}}}, color: {{color: "#999999"}}}},
    physics: {{solver: "forceAtlas2Based", stabilization: {{iterations: 150}}}},
    interaction: {{hover: true}}
  }});
  // The map is built while its tab is still hidden, so centre it once it has a size.
  net.once("stabilizationIterationsDone", () => net.fit());
  net.on("resize", () => net.fit());
  // Clicking a paper opens it in Read. The map's frame may not move the page itself,
  // so it follows a link placed in the page around it.
  net.on("click", (e) => {{
    if (!e.nodes.length) return;
    const page = window.parent.document, a = page.createElement("a");
    a.href = "?paper=" + e.nodes[0];
    page.body.appendChild(a);
    a.click();
  }});
  net.on("hoverNode", () => document.body.style.cursor = "pointer");
  net.on("blurNode", () => document.body.style.cursor = "default");
</script>""", height=620)

        with st.expander("Edit links"):
            st.caption("Add a link the app missed, or remove one that is wrong. Added links are dashed.")
            by_label = {paper_label(p): p["id"] for p in papers}
            a = by_label[st.selectbox("This paper", list(by_label), key="link_a")]
            b = by_label[st.selectbox("cites this paper", list(by_label), key="link_b")]
            col_link, col_unlink = st.columns(2)
            if col_link.button("Link", disabled=a == b):
                db.set_link(a, b, True)
                st.rerun()
            if col_unlink.button("Unlink", disabled=a == b):
                db.set_link(a, b, False)
                db.set_link(b, a, False)
                st.rerun()
