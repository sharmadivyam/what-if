"""WHAT IF? — Streamlit frontend (Historical Premium aesthetic).

A warm, editorial, museum-quality UI for historical counterfactuals. Dark forest /
parchment themes, gold + olive + caramel accents, Playfair Display / Lora / Inter
type, a painting hero on the landing page, and a simulation-first results timeline.

This is a PURE VIEW over the pipeline — it changes only presentation:
- Calls ``pipeline.historios_pipeline.run(question, progress_callback=...)`` and uses
  the per-node hook to light up the loading stages live.
- Renders ``output.report_generator.generate_report(...)`` output — VERIFIED facts
  kept visually separate from SIMULATED, confidence-scored consequences (Rule #1).
- Handles loading / error / empty states gracefully — never crashes.

Run with:  ``streamlit run frontend/app.py``
       or:  ``D:\\historyos\\venv\\Scripts\\python.exe -m streamlit run frontend/app.py``
"""

from __future__ import annotations

import os

# Quiet transformers/torch noise (set BEFORE any transformers/torch import, which
# happens lazily inside the agents during a run, so these process-global filters
# apply). We suppress the SPECIFIC nuisance warnings only — real logs stay visible.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import warnings

# The "[transformers] Accessing `__path__` from .models.… Returning `__path__`
# instead." advisory comes from transformers' lazy-module shim via warnings.warn.
warnings.filterwarnings("ignore", message=r".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers.*")

# Show useful logs in the terminal (pipeline node timings, scoring summary). The
# pipeline's module loggers have no handler under `streamlit run`, so add one once.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
# …but silence the one chatty Streamlit watcher line ("Examining the path of
# torch.classes…") that would otherwise spam the console on every rerun.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import base64
import re
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from string import Template

# ``streamlit run`` puts frontend/ (not the project root) on sys.path, so absolute
# imports like ``from pipeline...`` would fail. Put the project root first.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from output.report_generator import generate_report  # noqa: E402
from pipeline.historios_pipeline import run as run_pipeline  # noqa: E402

# --- Constants ---------------------------------------------------------------

APP_TAGLINE = "Counterfactual History Engine"
GITHUB_URL = "https://github.com/"  # TODO: point at the real repository
DISCLAIMER = "Simulated consequences are AI-generated inferences, not historical fact."
LOADING_NOTE = "This takes ~2 minutes on the free tier — worth the wait."

ABOUT_TEXT = (
    "**WHAT IF?** answers historical *what-if* questions. For any counterfactual it "
    "retrieves **verified facts** from a cited corpus, **reasons** through the likely "
    "consequences in a short causal chain, and attaches a **confidence score** to "
    "every simulated claim — keeping what's verified strictly separate from what's "
    "simulated."
)

EXAMPLES = [
    "What if the Mughal Empire had industrialized before the British arrived?",
    "What if Britain had never colonized India?",
    "What if the Western Roman Empire had never fallen?",
    "What if Genghis Khan had died in childhood before unifying the Mongols?",
    "What if the Cuban Missile Crisis had escalated into nuclear war?",
    "What if the Ottoman Empire had survived past 1922?",
]

HOW_IT_WORKS = [
    ("Ⅰ", "Retrieves verified facts", "Searches a cited historical corpus for grounded context."),
    ("Ⅱ", "Reasons through consequences", "Builds a short causal chain of what might follow."),
    ("Ⅲ", "Scores every claim", "Each step is rated by how well evidence supports it."),
]

# Pipeline node → loading-stage label (one per node; ``score`` is the 5th/instant one).
STAGES = [
    ("understand_query", "Understanding your question"),
    ("retrieve", "Searching historical sources"),
    ("ground", "Extracting verified facts"),
    ("reason", "Reasoning through history"),
    ("score", "Scoring confidence"),
]
_STAGE_INDEX = {name: i for i, (name, _) in enumerate(STAGES)}

# Historical premium palette → theme tokens. Confidence colours (visual only;
# LEVEL semantics unchanged): HIGH=gold, MEDIUM=olive, LOW=caramel, SPECULATIVE=red.
THEMES = {
    "dark": {
        "bg": "#111A19", "surface": "#1B2625", "surface_alt": "#15201E",
        "text": "#F0E6C8", "muted": "#A89B7C", "border": "#2C3A33",
        "gold": "#E0B873", "gold_text": "#E0B873", "olive": "#4C6338",
        "caramel": "#BF8336", "spec": "#DA6A5F", "shadow": "rgba(0,0,0,0.5)",
        # Radial vignette wash over the page painting → frames the centred paper.
        "overlay": "radial-gradient(ellipse at 50% 28%, rgba(15,22,21,0.72), rgba(15,22,21,0.95))",
        "panel": "rgba(22,30,28,0.42)", "hair": "rgba(240,230,200,0.16)",
        "paper": "rgba(18,26,24,0.92)",
    },
    "light": {
        "bg": "#F5EDD6", "surface": "#FBF5E1", "surface_alt": "#EEE4C8",
        "text": "#1C1C1C", "muted": "#6B5E45", "border": "#D9C9A0",
        # pale gold is unreadable as text on parchment → deeper gold for text roles.
        "gold": "#E0B873", "gold_text": "#A9712B", "olive": "#4C6338",
        "caramel": "#BF8336", "spec": "#B23A30", "shadow": "rgba(60,40,10,0.12)",
        # Parchment vignette wash so the painting only sets the mood; dark text reads.
        "overlay": "radial-gradient(ellipse at 50% 28%, rgba(245,237,214,0.84), rgba(245,237,214,0.97))",
        "panel": "rgba(251,245,225,0.62)", "hair": "rgba(28,28,28,0.16)",
        "paper": "rgba(249,242,223,0.93)",
    },
}


# --- CSS ---------------------------------------------------------------------

_CSS = Template(
    "@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;0,800;1,600&family=Lora:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');"
    """
    /* ---- base: the saved battle painting as the full page background ---- */
    .stApp { background-color: $bg;
             background-image: $overlay, url("$bg_uri");
             background-size: cover; background-position: center; background-repeat: no-repeat;
             background-attachment: fixed;
             color: $text; font-family: 'Lora', Georgia, serif; overflow-x: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
    /* centred translucent 'paper' panel over the fixed painting (gallery placard) */
    .block-container { max-width: 860px; margin: 2.2rem auto 3rem;
        padding: clamp(1.3rem, 4vw, 2.6rem);
        background: $paper; border: 1px solid $hair; border-radius: 4px;
        box-shadow: 0 12px 48px rgba(0,0,0,0.32);
        backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px);
        animation: wf-fade .55s ease both; }
    @keyframes wf-fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    [data-testid="stSidebar"] { background: $surface_alt; border-right: 1px solid $border; }
    h1,h2,h3,h4 { font-family: 'Playfair Display', Georgia, serif; }

    /* ---- navbar ---- */
    .nav-logo { font-family: 'Playfair Display', serif; font-weight: 800; font-size: 1.4rem;
                letter-spacing: 0.01em; color: $gold_text; }
    .nav-tag { color: $muted; font-family: 'Inter', sans-serif; font-size: 0.7rem;
               letter-spacing: 0.16em; text-transform: uppercase; margin-top: -3px; }
    .navbar-rule { border-bottom: 1px solid $gold; opacity: 0.4; margin: 0.4rem 0 1.5rem; }

    /* ---- buttons / example chips: quiet gallery panels (gold only on hover) ---- */
    .stButton > button, .stLinkButton > a, .stFormSubmitButton button,
    div[data-testid="stPopover"] > button {
        border-radius: 0; border: 1px solid $hair; background: $panel;
        color: $text; font-family: 'Inter', sans-serif; font-weight: 500; white-space: normal;
        line-height: 1.45; box-shadow: none; transition: border-color .18s ease, color .18s ease;
    }
    .stButton > button:hover, .stLinkButton > a:hover, .stFormSubmitButton button:hover,
    div[data-testid="stPopover"] > button:hover {
        border-color: $gold; color: $gold_text; background: $panel;
    }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] .stButton > button { width: 100%; text-align: left; }
    [data-testid="stSidebar"] .stButton:first-of-type > button {
        border-color: $gold; color: $gold_text; font-weight: 600; background: rgba(224,184,115,0.10);
    }
    .sb-label { color: $muted; font-family: 'Inter', sans-serif; font-size: 0.68rem;
                letter-spacing: 0.18em; text-transform: uppercase; margin: 1.3rem 0 0.5rem; }

    /* ---- text input ---- */
    .stTextInput > div > div > input {
        background: $panel; color: $text; border: 1px solid $hair; border-radius: 0;
        font-family: 'Lora', serif; font-size: 1.08rem; padding: 0.95rem 1.1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: $gold; box-shadow: none;
    }
    .stTextInput > div > div > input::placeholder { color: $muted; opacity: 0.8; }
    .stForm { border: none; padding: 0; }

    /* ---- hero (landing): refined editorial heading over the page painting ---- */
    .hero { text-align: center; padding: 2.8rem 1rem 1.2rem; }
    .hero-inner { max-width: 760px; margin: 0 auto; }
    .hero-title { font-family: 'Playfair Display', serif; font-weight: 700; color: $text;
                  font-size: 2.5rem; line-height: 1.08; letter-spacing: 0.03em;
                  text-shadow: 0 1px 8px rgba(0,0,0,0.30); }
    .hero-rule { width: 56px; height: 1px; background: $gold; opacity: 0.75; margin: 1.1rem auto 1.2rem; }
    .hero-tag { font-family: 'Lora', serif; font-style: italic; color: $muted; font-size: 1.12rem;
                line-height: 1.6; }
    .hint { text-align:center; color: $muted; font-family:'Inter',sans-serif; font-size: 0.72rem;
            letter-spacing: 0.14em; text-transform: uppercase; margin: 1.2rem 0 0.9rem; }

    /* ---- how-it-works cards ---- */
    .hiw { display: flex; gap: 1.6rem; margin: 3rem 0 1rem; flex-wrap: wrap; }
    .hiw .card { flex: 1; min-width: 200px; background: transparent; border: none;
                 border-top: 1px solid $hair; border-radius: 0; padding: 1.2rem 0.1rem 0.4rem;
                 box-shadow: none; }
    .hiw .num { color: $gold_text; font-family: 'Playfair Display', serif; font-size: 1.5rem;
                font-weight: 700; opacity: 0.9; }
    .hiw .t { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.05rem;
              margin: 0.5rem 0 0.45rem; color: $text; }
    .hiw .d { font-family: 'Lora', serif; color: $muted; font-size: 0.92rem; line-height: 1.6; }

    /* ---- loading ---- */
    .load-wrap { max-width: 580px; margin: 2.4rem auto; }
    .stage { display: flex; align-items: center; gap: 0.8rem; font-family: 'Lora', serif;
             font-size: 1.05rem; padding: 0.5rem 0; color: $muted; }
    .stage .ic { width: 1.4rem; display: inline-block; text-align: center; font-size: 1.1rem; }
    .stage.done { color: $text; } .stage.done .ic { color: $gold; }
    .stage.current { color: $text; font-weight: 600; } .stage.current .ic { color: $gold; }
    .timer { text-align: center; color: $gold_text; font-family:'Inter',sans-serif; font-size: 0.9rem;
             margin-top: 1.1rem; letter-spacing: 0.05em; }
    .note { text-align: center; color: $muted; font-family:'Lora',serif; font-style: italic;
            font-size: 0.82rem; margin-top: 0.3rem; }

    /* ---- results ---- */
    .q-title { font-family: 'Playfair Display', serif; font-size: 2.0rem; font-weight: 700;
               color: $text; line-height: 1.18; margin: 0.4rem 0 0.7rem; }
    .sec-h { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.0rem;
             letter-spacing: 0.16em; text-transform: uppercase; color: $gold_text;
             display: inline-block; border-bottom: 1px solid $gold; padding-bottom: 6px;
             margin: 2.4rem 0 1.1rem; }

    /* ---- confidence pill badges ---- */
    .cbadge { display:inline-block; font-family:'Inter',sans-serif; font-weight:700; font-size:0.72rem;
              letter-spacing:0.06em; padding:0.2rem 0.7rem; border-radius:999px; white-space:nowrap; }
    .cbadge-high { background:$gold; color:#1C1C1C; }
    .cbadge-medium { background:$olive; color:#F0E6C8; }
    .cbadge-low { background:$caramel; color:#1C1C1C; }
    .cbadge-spec { background:$spec; color:#F0E6C8; }
    .cbadge-muted { background:$muted; color:$bg; }
    .bg-high{background:$gold;} .bg-medium{background:$olive;} .bg-low{background:$caramel;}
    .bg-spec{background:$spec;} .bg-muted{background:$muted;}

    /* ---- timeline ---- */
    .timeline { position: relative; padding-left: 30px; margin: 0.4rem 0; }
    .timeline::before { content:""; position:absolute; left:6px; top:10px; bottom:10px; width:2px;
                        background:$gold; opacity:0.4; }
    .tl-item { position: relative; margin-bottom: 18px; }
    .tl-dot { position:absolute; left:-30px; top:18px; width:14px; height:14px; border-radius:50%;
              border:3px solid $bg; }
    .tl-card { background:$surface; border:1px solid $border; border-left-width:5px; border-radius:7px;
               padding:15px 18px; box-shadow:0 1px 4px $shadow; }
    .bd-left-high{border-left-color:$gold;} .bd-left-medium{border-left-color:$olive;}
    .bd-left-low{border-left-color:$caramel;} .bd-left-spec{border-left-color:$spec;}
    .bd-left-muted{border-left-color:$muted;}
    .tl-top { display:flex; justify-content:space-between; align-items:center; gap:0.8rem; }
    .tl-horizon { font-family:'Inter',sans-serif; font-weight:600; font-size:0.78rem; color:$muted;
                  letter-spacing:0.08em; text-transform:uppercase; }
    .tl-body { font-family:'Lora',serif; font-size:1.05rem; line-height:1.6; color:$text;
               margin:0.7rem 0 0.8rem; }
    .tl-foot { font-family:'Inter',sans-serif; font-size:0.76rem; color:$muted; border-top:1px solid $border;
               padding-top:0.65rem; }
    .chip-ev { display:inline-block; font-family:'Inter',sans-serif; font-size:0.72rem; background:$surface_alt;
               border:1px solid $border; color:$muted; padding:0.05rem 0.45rem; border-radius:3px;
               margin:0 0.3rem 0.3rem 0; }
    .chip-bad { border-color:$spec; color:$spec; }

    /* ---- details / collapsibles ---- */
    details.box, details.evidence { border:1px solid $border; border-radius:6px; background:$surface;
        margin:0.6rem 0; padding:0 1.1rem; }
    details.evidence { border-color:$gold; }
    details > summary { cursor:pointer; list-style:none; padding:0.85rem 0; font-family:'Inter',sans-serif;
        font-weight:600; color:$gold_text; font-size:0.9rem; }
    details > summary::-webkit-details-marker { display:none; }
    details .inner { font-family:'Lora',serif; padding:0 0 1rem; color:$text; font-size:0.96rem; line-height:1.6; }
    details .inner p { margin: 0; }
    details .inner ol.points { margin: 0; padding-left: 1.35rem; }
    details .inner ol.points li { margin: 0.4rem 0; line-height: 1.6; }
    .ev-item { padding:0.55rem 0; border-top:1px solid $border; font-size:0.9rem; }
    .ev-claim { color:$text; } .ev-meta { color:$muted; font-family:'Inter',sans-serif; font-size:0.74rem; }
    .ev-meta a { color:$gold_text; text-decoration:none; } .ev-meta a:hover { text-decoration:underline; }

    /* ---- notices + disclaimer ---- */
    .notice { border:1px solid $border; border-left:4px solid $gold; border-radius:6px;
              padding:1rem 1.2rem; margin:1rem 0; background:$surface; font-family:'Lora',serif; }
    .notice.err { border-left-color:$spec; }
    .disclaimer { color:$muted; font-family:'Inter',sans-serif; font-size:0.74rem; border-top:1px solid $border;
                  margin-top:2.2rem; padding-top:0.9rem; font-style:italic; }
    """
)


@st.cache_data(show_spinner=False)
def _bg_data_uri() -> str:
    """Base64 data-URI for the page-background painting (frontend/1.jpg).

    Streamlit can't reference a local file path in CSS ``url()``, so the image is
    embedded as a data URI. Cached so the encode runs once. Returns "" if the file
    is missing — the overlay/base colour then shows on its own (graceful).
    """
    try:
        data = (Path(__file__).resolve().parent / "1.jpg").read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        return ""


def inject_css(theme: str) -> None:
    tokens = dict(THEMES.get(theme, THEMES["dark"]))
    tokens["bg_uri"] = _bg_data_uri()
    st.markdown(f"<style>{_CSS.substitute(**tokens)}</style>", unsafe_allow_html=True)


# --- Helpers -----------------------------------------------------------------


def _esc(text) -> str:
    return (str(text or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lvl_key(level: str | None) -> str:
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low", "SPECULATIVE": "spec"}.get(
        level or "", "muted"
    )


def _cbadge(level: str | None, prefix: str = "", none_text: str = "—") -> str:
    key = _lvl_key(level)
    text = f"{prefix}{level}" if level else f"{prefix}{none_text}"
    return f'<span class="cbadge cbadge-{key}">{_esc(text)}</span>'


# Numbered points like "1. … 2. … 3. …" (the trailing space guards against decimals
# such as "24.5%"); used to turn a run-on tail section into a real <ol>.
_NUM_POINT_RE = re.compile(r"(?:^|\s)\d+\.\s+(.*?)(?=\s+\d+\.\s+|\Z)", re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _richtext(s: str) -> str:
    """Escape, render **bold**, and flatten internal whitespace for inline HTML."""
    out = _BOLD_RE.sub(r"<strong>\1</strong>", _esc(s))
    return " ".join(out.split())


def _format_points(text: str) -> str:
    """Render numbered points (1. 2. 3.) as an ordered list; else a paragraph."""
    body = (text or "").strip()
    if not body:
        return ""
    items = [m.strip() for m in _NUM_POINT_RE.findall(body) if m.strip()]
    if len(items) >= 2:
        return '<ol class="points">' + "".join(f"<li>{_richtext(i)}</li>" for i in items) + "</ol>"
    return f"<p>{_richtext(body)}</p>"


def _strip_sim(text: str) -> str:
    stripped = (text or "").lstrip()
    if stripped[:11].upper() == "[SIMULATED]":
        return stripped[11:].lstrip()
    return text


def _go(question: str) -> None:
    st.session_state.pending = question


# --- Navbar ------------------------------------------------------------------


def render_navbar(theme: str) -> None:
    left, about, gh, toggle = st.columns([5, 1.1, 1.2, 0.9], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="nav-logo">WHAT IF?</div>'
            f'<div class="nav-tag">{APP_TAGLINE}</div>',
            unsafe_allow_html=True,
        )
    with about:
        with st.popover("About", use_container_width=True):
            st.markdown(ABOUT_TEXT)
    with gh:
        st.link_button("★ GitHub", GITHUB_URL, use_container_width=True)
    with toggle:
        icon = "🌙" if theme == "light" else "☀"
        if st.button(icon, key="theme_toggle", use_container_width=True, help="Toggle light/dark"):
            st.session_state.theme = "dark" if theme == "light" else "light"
            st.rerun()
    st.markdown('<div class="navbar-rule"></div>', unsafe_allow_html=True)


# --- Sidebar -----------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        if st.button("✦  New question", key="newq", use_container_width=True):
            st.session_state.result = None
            st.session_state.pop("pending", None)
            st.rerun()

        st.markdown('<div class="sb-label">Recent</div>', unsafe_allow_html=True)
        history = st.session_state.get("history", [])
        if not history:
            st.markdown(
                '<div style="color:#A89B7C;font-family:Lora,serif;font-size:0.82rem;font-style:italic;">'
                'No questions yet.</div>',
                unsafe_allow_html=True,
            )
        for i, item in enumerate(history[:3]):
            q = item["question"]
            short = q if len(q) <= 52 else q[:51] + "…"
            if st.button(short, key=f"hist_{i}", use_container_width=True):
                _go(q)
            level = item.get("overall_confidence")
            st.markdown(
                f'<div style="margin:-0.4rem 0 0.7rem 0.2rem;">'
                f'{_cbadge(level) if level else _cbadge(None, "")}</div>',
                unsafe_allow_html=True,
            )


# --- Search + landing --------------------------------------------------------


def render_search(key: str, placeholder: str) -> None:
    with st.form(key, clear_on_submit=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            value = st.text_input(
                "q", value="", placeholder=placeholder, label_visibility="collapsed", key=f"{key}_in"
            )
        with c2:
            submitted = st.form_submit_button("Ask →", use_container_width=True)
    if submitted and value.strip():
        _go(value.strip())


def render_landing() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-inner">'
        '<div class="hero-title">WHAT IF?</div>'
        '<div class="hero-rule"></div>'
        '<div class="hero-tag">Counterfactual history, grounded in verified sources.</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    render_search("q_hero", "What if the Mughal Empire never fell?")
    st.markdown('<div class="hint">Try an example</div>', unsafe_allow_html=True)

    # 6 example chips, 2 rows of 3 (rectangular, gold-bordered via the button CSS).
    for row in (EXAMPLES[:3], EXAMPLES[3:6]):
        cols = st.columns(3)
        for col, q in zip(cols, row):
            with col:
                if st.button(q, key=f"chip_{hash(q)}", use_container_width=True):
                    _go(q)

    cards = "".join(
        f'<div class="card"><div class="num">{n}</div><div class="t">{_esc(t)}</div>'
        f'<div class="d">{_esc(d)}</div></div>'
        for n, t, d in HOW_IT_WORKS
    )
    st.markdown(f'<div class="hiw">{cards}</div>', unsafe_allow_html=True)


# --- Loading (STATE 2) -------------------------------------------------------


def _loading_html(done_idx: int, elapsed: float) -> str:
    rows = []
    for i, (_, label) in enumerate(STAGES):
        if i < done_idx:
            cls, ic = "done", "●"
        elif i == done_idx:
            cls, ic = "current", "◐"
        else:
            cls, ic = "pending", "○"
        check = "  ✓" if i < done_idx else ""
        rows.append(f'<div class="stage {cls}"><span class="ic">{ic}</span>{label}{check}</div>')
    mm, ss = divmod(int(elapsed), 60)
    return (
        f'<div class="load-wrap">{"".join(rows)}'
        f'<div class="timer">elapsed {mm:02d}:{ss:02d}</div>'
        f'<div class="note">{LOADING_NOTE}</div></div>'
    )


def _start_job(question: str) -> None:
    """Kick off the pipeline in a worker thread, tracked in ``st.session_state``.

    The thread + queue live in session_state so they SURVIVE full-app reruns (a
    theme toggle, a resize, any widget interaction). The pipeline call itself is
    unchanged — exactly one ``run_pipeline(question, progress_callback=cb)``.
    """
    events: Queue = Queue()
    holder: dict = {}

    def cb(name: str, elapsed: float, errored: bool) -> None:
        events.put(name)

    def worker() -> None:
        try:
            holder["state"] = run_pipeline(question, progress_callback=cb)
        except Exception as exc:  # noqa: BLE001 — surface as a graceful error state
            holder["state"] = {
                "question": question, "status": "error", "timings": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            events.put("__done__")

    thread = threading.Thread(target=worker, daemon=True)
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx

        add_script_run_ctx(thread)
    except Exception:
        pass
    thread.start()

    st.session_state["job"] = {
        "question": question, "events": events, "holder": holder,
        "thread": thread, "done_idx": 0, "start": time.monotonic(),
    }


def _drain_events(job: dict) -> bool:
    """Consume any queued stage events (non-blocking); return True when finished."""
    finished = False
    while True:
        try:
            evt = job["events"].get_nowait()
        except Empty:
            break
        if evt == "__done__":
            finished = True
        else:
            idx = _STAGE_INDEX.get(evt)
            if idx is not None:
                job["done_idx"] = max(job["done_idx"], idx + 1)
    return finished


@st.fragment(run_every="0.5s")
def render_loading() -> None:
    """Poll the active job every 0.5s as an isolated fragment (no blocking loop).

    Because the job is in session_state and this fragment reruns on its OWN timer,
    a full-app rerun (e.g. the theme toggle) no longer discards the in-flight run —
    the fragment simply re-attaches and keeps polling; the worker thread is
    untouched. When the run finishes we commit the result and trigger a full rerun
    to leave the loading state.
    """
    job = st.session_state.get("job")
    if not job:
        return
    finished = _drain_events(job)
    st.markdown(
        _loading_html(job["done_idx"], time.monotonic() - job["start"]),
        unsafe_allow_html=True,
    )
    if finished:
        state = job["holder"].get(
            "state",
            {"question": job["question"], "status": "error", "timings": {},
             "error": "RuntimeError: pipeline returned no state"},
        )
        st.session_state["result"] = (job["question"], state)
        _record_history(job["question"], state)
        del st.session_state["job"]
        st.rerun()  # full app rerun → leave loading, render results


# --- Results (STATE 3) — simulation first ------------------------------------


def render_results(question: str, state: dict, theme: str) -> None:
    status = state.get("status", "ok")
    grounded = state.get("grounded")
    report = generate_report(
        scored=state.get("scored"), grounded=grounded,
        error=state.get("error"), status=status, timings=state.get("timings"),
    )
    source_map = dict(getattr(grounded, "source_map", {}) or {}) if grounded else {}

    st.markdown(f'<div class="q-title">{_esc(question)}</div>', unsafe_allow_html=True)

    if status == "error":
        st.markdown(
            f'<div class="notice err"><b class="cbadge cbadge-spec">PIPELINE ERROR</b><br><br>'
            f'<span style="color:#A89B7C;">Failed at: {_esc(state.get("failed_node") or "—")}</span><br>'
            f'<code>{_esc(state.get("error") or "unknown error")}</code><br><br>'
            f'This is a graceful failure — no simulated content is shown.</div>',
            unsafe_allow_html=True,
        )
        _render_disclaimer()
        return
    if status == "no_context":
        st.markdown(
            '<div class="notice"><b class="cbadge cbadge-medium">NO SOURCES</b><br><br>'
            'No verified sources were found for this question. Try one of the example '
            'questions — they target the ingested topics.</div>',
            unsafe_allow_html=True,
        )
        _render_disclaimer()
        return

    # ---- SECTION 1 — THE ANSWER (simulation first) -------------------------
    st.markdown(
        f'<div style="margin:0.2rem 0 0.4rem;">'
        f'{_cbadge(report.overall_confidence, "OVERALL · ", none_text="UNAVAILABLE")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-h">What might have happened</div>', unsafe_allow_html=True)
    _render_timeline(report.simulated_steps)
    _render_collapsibles(report)

    # ---- SECTION 2 — THE EVIDENCE (secondary, collapsed) -------------------
    st.markdown('<div class="sec-h">The evidence</div>', unsafe_allow_html=True)
    _render_evidence(report.verified_facts, source_map)

    _render_disclaimer()


def _render_timeline(steps) -> None:
    if not steps:
        st.markdown(
            '<div class="notice">No simulated consequences were produced.</div>',
            unsafe_allow_html=True,
        )
        return
    items = []
    for s in steps:
        key = _lvl_key(s.confidence_level)
        chips = "".join(
            f'<span class="chip-ev">{_esc(c)}</span>' for c in s.evidence_chunk_ids
        ) or '<span class="chip-ev">ungrounded</span>'
        for bad in s.unknown_evidence_ids:
            chips += f'<span class="chip-ev chip-bad">⚠ {_esc(bad)}</span>'
        items.append(
            f'<div class="tl-item"><span class="tl-dot bg-{key}"></span>'
            f'<div class="tl-card bd-left-{key}">'
            f'<div class="tl-top"><span class="tl-horizon">{_esc(s.time_horizon)}</span>'
            f'{_cbadge(s.confidence_level)}</div>'
            f'<div class="tl-body">{_esc(_strip_sim(s.consequence))}</div>'
            f'<div class="tl-foot"><b>Based on:</b> {chips}<br>'
            f'<span style="opacity:0.9;">{_esc(s.confidence_explanation)}</span></div>'
            f'</div></div>'
        )
    st.markdown('<div class="timeline">' + "".join(items) + "</div>", unsafe_allow_html=True)


def _render_collapsibles(report) -> None:
    for title, body in (
        ("What remains unknowable", report.what_remains_unknowable),
        ("Historian's note", report.historians_note),
    ):
        if body and body.strip():
            st.markdown(
                f'<details class="box"><summary>↓ {title}</summary>'
                f'<div class="inner">{_format_points(body)}</div></details>',
                unsafe_allow_html=True,
            )


def _render_evidence(facts, source_map: dict) -> None:
    n = len(facts)
    n_sources = len({f.source_title for f in facts})
    if not facts:
        st.markdown(
            '<div class="notice">No verified facts were retrieved — treat the simulation '
            'above as especially speculative.</div>',
            unsafe_allow_html=True,
        )
        return
    rows = []
    for f in facts:
        url = source_map.get(f.source_chunk_id, "")
        link = f' · <a href="{_esc(url)}" target="_blank">source ↗</a>' if url else ""
        rows.append(
            f'<div class="ev-item"><span class="ev-claim">{_esc(f.claim)}</span><br>'
            f'<span class="ev-meta">[{_esc(f.source_chunk_id)}] · {_esc(f.source_title)}{link}</span></div>'
        )
    st.markdown(
        f'<details class="evidence"><summary>📜 View {n} verified fact{"s" if n != 1 else ""} '
        f'from {n_sources} source{"s" if n_sources != 1 else ""} (click to expand)</summary>'
        f'<div class="inner">{"".join(rows)}</div></details>',
        unsafe_allow_html=True,
    )


def _render_disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">⚠ {DISCLAIMER}</div>', unsafe_allow_html=True)


# --- History bookkeeping -----------------------------------------------------


def _record_history(question: str, state: dict) -> None:
    scored = state.get("scored")
    overall = getattr(scored, "overall_confidence", None) if scored else None
    entry = {"question": question, "status": state.get("status", "ok"), "overall_confidence": overall}
    hist = [h for h in st.session_state.get("history", []) if h["question"] != question]
    hist.insert(0, entry)
    st.session_state["history"] = hist[:10]


# --- Main --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="WHAT IF?", page_icon="❓", layout="wide",
                       initial_sidebar_state="expanded")
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("result", None)

    theme = st.session_state["theme"]
    inject_css(theme)
    render_navbar(theme)
    render_sidebar()

    pending = st.session_state.pop("pending", None)
    # Start a new run only if one isn't already in flight (ignore submits while busy).
    if pending and "job" not in st.session_state:
        _start_job(pending)

    if "job" in st.session_state:
        # STATE 2 — loading. Search stays visible; the fragment polls on its own
        # timer so a theme toggle (full rerun) cannot discard the in-flight run.
        render_search("q_top", "Ask another counterfactual…")
        render_loading()
    elif st.session_state.get("result") is not None:
        render_search("q_top", "Ask another counterfactual…")
        q, state = st.session_state["result"]
        render_results(q, state, theme)  # STATE 3 — results
    else:
        render_landing()  # STATE 1 — landing


# ``streamlit run`` imports the module (no __main__), so render on import.
main()
