"""Streamlit chat UI for DocuMind.

Run with:  streamlit run src/documind/app.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

# Allow `streamlit run src/documind/app.py` to work without `pip install -e .`
# by ensuring the project's `src` directory is importable.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from documind import history, summary, vectorstore
from documind.config import get_settings
from documind.ingestion import document_stats, normalize_name, process_uploaded_file
from documind.pipeline import answer
from documind.reranker import RankedChunk

settings = get_settings()

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

USER_AVATAR = "🧑"
BOT_AVATAR = "📄"

SUGGESTED_QUESTIONS = [
    "Summarize this document in a few sentences.",
    "What are the key points?",
    "Are there any important dates or numbers?",
    "What conclusions does it reach?",
]

# --------------------------------------------------------------------------- #
# Styling — warm toasted-tan, minimalist top bar
# --------------------------------------------------------------------------- #
CSS = """
<style>
/* Two CSS palettes live in :root.light and :root.dark. The page <html> gets
   the matching class via a small inline script (see _render_theme) so the
   user can flip themes without a full Streamlit rerun. */
:root.light {
  --dm-bg: #CDBC97;
  --dm-card: #FBF7EC;
  --dm-card-2: #EFE7D6;
  --dm-border: rgba(70,52,28,0.22);
  --dm-accent: #8A5A33;
  --dm-accent-2: #B07F4A;
  --dm-text: #2A2620;
  --dm-muted: #6B5F49;
  --dm-bar-grad: linear-gradient(180deg, #FBF6EA 0%, #F3EBD9 100%);
  --dm-bar-shadow: 0 2px 10px rgba(120,90,50,0.06);
  --dm-bg-veil: rgba(255,255,255,0.18);
  --dm-bubble-soft: var(--dm-card);
}
:root.dark {
  /* Sporty deep-navy palette: cool, elevated, "garage at night". */
  --dm-bg: #050A18;
  --dm-card: #0E1A33;
  --dm-card-2: #142447;
  --dm-border: rgba(120,170,255,0.18);
  --dm-accent: #4FA3FF;
  --dm-accent-2: #2C7BE0;
  --dm-text: #E6EEF8;
  --dm-muted: #8FA6C4;
  --dm-bar-grad: linear-gradient(180deg, #0B142A 0%, #0A1A36 100%);
  --dm-bar-shadow: 0 4px 24px rgba(0,16,48,0.55);
  --dm-bg-veil: rgba(0,8,24,0.55);
  --dm-bubble-soft: #0F1E3D;
}

/* Generous top padding so the brand bar clears Streamlit's fixed header. */
.block-container { padding-top: 3.25rem; max-width: 1040px; position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* Minimalist top bar */
.dm-bar {
  display:flex; align-items:center; gap:10px;
  background: var(--dm-bar-grad);
  border:1px solid var(--dm-border); border-radius:14px;
  padding:10px 16px; margin-bottom:14px;
  box-shadow: var(--dm-bar-shadow);
  animation: dmFade .4s ease both;
}
@keyframes dmFade { from {opacity:0; transform:translateY(-6px);} to {opacity:1; transform:none;} }
.dm-bar .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
  font-size:1.05rem; background:linear-gradient(135deg,var(--dm-accent),var(--dm-accent-2)); }
.dm-bar .name { font-weight:800; font-size:1.12rem; color:var(--dm-text); letter-spacing:-0.01em; }
.dm-bar .tag { color:var(--dm-muted); font-size:0.78rem; margin-left:2px; }

/* Theme pill on the right of the brand bar (mirrors the toggle button). */
.dm-theme {
  margin-left: auto;
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.78rem; color: var(--dm-muted);
  padding: 4px 10px; border-radius: 999px;
  border: 1px solid var(--dm-border);
  background: var(--dm-card-2);
}
.dm-theme:hover { color: var(--dm-accent); }

/* ---------- Elevated sporty-blue background (page-wide) ---------- */
/* Built from three layers, top to bottom:
   1) a soft blue radial glow at top-centre (the "elevated" highlight),
   2) a second blue radial in the lower-right (ties the composition),
   3) a deep navy-to-black linear gradient (the "garage" base). */
.stApp {
  background:
    radial-gradient(1200px 600px at 50% 0%, rgba(79,163,255,0.30) 0%, rgba(79,163,255,0) 60%),
    radial-gradient(900px 500px at 85% 90%, rgba(44,123,224,0.22) 0%, rgba(44,123,224,0) 65%),
    linear-gradient(180deg, #0A1A36 0%, #050A18 60%, #02060F 100%) !important;
}
/* In light mode the same shapes dial back to a subtle warm glow so the
   toasted-tan page still feels like the original. */
:root.light .stApp {
  background:
    radial-gradient(1100px 540px at 50% -10%, rgba(79,163,255,0.20) 0%, rgba(79,163,255,0) 60%),
    radial-gradient(900px 500px at 90% 95%, rgba(44,123,224,0.14) 0%, rgba(44,123,224,0) 70%),
    linear-gradient(180deg, #CDBC97 0%, #C2B186 100%) !important;
}

/* Veil dims the lighting so foreground stays readable. Stronger in dark
   mode so the navy "garage" feels moody. */
.stApp::before {
  content: "";
  position: fixed; inset: 0;
  background: var(--dm-bg-veil);
  pointer-events: none;
  z-index: 0;
  transition: background .35s ease;
}

/* Streamlit's own containers stay transparent so the painted background
   shines through. */
.main, [data-testid="stAppViewContainer"] { background: transparent !important; }

/* Slide-down document-history list (rendered inside a popover) */
.dm-drop-head { font-weight:800; color:var(--dm-text); margin:0 0 2px 0; font-size:0.95rem; }
.dm-drop-sub { color:var(--dm-muted); font-size:0.78rem; margin:0 0 8px 0; }
.dm-doc-meta { color:var(--dm-muted); font-size:0.72rem; margin:-6px 0 8px 6px; }

/* Stat chips */
.dm-stats { display:flex; gap:10px; margin: 2px 0 14px 0; flex-wrap:wrap; }
.dm-chip { background:var(--dm-card); border:1px solid var(--dm-border); border-radius:12px;
  padding:8px 14px; min-width:108px; }
.dm-chip .n { font-size:1.35rem; font-weight:800; color:var(--dm-accent); }
.dm-chip .l { font-size:0.72rem; color:var(--dm-muted); text-transform:uppercase; letter-spacing:0.08em; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
  background: var(--dm-bubble-soft); border:1px solid var(--dm-border);
  border-radius:16px; padding:6px 12px; margin-bottom:6px;
  color: var(--dm-text);
}

/* History detail card */
.dm-hist { background:var(--dm-card); border:1px solid var(--dm-border); border-radius:16px;
  padding:18px 20px; margin-bottom:14px; }
.dm-hist .q { font-weight:700; font-size:1.05rem; margin-bottom:6px; color:var(--dm-text); }
.dm-hist .t { color:var(--dm-muted); font-size:0.78rem; }

/* Minimalist empty state */
.dm-empty { text-align:center; padding:48px 20px 24px; color:var(--dm-muted);
  animation: dmFade .5s ease both; }
.dm-empty .big { font-size:3rem; }
.dm-empty h2 { color:var(--dm-text); font-weight:800; margin:8px 0 6px; }
.dm-empty p { max-width:460px; margin:0 auto; font-size:0.95rem; }

/* Section heading */
.dm-suggest-label { color:var(--dm-muted); font-size:0.85rem; margin: 6px 0 2px 0; }

/* Highlight the (+) attach control built into the chat-input toolbar. */
[data-testid="stChatInput"] { border:1px solid var(--dm-border); background: var(--dm-card); }
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInputFileUploadButton"] { color: var(--dm-accent); }
[data-testid="stChatInputFileUploadButton"]:hover { color: var(--dm-accent-2); }

/* Theme-aware tweaks so secondary controls don't look out of place in dark. */
:root.dark  .stButton > button[kind="secondary"] { background: #10204A; color: var(--dm-text); border-color: var(--dm-border); }
:root.dark  .stButton > button[kind="secondary"]:hover { background: #163066; }
:root.dark  .stButton > button[kind="primary"] { background: var(--dm-accent); color: #fff; }
:root.dark  input, :root.dark textarea { background: #0B1733 !important; color: var(--dm-text) !important; }
:root.dark  [data-testid="stChatInput"] { background: #0B1733 !important; border-color: var(--dm-border) !important; color: var(--dm-text) !important; }
:root.dark  hr { border-color: var(--dm-border); }
:root.light .stButton > button[kind="secondary"] { background: var(--dm-card); }

/* Source citation cards */
.dm-src-head { font-weight:700; color:var(--dm-text); font-size:0.92rem; }
.dm-src-rel { color:var(--dm-accent); font-weight:700; font-size:0.8rem; }
</style>
"""

# Session state -------------------------------------------------------------- #
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_history_id", None)
st.session_state.setdefault("active_doc", None)
st.session_state.setdefault("pending_prompt", None)
# Theme toggle: light (default) vs. dark. Dark swaps the CSS palette to a deep
# navy/sporty-blue scheme and switches the background to an elevated dimmed
# blue scene.
st.session_state.setdefault("dark_mode", False)

# Apply the theme class to <html> on first render and whenever the toggle
# changes. This keeps the palette switch instant without re-running Streamlit.
_theme = "dark" if st.session_state.dark_mode else "light"
# st.html sanitizes HTML by default and drops scripts. We need the inline
# classList code to actually run, so opt into unsafe_allow_javascript. The
# payload is a hard-coded string (no user input), so the usual safety
# concerns don't apply here.
st.html(f"""
    <script>
      (function() {{
        try {{
          var root = window.parent.document.documentElement;
          root.classList.remove('light', 'dark');
          root.classList.add('{_theme}');
        }} catch (e) {{}}
      }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ingest(files) -> list[str]:
    """Process each file with live progress and an auto-generated summary."""
    processed: list[str] = []
    for f in files:
        with st.status(f"Processing {f.name}…", expanded=True) as status:
            st.write("📖 Reading the PDF…")
            chunks = process_uploaded_file(f)
            name = chunks[0].metadata["source"] if chunks else normalize_name(f.name)

            stats = document_stats(chunks)
            st.write(
                f"✂️ Split into **{stats['chunks']}** chunks "
                f"(~{stats['words']:,} words across {stats['pages']} page(s))."
            )

            st.write("🧠 Creating embeddings & indexing…")
            # Large PDFs embed thousands of chunks; show batch progress so the
            # step doesn't look frozen while Ollama works through them.
            embed_bar = st.progress(0.0, text="Embedding chunks…")

            # ``bar`` is bound as a default so the closure captures *this*
            # iteration's progress widget (ruff B023), not the loop variable.
            def _on_progress(done: int, total: int, bar=embed_bar) -> None:
                bar.progress(done / total, text=f"Embedded {done:,}/{total:,} chunks")

            vectorstore.add_documents(chunks, source_name=name, progress=_on_progress)
            embed_bar.empty()

            st.write("📝 Summarizing the document…")
            box = st.empty()
            acc: list[str] = []
            try:
                for tok in summary.stream_summary(name, chunks):
                    acc.append(tok)
                    box.markdown("".join(acc))
                md = "".join(acc).strip()
            except Exception as exc:  # Ollama unavailable etc.
                md = ""
                st.write(f"⚠️ Could not summarize automatically ({exc}).")
            summary.save(name, md or "_Summary unavailable._", stats)

            status.update(label=f"✓ {f.name} is ready", state="complete", expanded=False)
        st.toast(f"{f.name} processed", icon="✅")
        processed.append(name)
    return processed


def _summary_message(doc: str, summ) -> str:
    return (
        f"👋 Here's a quick briefing on **{doc}**:\n\n"
        f"{summ.markdown}\n\n"
        "Ask me anything about it — I'll answer from this document."
    )


def _relevance_pct(score: float) -> int:
    """Map a cross-encoder logit to a friendly 0–100% relevance.

    The re-ranker emits unbounded logits (roughly -11…+11), which read poorly as
    a raw "relevance 3.42". A logistic squash turns them into an intuitive
    percentage for the citation cards.
    """
    return round(100 / (1 + math.exp(-score)))


def _snippet(text: str, limit: int = 480) -> str:
    """Trim a chunk to ``limit`` chars on a word boundary so it doesn't cut mid-word."""
    text = " ".join(text.split())  # collapse stray whitespace for a tidy preview
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def _render_sources(chunks) -> None:
    if not chunks:
        return
    with st.expander(f"📎 Sources ({len(chunks)})"):
        for i, chunk in enumerate(chunks, start=1):
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page")
            page_str = f" · p.{page + 1}" if isinstance(page, int) else ""
            st.markdown(
                f'<span class="dm-src-head">{i}. {src}{page_str}</span> &nbsp; '
                f'<span class="dm-src-rel">{_relevance_pct(chunk.score)}% match</span>',
                unsafe_allow_html=True,
            )
            st.caption(_snippet(chunk.text))


def _fmt_ts(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso


def _chunks_from_sources(sources: list[dict]) -> list[RankedChunk]:
    """Rebuild RankedChunk objects from stored history sources for uniform rendering."""
    return [
        RankedChunk(
            text=s.get("snippet", ""),
            metadata={"source": s.get("source", "unknown"), "page": s.get("page")},
            score=float(s.get("score", 0.0)),
        )
        for s in sources
    ]


def _messages_from_history(entries) -> list[dict]:
    """Turn saved history entries (oldest→newest) into chat messages."""
    messages: list[dict] = []
    for e in sorted(entries, key=lambda x: x.timestamp):
        messages.append({"role": "user", "content": e.question})
        messages.append(
            {
                "role": "assistant",
                "content": e.answer,
                "sources": _chunks_from_sources(e.sources),
            }
        )
    return messages


def _open_document(doc: str, hist) -> None:
    """Scope the session to ``doc`` and restore its related chat history.

    If there's no prior conversation for this document, seed the chat with its
    auto-generated summary so it opens with a useful briefing.
    """
    st.session_state.active_doc = doc
    st.session_state.selected_history_id = None
    related = [e for e in hist if doc in e.documents]
    if related:
        st.session_state.messages = _messages_from_history(related)
    else:
        summ = summary.get(doc)
        st.session_state.messages = (
            [{"role": "assistant", "content": _summary_message(doc, summ)}]
            if summ and summ.markdown
            else []
        )
    st.rerun()


def _upload_history(sources: list[str]) -> list[tuple[str, str]]:
    """Indexed documents paired with their upload time, newest first.

    Falls back to an empty timestamp for docs without a saved summary so they
    still appear in the list (ordered last).
    """
    items: list[tuple[str, str]] = []
    for s in sources:
        summ = summary.get(s)
        items.append((s, summ.timestamp if summ else ""))
    items.sort(key=lambda it: it[1], reverse=True)
    return items


def _empty_state() -> None:
    st.markdown(
        """
        <div class="dm-empty">
          <div class="big">📄</div>
          <h2>Ask your documents anything.</h2>
          <p>Open <b>📤 Upload</b> in the bar above, add a PDF, and DocuMind will index it
          locally and answer your questions with citations — nothing leaves this machine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_doc_stats(doc: str) -> None:
    summ = summary.get(doc)
    stats = summ.stats if summ else {}
    if not stats:
        return
    st.markdown(
        f"""
        <div class="dm-stats">
          <div class="dm-chip"><div class="n">{stats.get('pages', 0)}</div><div class="l">Pages</div></div>
          <div class="dm-chip"><div class="n">{stats.get('words', 0):,}</div><div class="l">Words</div></div>
          <div class="dm-chip"><div class="n">{stats.get('chunks', 0)}</div><div class="l">Chunks</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stats(sources, hist) -> None:
    questions = len(hist)
    answered_docs = len({d for e in hist for d in e.documents})
    st.markdown(
        f"""
        <div class="dm-stats">
          <div class="dm-chip"><div class="n">{len(sources)}</div><div class="l">Documents</div></div>
          <div class="dm-chip"><div class="n">{questions}</div><div class="l">Questions asked</div></div>
          <div class="dm-chip"><div class="n">{answered_docs}</div><div class="l">Docs referenced</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_history_detail(entry) -> None:
    st.markdown(
        f"""<div class="dm-hist"><div class="t">🕘 {_fmt_ts(entry.timestamp)}</div>
        <div class="q">{entry.question}</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(entry.answer)
    if entry.sources:
        with st.expander(f"📎 Sources ({len(entry.sources)})"):
            for i, s in enumerate(entry.sources, start=1):
                page = s.get("page")
                page_str = f" · p.{page + 1}" if isinstance(page, int) else ""
                st.markdown(
                    f'<span class="dm-src-head">{i}. {s.get("source","unknown")}{page_str}</span> '
                    f'&nbsp; <span class="dm-src-rel">{_relevance_pct(float(s.get("score", 0)))}% '
                    f"match</span>",
                    unsafe_allow_html=True,
                )
                st.caption(_snippet(s.get("snippet", "")))
    if st.button("← Back to chat"):
        st.session_state.selected_history_id = None
        st.rerun()


def _suggestion_chips() -> None:
    """Clickable example questions shown when a chat is empty."""
    st.markdown('<p class="dm-suggest-label">✨ Try asking…</p>', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(cols, SUGGESTED_QUESTIONS, strict=False):
        with col:
            if st.button(q, key=f"sugg_{q}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()


# --------------------------------------------------------------------------- #
# Minimalist top bar
# --------------------------------------------------------------------------- #
def _upload_popover() -> None:
    """Slide-down panel for adding new PDFs."""
    with st.popover("📤 Upload", use_container_width=True):
        st.markdown('<p class="dm-drop-head">Add documents</p>', unsafe_allow_html=True)
        st.caption(f"PDF only · up to **{settings.max_upload_mb} MB** per file.")
        uploaded = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button(
            "Process documents", type="primary", use_container_width=True
        ):
            processed = _ingest(uploaded)
            if processed:
                # Jump straight into the freshly processed document's chat.
                _open_document(processed[-1], history.load())
            st.rerun()


def _documents_popover(sources, hist) -> None:
    """Slide-down list of the history of uploaded documents (newest first)."""
    with st.popover(f"📚 Documents ({len(sources)})", use_container_width=True):
        st.markdown(
            '<p class="dm-drop-head">Uploaded documents</p>'
            '<p class="dm-drop-sub">Most recent first — click one to open its chat.</p>',
            unsafe_allow_html=True,
        )
        if not sources:
            st.caption("Nothing indexed yet. Use 📤 Upload to add a PDF.")
            return
        for doc, ts in _upload_history(sources):
            active = st.session_state.active_doc == doc
            if st.button(
                f"{'📂' if active else '📄'} {doc}",
                key=f"doc_{doc}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                _open_document(doc, hist)
            meta = f"🕒 {_fmt_ts(ts)}" if ts else "🕒 upload time unknown"
            st.markdown(f'<div class="dm-doc-meta">{meta}</div>', unsafe_allow_html=True)


def _history_popover(hist) -> None:
    """Slide-down list of past questions."""
    with st.popover("🕘 History", use_container_width=True):
        st.markdown('<p class="dm-drop-head">Recent questions</p>', unsafe_allow_html=True)
        if not hist:
            st.caption("Your past questions will appear here.")
            return
        for entry in hist[:10]:
            label = (entry.question[:42] + "…") if len(entry.question) > 42 else entry.question
            if st.button(f"🔹 {label}", key=f"hist_{entry.id}", use_container_width=True):
                st.session_state.selected_history_id = entry.id
                st.rerun()


def _actions_popover() -> None:
    """Slide-down panel for chat/data actions."""
    with st.popover("⋯", use_container_width=True):
        if st.button("🧹 New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_history_id = None
            st.session_state.active_doc = None
            st.rerun()
        if st.button("🗑️ Clear all", use_container_width=True):
            vectorstore.reset()
            history.clear()
            summary.clear()
            st.session_state.messages = []
            st.session_state.selected_history_id = None
            st.session_state.active_doc = None
            st.rerun()


def _topbar(sources, hist) -> None:
    is_dark = st.session_state.dark_mode
    # The chip on the right of the bar doubles as a quick visual cue
    # (moon = dark, sun = light) so the user can tell which mode is active.
    theme_label = "🌙 Dark" if not is_dark else "☀️ Light"
    st.markdown(
        f"""<div class="dm-bar"><div class="logo">📄</div>
        <div class="name">DocuMind</div>
        <div class="tag">· private PDF research, on your machine</div>
        <div class="dm-theme" id="dm-theme-pill">{theme_label}</div></div>""",
        unsafe_allow_html=True,
    )
    # Brand sits in its own row above; the controls line up beneath it.
    # A leading 0.8-wide column holds the actual theme toggle button so the
    # visual chip and the clickable control stay aligned.
    c0, c1, c2, c3, c4 = st.columns([0.8, 1.2, 1.6, 1.2, 0.6])
    with c0:
        if st.button(theme_label, key="dm_theme_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with c1:
        _upload_popover()
    with c2:
        _documents_popover(sources, hist)
    with c3:
        _history_popover(hist)
    with c4:
        _actions_popover()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    sources = vectorstore.list_sources()
    hist = history.load()

    _topbar(sources, hist)

    # History detail view takes over the main pane when an entry is selected.
    if st.session_state.selected_history_id:
        entry = next(
            (e for e in hist if e.id == st.session_state.selected_history_id), None
        )
        if entry:
            st.title("History")
            _render_history_detail(entry)
            return
        st.session_state.selected_history_id = None

    if not sources:
        _empty_state()
        return

    # A scoped document may have been removed since it was opened.
    if st.session_state.active_doc and st.session_state.active_doc not in sources:
        st.session_state.active_doc = None
    active_doc = st.session_state.active_doc

    if active_doc:
        st.title(f"📂 {active_doc}")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption("Questions are scoped to this document and its saved history.")
        with c2:
            if st.button("✕ Exit document", use_container_width=True):
                st.session_state.active_doc = None
                st.session_state.messages = []
                st.rerun()
        _render_doc_stats(active_doc)
    else:
        st.title("Chat with your documents")
        _render_stats(sources, hist)

    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # Example-question chips when the conversation is empty.
    if not st.session_state.messages:
        _suggestion_chips()

    # Composer: a single chat-input toolbar whose built-in (+) attaches PDFs and
    # whose text box asks questions. Attaching a PDF here runs it through the same
    # ingest pipeline as the top-bar uploader.
    placeholder = (
        f"Ask about {active_doc}…" if active_doc else "Ask a question about your documents…"
    )
    submission = st.chat_input(
        placeholder, accept_file="multiple", file_type=["pdf"]
    )

    pending = st.session_state.pop("pending_prompt", None)
    prompt = pending
    files = []
    if submission:
        # With accept_file set, chat_input returns an object carrying both the
        # typed text and any attached files (either may be empty).
        prompt = (submission.text or "").strip() or pending
        files = submission.files or []

    if files:
        processed = _ingest(files)
        if processed and not prompt:
            # Attached a PDF with no question → open its chat with the summary.
            _open_document(processed[-1], history.load())

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    conv = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        stream, retrieval = answer(prompt, history=conv, source=active_doc)
        if retrieval.is_empty:
            reply = "I couldn't find anything relevant in the indexed documents."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            history.add(prompt, reply, chunks=[])
            return
        reply = st.write_stream(stream)
        _render_sources(retrieval.chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": retrieval.chunks}
    )
    history.add(prompt, reply, chunks=retrieval.chunks)


if __name__ == "__main__":
    main()
