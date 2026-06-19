"""Streamlit chat UI for DocuMind.

Run with:  streamlit run src/documind/app.py
"""

from __future__ import annotations

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
# Styling — warm beige, minimalist top bar
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root {
  --dm-bg: #F4EEE0;
  --dm-card: #FBF7EC;
  --dm-card-2: #EFE7D6;
  --dm-border: rgba(90,70,40,0.18);
  --dm-accent: #9C6B3F;
  --dm-accent-2: #C2925A;
  --dm-text: #2E2A22;
  --dm-muted: #7A6F5C;
}
.block-container { padding-top: 1rem; max-width: 1040px; }

/* Hide the default sidebar/collapse chrome — the top bar replaces it */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* Minimalist top bar */
.dm-bar {
  display:flex; align-items:center; gap:10px;
  background: linear-gradient(180deg, #FBF6EA 0%, #F3EBD9 100%);
  border:1px solid var(--dm-border); border-radius:14px;
  padding:10px 16px; margin-bottom:14px;
  box-shadow:0 2px 10px rgba(120,90,50,0.06);
  animation: dmFade .4s ease both;
}
@keyframes dmFade { from {opacity:0; transform:translateY(-6px);} to {opacity:1; transform:none;} }
.dm-bar .logo { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
  font-size:1.05rem; background:linear-gradient(135deg,var(--dm-accent),var(--dm-accent-2)); }
.dm-bar .name { font-weight:800; font-size:1.12rem; color:var(--dm-text); letter-spacing:-0.01em; }
.dm-bar .tag { color:var(--dm-muted); font-size:0.78rem; margin-left:2px; }

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
  background: var(--dm-card); border:1px solid var(--dm-border);
  border-radius:16px; padding:6px 12px; margin-bottom:6px;
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

/* Chat composer attach control: a (+) that morphs into a 🔗 on click */
.st-key-dm_attach button, .st-key-dm_attach_open button {
  width:46px; height:46px; min-height:46px; padding:0; border-radius:50%;
  font-size:1.3rem; line-height:1; display:grid; place-items:center;
  border:1px solid var(--dm-border); color:#fff;
  background:linear-gradient(135deg,var(--dm-accent),var(--dm-accent-2));
  box-shadow:0 4px 12px rgba(120,90,50,0.22);
  transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.st-key-dm_attach button:hover, .st-key-dm_attach_open button:hover {
  transform: translateY(-2px) scale(1.06); filter:brightness(1.05);
  box-shadow:0 6px 16px rgba(120,90,50,0.28);
}
/* The "broaden into a link" morph, played when the open-state button mounts */
.st-key-dm_attach_open button { animation: dmMorph .34s cubic-bezier(.34,1.56,.64,1) both; }
@keyframes dmMorph {
  0%   { transform: scale(.45) rotate(-90deg); opacity:.35; }
  60%  { transform: scale(1.18) rotate(10deg); }
  100% { transform: scale(1) rotate(0); opacity:1; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Session state -------------------------------------------------------------- #
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_history_id", None)
st.session_state.setdefault("active_doc", None)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("attach_open", False)


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
            vectorstore.add_documents(chunks, source_name=name)

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


def _render_sources(chunks) -> None:
    if not chunks:
        return
    with st.expander(f"📎 Sources ({len(chunks)})"):
        for i, chunk in enumerate(chunks, start=1):
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page")
            page_str = f", p.{page + 1}" if isinstance(page, int) else ""
            st.markdown(f"**{i}. {src}{page_str}** · relevance `{chunk.score:.2f}`")
            st.caption(chunk.text[:500] + ("…" if len(chunk.text) > 500 else ""))


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
                page_str = f", p.{page + 1}" if isinstance(page, int) else ""
                st.markdown(
                    f"**{i}. {s.get('source','unknown')}{page_str}** · "
                    f"relevance `{s.get('score', 0):.2f}`"
                )
                st.caption(s.get("snippet", ""))
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
    st.markdown(
        """<div class="dm-bar"><div class="logo">📄</div>
        <div class="name">DocuMind</div>
        <div class="tag">· private PDF research, on your machine</div></div>""",
        unsafe_allow_html=True,
    )
    # Brand sits in its own row above; the controls line up beneath it.
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.2, 0.6])
    with c1:
        _upload_popover()
    with c2:
        _documents_popover(sources, hist)
    with c3:
        _history_popover(hist)
    with c4:
        _actions_popover()


def _chat_attach() -> None:
    """A (+) in the chat composer that broadens into a 🔗 and reveals an uploader.

    Collapsed it shows a circular ``＋``; clicking it remounts the button as a
    ``🔗`` with a spring "morph" animation and drops a file uploader beneath the
    composer. Files are processed through the same pipeline as the top-bar
    uploader, then the chat jumps into the freshly indexed document.
    """
    open_ = st.session_state.attach_open
    # The key encodes the open/closed state so scoped CSS can style each phase.
    key = "dm_attach_open" if open_ else "dm_attach"
    glyph = "🔗" if open_ else "＋"
    col, _ = st.columns([1, 9])
    with col:
        if st.button(glyph, key=key, help="Attach PDFs to this chat"):
            st.session_state.attach_open = not open_
            st.rerun()
    if not open_:
        return
    uploaded = st.file_uploader(
        "Attach PDF(s)", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed", key="dm_attach_files",
    )
    if uploaded and st.button(
        "Upload & process", type="primary", key="dm_attach_go", use_container_width=True
    ):
        processed = _ingest(uploaded)
        st.session_state.attach_open = False
        if processed:
            _open_document(processed[-1], history.load())
        st.rerun()


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

    # Composer: a (+) attach control that morphs into a 🔗, then the input box.
    _chat_attach()
    placeholder = (
        f"Ask about {active_doc}…" if active_doc else "Ask a question about your documents…"
    )
    typed = st.chat_input(placeholder)
    prompt = st.session_state.pop("pending_prompt", None) or typed
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
