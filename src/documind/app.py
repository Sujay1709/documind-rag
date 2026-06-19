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
    initial_sidebar_state="expanded",
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
# Styling — warm beige, light, a little playful
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
.block-container { padding-top: 2rem; max-width: 1080px; }

/* Hero */
.dm-hero {
  position: relative; overflow: hidden;
  background:
    radial-gradient(900px 300px at 12% -20%, rgba(194,146,90,0.28), transparent 60%),
    linear-gradient(180deg, #FBF6EA 0%, #F0E7D4 100%);
  border: 1px solid var(--dm-border);
  border-radius: 24px;
  padding: 46px 44px;
  margin-bottom: 22px;
  animation: dmFade .6s ease both;
}
@keyframes dmFade { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:none;} }
@keyframes dmFloat { 0%,100% {transform:translateY(0) rotate(-3deg);} 50% {transform:translateY(-10px) rotate(3deg);} }
.dm-hero .float {
  position:absolute; right:42px; top:34px; font-size:4.6rem;
  animation: dmFloat 4s ease-in-out infinite; filter: drop-shadow(0 8px 14px rgba(120,90,50,0.25));
}
.dm-badge {
  display:inline-block; font-size:0.74rem; letter-spacing:0.14em; text-transform:uppercase;
  color:#6b4a28; background:rgba(156,107,63,0.14); border:1px solid rgba(156,107,63,0.4);
  padding:5px 12px; border-radius:999px; margin-bottom:16px;
}
.dm-hero h1 {
  font-size: 2.8rem; line-height:1.1; margin:0 0 12px 0; font-weight:800; color:var(--dm-text);
}
.dm-hero h1 .accent { color: var(--dm-accent); }
.dm-hero p.dm-sub { font-size:1.1rem; color:var(--dm-muted); max-width:600px; margin:0; }

/* Feature cards */
.dm-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:6px; }
.dm-card {
  background: var(--dm-card); border:1px solid var(--dm-border); border-radius:16px;
  padding:20px; transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.dm-card:hover { transform:translateY(-4px); border-color:var(--dm-accent-2);
  box-shadow:0 10px 24px rgba(120,90,50,0.12); }
.dm-card .ic { font-size:1.7rem; }
.dm-card h4 { margin:10px 0 6px 0; font-size:1.02rem; color:var(--dm-text); }
.dm-card p { margin:0; color:var(--dm-muted); font-size:0.9rem; line-height:1.45; }

/* How-it-works strip */
.dm-steps { display:flex; gap:14px; margin-top:16px; flex-wrap:wrap; }
.dm-step { flex:1; min-width:180px; background:var(--dm-card); border:1px solid var(--dm-border);
  border-radius:14px; padding:14px 16px; }
.dm-step .num { display:inline-grid; place-items:center; width:26px; height:26px; border-radius:50%;
  background:linear-gradient(135deg,var(--dm-accent),var(--dm-accent-2)); color:#fff; font-weight:700;
  font-size:0.85rem; margin-bottom:8px; }
.dm-step b { color:var(--dm-text); } .dm-step p { margin:4px 0 0 0; color:var(--dm-muted); font-size:0.86rem; }

/* Stat chips */
.dm-stats { display:flex; gap:12px; margin: 2px 0 14px 0; flex-wrap:wrap; }
.dm-chip { background:var(--dm-card); border:1px solid var(--dm-border); border-radius:12px;
  padding:10px 16px; min-width:120px; }
.dm-chip .n { font-size:1.5rem; font-weight:800; color:var(--dm-accent); }
.dm-chip .l { font-size:0.76rem; color:var(--dm-muted); text-transform:uppercase; letter-spacing:0.08em; }

/* Sidebar brand */
.dm-brand { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.dm-brand .logo { width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
  font-size:1.25rem; background:linear-gradient(135deg,var(--dm-accent),var(--dm-accent-2)); }
.dm-brand .name { font-weight:800; font-size:1.25rem; color:var(--dm-text); }
.dm-tagline { color:var(--dm-muted); font-size:0.82rem; margin:0 0 6px 2px; }
.dm-lock { background:rgba(156,107,63,0.10); border:1px solid var(--dm-border); border-radius:12px;
  padding:10px 12px; font-size:0.82rem; color:#5d4a30; }

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

/* Section heading */
.dm-h2 { font-weight:800; color:var(--dm-text); margin: 6px 0 2px 0; }
.dm-suggest-label { color:var(--dm-muted); font-size:0.85rem; margin: 6px 0 2px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Session state -------------------------------------------------------------- #
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_history_id", None)
st.session_state.setdefault("active_doc", None)
st.session_state.setdefault("pending_prompt", None)


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


def _hero() -> None:
    st.markdown(
        """
        <div class="dm-hero">
          <span class="float">📚</span>
          <span class="dm-badge">Local · Private · Offline</span>
          <h1>Ask your documents <span class="accent">anything.</span></h1>
          <p class="dm-sub">Upload a PDF and chat with it. DocuMind finds the passages that
          matter and answers with citations — all on your machine, nothing ever leaves it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="dm-grid">
          <div class="dm-card"><div class="ic">📥</div><h4>Ingest any PDF</h4>
            <p>Upload one or many PDFs — DocuMind splits and indexes them locally with Ollama embeddings.</p></div>
          <div class="dm-card"><div class="ic">🎯</div><h4>Precise retrieval</h4>
            <p>A cross-encoder re-ranks candidates so the most relevant passages reach the model.</p></div>
          <div class="dm-card"><div class="ic">🔐</div><h4>Private by design</h4>
            <p>Answers come only from your documents, and your data never leaves this computer.</p></div>
        </div>
        <div class="dm-steps">
          <div class="dm-step"><div class="num">1</div><b>Upload</b><p>Drop your PDFs in the sidebar.</p></div>
          <div class="dm-step"><div class="num">2</div><b>Process</b><p>They're indexed in seconds, locally.</p></div>
          <div class="dm-step"><div class="num">3</div><b>Chat</b><p>Ask away and get cited answers.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Upload a PDF in the sidebar and click **Process documents** to begin.")


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
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar(sources, hist):
    with st.sidebar:
        st.markdown(
            """<div class="dm-brand"><div class="logo">📄</div>
            <div class="name">DocuMind</div></div>
            <p class="dm-tagline">Your private PDF research assistant</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="dm-lock">🔒 Answers come only from your uploaded documents. '
            "Nothing is sent off this machine.</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("📥 Add documents")
        st.caption(f"PDF only · up to **{settings.max_upload_mb} MB** per file.")
        uploaded = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Process documents", type="primary", use_container_width=True):
            processed = _ingest(uploaded)
            if processed:
                # Jump straight into the freshly processed document's chat.
                _open_document(processed[-1], history.load())
            st.rerun()

        st.divider()
        st.subheader("📚 Indexed documents")
        st.caption("Click a document to revisit it with its chat history.")
        if sources:
            for s in sources:
                active = st.session_state.active_doc == s
                if st.button(
                    f"{'📂' if active else '📄'} {s}",
                    key=f"doc_{s}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    _open_document(s, hist)
        else:
            st.caption("Nothing indexed yet.")

        st.divider()
        st.subheader("🕘 History")
        if hist:
            for entry in hist[:10]:
                label = (entry.question[:42] + "…") if len(entry.question) > 42 else entry.question
                if st.button(f"🔹 {label}", key=f"hist_{entry.id}", use_container_width=True):
                    st.session_state.selected_history_id = entry.id
                    st.rerun()
        else:
            st.caption("Your past questions will appear here.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 New chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.selected_history_id = None
                st.session_state.active_doc = None
                st.rerun()
        with c2:
            if st.button("🗑️ Clear all", use_container_width=True):
                vectorstore.reset()
                history.clear()
                summary.clear()
                st.session_state.messages = []
                st.session_state.selected_history_id = None
                st.session_state.active_doc = None
                st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    sources = vectorstore.list_sources()
    hist = history.load()

    _sidebar(sources, hist)

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
        _hero()
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
