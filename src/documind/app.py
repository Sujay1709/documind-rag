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

from documind import history, vectorstore
from documind.config import get_settings
from documind.ingestion import process_uploaded_file
from documind.pipeline import answer

st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root {
  --dm-accent: #6C5CE7;
  --dm-accent-2: #00B8D9;
  --dm-card: #161B27;
  --dm-border: rgba(255,255,255,0.08);
}
/* Tighten the default top padding */
.block-container { padding-top: 2.2rem; max-width: 1100px; }

/* Hero */
.dm-hero {
  background: radial-gradient(1200px 400px at 10% -10%, rgba(108,92,231,0.35), transparent 60%),
              radial-gradient(900px 400px at 90% -20%, rgba(0,184,217,0.25), transparent 55%),
              linear-gradient(180deg, #141A28 0%, #0E1117 100%);
  border: 1px solid var(--dm-border);
  border-radius: 22px;
  padding: 48px 44px;
  margin-bottom: 26px;
}
.dm-badge {
  display:inline-block; font-size:0.78rem; letter-spacing:0.12em; text-transform:uppercase;
  color:#cdd2ff; background:rgba(108,92,231,0.18); border:1px solid rgba(108,92,231,0.45);
  padding:5px 12px; border-radius:999px; margin-bottom:18px;
}
.dm-hero h1 {
  font-size: 2.9rem; line-height:1.08; margin:0 0 12px 0; font-weight:800;
  background: linear-gradient(92deg, #ffffff 0%, #b9b4ff 55%, #7ee7ff 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.dm-hero p.dm-sub { font-size:1.12rem; color:#aab1c4; max-width:620px; margin:0; }

/* Feature cards */
.dm-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:8px; }
.dm-card {
  background: var(--dm-card); border:1px solid var(--dm-border); border-radius:16px;
  padding:20px; transition:transform .15s ease, border-color .15s ease;
}
.dm-card:hover { transform:translateY(-3px); border-color:rgba(108,92,231,0.55); }
.dm-card .ic { font-size:1.6rem; }
.dm-card h4 { margin:10px 0 6px 0; font-size:1.02rem; }
.dm-card p { margin:0; color:#9aa2b6; font-size:0.9rem; line-height:1.45; }

/* Stat chips */
.dm-stats { display:flex; gap:12px; margin: 4px 0 18px 0; flex-wrap:wrap; }
.dm-chip {
  background:var(--dm-card); border:1px solid var(--dm-border); border-radius:12px;
  padding:10px 16px; min-width:120px;
}
.dm-chip .n { font-size:1.5rem; font-weight:700; }
.dm-chip .l { font-size:0.78rem; color:#8b93a7; text-transform:uppercase; letter-spacing:0.08em; }

/* Sidebar brand */
.dm-brand { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.dm-brand .logo {
  width:34px; height:34px; border-radius:9px; display:grid; place-items:center; font-size:1.2rem;
  background:linear-gradient(135deg, var(--dm-accent), var(--dm-accent-2));
}
.dm-brand .name { font-weight:800; font-size:1.2rem; }
.dm-tagline { color:#8b93a7; font-size:0.82rem; margin:0 0 6px 2px; }

/* History detail card */
.dm-hist {
  background:var(--dm-card); border:1px solid var(--dm-border); border-radius:16px;
  padding:18px 20px; margin-bottom:14px;
}
.dm-hist .q { font-weight:700; font-size:1.05rem; margin-bottom:6px; }
.dm-hist .t { color:#8b93a7; font-size:0.78rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Session state -------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role","content", "sources"?}]
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ingest(files) -> None:
    for f in files:
        with st.spinner(f"Processing {f.name}…"):
            chunks = process_uploaded_file(f)
            name = chunks[0].metadata["source"] if chunks else f.name
            stored = vectorstore.add_documents(chunks, source_name=name)
        st.success(f"Indexed {stored} chunk(s) from {f.name}")


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


def _hero() -> None:
    st.markdown(
        """
        <div class="dm-hero">
          <span class="dm-badge">Local · Private · Offline</span>
          <h1>Chat with your documents,<br/>powered entirely on your machine.</h1>
          <p class="dm-sub">DocuMind reads your PDFs, finds the passages that matter,
          and answers your questions with citations — no data ever leaves your computer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="dm-grid">
          <div class="dm-card"><div class="ic">📥</div><h4>Ingest any PDF</h4>
            <p>Upload one or many PDFs. DocuMind splits and indexes them locally with Ollama embeddings.</p></div>
          <div class="dm-card"><div class="ic">🎯</div><h4>Precise retrieval</h4>
            <p>A cross-encoder re-ranks candidates so the most relevant passages reach the model.</p></div>
          <div class="dm-card"><div class="ic">🔎</div><h4>Grounded answers</h4>
            <p>Every response streams in with sources — file, page, and relevance score.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Upload a PDF in the sidebar and click **Process documents** to begin.")


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
        st.divider()

        st.subheader("📥 Add documents")
        uploaded = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded and st.button("Process documents", type="primary", use_container_width=True):
            _ingest(uploaded)
            st.rerun()

        st.divider()
        st.subheader("📚 Indexed documents")
        if sources:
            for s in sources:
                st.markdown(f"- {s}")
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
                st.rerun()
        with c2:
            if st.button("🗑️ Clear all", use_container_width=True):
                vectorstore.reset()
                history.clear()
                st.session_state.messages = []
                st.session_state.selected_history_id = None
                st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    get_settings()  # validate config early
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

    st.title("Chat with your documents")
    _render_stats(sources, hist)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    conv = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        stream, retrieval = answer(prompt, history=conv)
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
    # Persist this interaction so it survives restarts and appears in History.
    history.add(prompt, reply, chunks=retrieval.chunks)


if __name__ == "__main__":
    main()
