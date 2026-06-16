"""Streamlit chat UI for DocuMind.

Run with:  streamlit run src/documind/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/documind/app.py` to work without `pip install -e .`
# by ensuring the project's `src` directory is importable.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from documind import vectorstore
from documind.config import get_settings
from documind.ingestion import process_uploaded_file
from documind.pipeline import answer

st.set_page_config(page_title="DocuMind — Chat with your PDFs", page_icon="📄", layout="wide")

# Keep the assistant turns we want to show + replay as conversational history.
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role", "content"}]


def _ingest(files) -> None:
    for f in files:
        with st.spinner(f"Processing {f.name}…"):
            chunks = process_uploaded_file(f)
            stored = vectorstore.add_documents(
                chunks, source_name=chunks[0].metadata["source"] if chunks else f.name
            )
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


def main() -> None:
    get_settings()  # validate config early

    with st.sidebar:
        st.header("📄 DocuMind")
        st.caption("Local, private RAG over your PDFs.")

        uploaded = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
        )
        if uploaded and st.button("Process documents", type="primary"):
            _ingest(uploaded)

        st.divider()
        sources = vectorstore.list_sources()
        st.subheader("Indexed documents")
        if sources:
            for s in sources:
                st.markdown(f"- {s}")
        else:
            st.caption("Nothing indexed yet.")

        if sources and st.button("Clear index"):
            vectorstore.reset()
            st.session_state.messages = []
            st.rerun()

    st.title("Chat with your documents")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    if not vectorstore.list_sources():
        st.warning("Upload and process at least one PDF first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversational history (text-only) for the model.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        stream, retrieval = answer(prompt, history=history)
        if retrieval.is_empty:
            reply = "I couldn't find anything relevant in the indexed documents."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return
        reply = st.write_stream(stream)
        _render_sources(retrieval.chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": retrieval.chunks}
    )


if __name__ == "__main__":
    main()
