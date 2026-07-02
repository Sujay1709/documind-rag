"""High-level RAG orchestration tying retrieval, re-ranking and generation together."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import vectorstore
from .config import Settings, get_settings
from .reranker import RankedChunk, rerank

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """The context assembled for a question, plus the chunks it came from."""

    context: str
    chunks: list[RankedChunk] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


def retrieve(
    question: str,
    source: str | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Retrieve candidate chunks for ``question`` and re-rank them.

    Returns the concatenated context string (fed to the LLM) and the ranked
    chunks (used for citations in the UI). When ``source`` is set, retrieval is
    restricted to that document.
    """
    settings = settings or get_settings()
    results = vectorstore.query(
        question, n_results=settings.n_results, source=source, settings=settings
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    if not documents:
        return RetrievalResult(context="", chunks=[])

    ranked = rerank(
        question,
        documents=documents,
        metadatas=metadatas,
        top_k=settings.top_k_rerank,
    )
    # The cross-encoder selects the most relevant chunks, but returns them in
    # relevance order. Reading them in *document* order (by page, then position
    # on the page) lets the model answer sequential questions — a table of
    # contents, numbered steps — in the document's own order instead of
    # jumbling them. Selection stays relevance-based; only presentation order
    # changes, for both the LLM context and the UI citations.
    ordered = sorted(ranked, key=_document_order)
    context = "\n\n".join(chunk.text for chunk in ordered)
    return RetrievalResult(context=context, chunks=ordered)


def _document_order(chunk: RankedChunk) -> tuple[int, int]:
    """Sort key placing a chunk at its natural position in the source document."""
    meta = chunk.metadata or {}
    page = meta.get("page")
    index = meta.get("chunk_index")
    return (
        page if isinstance(page, int) else 0,
        index if isinstance(index, int) else 0,
    )


def answer(
    question: str,
    history: list[dict] | None = None,
    source: str | None = None,
    settings: Settings | None = None,
) -> tuple[Iterator[str], RetrievalResult]:
    """Run the full RAG pipeline.

    Returns a streaming token iterator and the retrieval result so the caller
    can display both the answer and its sources. When ``source`` is set, the
    answer is grounded only in that document.
    """
    from .llm import stream_answer  # local import keeps ollama optional for tests

    retrieval = retrieve(question, source=source, settings=settings)
    stream = stream_answer(retrieval.context, question, history=history)
    return stream, retrieval
