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


def retrieve(question: str, settings: Settings | None = None) -> RetrievalResult:
    """Retrieve candidate chunks for ``question`` and re-rank them.

    Returns the concatenated context string (fed to the LLM) and the ranked
    chunks (used for citations in the UI).
    """
    settings = settings or get_settings()
    results = vectorstore.query(question, n_results=settings.n_results, settings=settings)

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
    context = "\n\n".join(chunk.text for chunk in ranked)
    return RetrievalResult(context=context, chunks=ranked)


def answer(
    question: str,
    history: list[dict] | None = None,
    settings: Settings | None = None,
) -> tuple[Iterator[str], RetrievalResult]:
    """Run the full RAG pipeline.

    Returns a streaming token iterator and the retrieval result so the caller
    can display both the answer and its sources.
    """
    from .llm import stream_answer  # local import keeps ollama optional for tests

    retrieval = retrieve(question, settings=settings)
    stream = stream_answer(retrieval.context, question, history=history)
    return stream, retrieval
