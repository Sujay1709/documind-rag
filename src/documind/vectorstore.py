"""Chroma vector store wrapper with Ollama embeddings."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from langchain_core.documents import Document

from .config import Settings, get_settings

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Return (and cache) the persistent Chroma collection.

    Caching avoids re-opening the DB and re-creating the embedding function on
    every Streamlit rerun. chromadb is imported lazily so the package can be
    imported without it (e.g. for unit tests that stub the store).
    """
    import chromadb
    from chromadb.api.types import Embeddable, EmbeddingFunction
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    settings = get_settings()
    embedding_fn = OllamaEmbeddingFunction(
        url=settings.ollama_embeddings_url,
        model_name=settings.embedding_model,
    )
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.persist_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=cast(EmbeddingFunction[Embeddable], embedding_fn),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(chunks: list[Document], source_name: str) -> int:
    """Upsert chunks into the collection. Returns the number stored.

    Ids are deterministic (``{source}_{idx}``) so re-uploading the same file
    updates rather than duplicates its chunks.
    """
    if not chunks:
        return 0
    collection = get_collection()
    documents = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Upserted %d chunk(s) for %s", len(chunks), source_name)
    return len(chunks)


def query(prompt: str, n_results: int | None = None, settings: Settings | None = None) -> dict:
    """Query the collection for the most similar chunks to ``prompt``."""
    settings = settings or get_settings()
    n_results = n_results or settings.n_results
    collection = get_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)


def list_sources() -> list[str]:
    """Return the distinct source file names currently indexed."""
    collection = get_collection()
    meta = collection.get(include=["metadatas"]).get("metadatas") or []
    return sorted({m.get("source", "unknown") for m in meta})


def reset() -> None:
    """Delete all documents from the collection (clears the cache too)."""
    collection = get_collection()
    existing = collection.get().get("ids") or []
    if existing:
        collection.delete(ids=existing)
    logger.info("Cleared %d document(s) from collection", len(existing))
