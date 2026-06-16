"""Cross-encoder re-ranking of retrieved chunks.

The vector store returns candidates by embedding similarity, which is fast but
coarse. A cross-encoder scores each (query, chunk) pair jointly and gives a much
more accurate relevance ordering, so we keep only the best few before sending
them to the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:  # avoid importing torch-heavy deps unless actually used
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    text: str
    metadata: dict
    score: float


@lru_cache(maxsize=1)
def _get_encoder(model_name: str) -> CrossEncoder:
    # Imported lazily so the package can be imported (and unit-tested) without
    # pulling in sentence-transformers/torch.
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder %s", model_name)
    return CrossEncoder(model_name)


def rerank(
    query: str,
    documents: list[str],
    metadatas: list[dict] | None = None,
    top_k: int | None = None,
) -> list[RankedChunk]:
    """Re-rank ``documents`` against ``query`` and return the top results.

    Unlike the original implementation, this takes an explicit ``query`` (no
    reliance on a global) and operates on a *list* of documents rather than a
    single concatenated string, so the cross-encoder can actually compare
    candidates.
    """
    settings = get_settings()
    top_k = top_k or settings.top_k_rerank
    if not documents:
        return []

    metadatas = metadatas or [{} for _ in documents]
    encoder = _get_encoder(settings.reranker_model)
    ranks = encoder.rank(query, documents, top_k=min(top_k, len(documents)))

    results: list[RankedChunk] = []
    for rank in ranks:
        idx = rank["corpus_id"]
        results.append(
            RankedChunk(
                text=documents[idx],
                metadata=metadatas[idx] if idx < len(metadatas) else {},
                score=float(rank["score"]),
            )
        )
    return results
