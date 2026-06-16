"""LLM answer generation via Ollama."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import ollama

from .config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are DocuMind, a careful assistant that answers questions about the "
    "user's documents. Use ONLY the information in the provided context. "
    "If the context does not contain the answer, say you don't know based on "
    "the document. Be concise and, where helpful, refer to the source and page "
    "the information came from. Do not invent facts."
)


def _build_messages(context: str, question: str, history: list[dict] | None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Keep prior turns so follow-up questions have conversational context.
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def stream_answer(
    context: str,
    question: str,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Stream the model's answer token-by-token.

    Yields content chunks as they arrive so the UI can render incrementally.
    """
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.chat_model,
        stream=True,
        messages=_build_messages(context, question, history),
    )
    for chunk in response:
        content = chunk.get("message", {}).get("content")
        if content:
            yield content
