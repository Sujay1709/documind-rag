"""Central configuration.

All settings can be overridden with environment variables (optionally via a
local ``.env`` file). This keeps secrets and machine-specific paths out of the
code and makes the app easy to run in different environments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for DocuMind.

    Values are read from environment variables prefixed with ``DOCUMIND_``,
    e.g. ``DOCUMIND_CHUNK_SIZE=512``.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / models -------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the running Ollama server.",
    )
    chat_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model used to generate answers.",
    )
    embedding_model: str = Field(
        default="nomic-embed-text:latest",
        description="Ollama model used to embed text.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence-Transformers cross-encoder used for re-ranking.",
    )

    # --- Retrieval -------------------------------------------------------
    # Larger, sentence-coherent chunks give the model enough surrounding context
    # to produce complete answers (small 400-char chunks fragmented sentences).
    chunk_size: int = Field(default=1000, ge=50, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    min_chunk_chars: int = Field(
        default=40, ge=0, le=500, description="Drop chunks shorter than this after cleaning."
    )
    # Recall vs. precision: pull a generous candidate pool from the vector store,
    # then let the cross-encoder keep the best few. ``top_k_rerank`` is the most
    # impactful single knob in our eval grid: keeping 12 chunks (vs. 8) raised
    # faithfulness from 0.62 -> 0.68 and keyword recall from 0.75 -> 0.88 on
    # ``eval/datasets/realistic.json``, at a small F1 cost. The default 12 is
    # the best answer-quality point on that dataset; the benchmark script
    # (``make benchmark``) lets you re-derive it for your own corpus.
    n_results: int = Field(
        default=30, ge=1, le=80, description="Candidates fetched from the vector store."
    )
    top_k_rerank: int = Field(
<<<<<<< .merge_file_h4CtzA
        default=8, ge=1, le=20, description="Chunks kept after re-ranking."
=======
        default=12, ge=1, le=20, description="Chunks kept after re-ranking."
>>>>>>> .merge_file_0bje0T
    )

    # --- Generation (Ollama runtime options) -----------------------------
    # The local model runs through Ollama; these map to `options` on the chat
    # call. num_ctx is the most important: Ollama silently truncates anything
    # past the context window, so a small default (2048) drops retrieved chunks
    # mid-answer — the cause of "half" answers on large documents.
    num_ctx: int = Field(
        default=8192,
        ge=512,
        le=131072,
        description="Token context window passed to Ollama (options.num_ctx).",
    )
    max_output_tokens: int = Field(
        default=2048,
        ge=-2,
        le=32768,
        description="Max tokens the model may generate (Ollama options.num_predict; "
        "-1 = unlimited, -2 = fill context).",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; low keeps grounded RAG answers factual.",
    )

    # --- Embedding / indexing -------------------------------------------
    embed_batch_size: int = Field(
        default=128,
        ge=1,
        le=4000,
        description="Chunks embedded and upserted per batch. Bounds memory and "
        "avoids Chroma max-batch errors when indexing very large PDFs "
        "(thousands of chunks).",
    )

    # --- Storage ---------------------------------------------------------
    persist_dir: Path = Field(
        default=Path("./.documind/chroma"),
        description="Directory where the Chroma vector store is persisted.",
    )
    collection_name: str = Field(default="documind")

    # --- Uploads ---------------------------------------------------------
    max_upload_mb: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Max upload size per file (MB). Keep in sync with "
        "[server].maxUploadSize in .streamlit/config.toml.",
    )

    # --- History ---------------------------------------------------------
    history_file: Path = Field(
        default=Path("./.documind/history.json"),
        description="JSON file storing past questions, answers, and sources.",
    )
    max_history: int = Field(
        default=200, ge=1, le=10000, description="Maximum history entries to retain."
    )

    # --- Summaries -------------------------------------------------------
    summaries_file: Path = Field(
        default=Path("./.documind/summaries.json"),
        description="JSON file storing per-document auto-generated summaries.",
    )
    summary_char_budget: int = Field(
        default=6000,
        ge=10,
        le=40000,
        description="Max characters of document text fed to the summarizer.",
    )

    # --- Web API uploads dir (admin path-uploads) ----------------------
    upload_dir: Path = Field(
        default=Path("./uploads"),
        description="Directory the admin path-upload endpoint is allowed to read from.",
    )

    # --- Web API (production hardening) ---------------------------------
    # DOCUMIND_API_TOKEN: if set, every /upload and /chat request must send it
    # in the ``X-Documind-Token`` header. Leave empty for an open public demo.
    api_token: str = Field(
        default="",
        description="Shared secret for /upload and /chat. Empty = public demo.",
    )
    # Per-visitor rate limit on /chat, in requests per minute. The web app
    # tracks an in-memory sliding window keyed by the visitor's IP (or the
    # value of ``X-Forwarded-For`` when running behind a proxy). 0 disables.
    rate_limit_per_min: int = Field(
        default=20,
        ge=0,
        le=10000,
        description="Per-visitor chat rate limit (req/min). 0 disables.",
    )
    # Per-request cap on streamed tokens. The cross-encoder + re-ranker keep
    # generation grounded, but a misbehaving prompt could otherwise ask the
    # model to print 30k tokens. 1500 covers a detailed answer in 1-2 paragraphs.
    max_answer_tokens: int = Field(
        default=1500,
        ge=64,
        le=4096,
        description="Max tokens streamed per chat answer.",
    )
    # Background summarization after upload is expensive (one extra LLM pass).
    # In a public demo we skip it and only show it when an admin token is used.
    summarize_on_upload: bool = Field(
        default=True,
        description="Generate an auto-summary after upload (skipped in public demo).",
    )

    @property
    def ollama_embeddings_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/embeddings"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
