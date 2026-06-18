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
    chunk_size: int = Field(default=400, ge=50, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)
    n_results: int = Field(
        default=10, ge=1, le=50, description="Candidates fetched from the vector store."
    )
    top_k_rerank: int = Field(
        default=3, ge=1, le=20, description="Chunks kept after re-ranking."
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

    @property
    def ollama_embeddings_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/embeddings"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
