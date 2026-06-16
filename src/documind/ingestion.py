"""PDF ingestion: load a PDF and split it into overlapping text chunks."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Separators ordered from coarse to fine so the splitter prefers natural
# boundaries (paragraphs, then sentences) before resorting to characters.
_SEPARATORS = ["\n\n", "\n", ".", "?", "!", " ", ""]


def _build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF from disk into LangChain ``Document`` objects (one per page)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    docs = PyMuPDFLoader(str(path)).load()
    logger.info("Loaded %d page(s) from %s", len(docs), path.name)
    return docs


def split_documents(
    docs: list[Document],
    source_name: str,
    settings: Settings | None = None,
) -> list[Document]:
    """Split page-level documents into chunks, tagging each with its source.

    The ``source`` metadata is normalised to ``source_name`` so chunks can be
    grouped and cited back to the originating file.
    """
    settings = settings or get_settings()
    chunks = _build_splitter(settings).split_documents(docs)
    for chunk in chunks:
        chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
        chunk.metadata["source"] = source_name
    logger.info("Split %s into %d chunk(s)", source_name, len(chunks))
    return chunks


def process_pdf_bytes(
    data: bytes,
    source_name: str,
    settings: Settings | None = None,
) -> list[Document]:
    """Process raw PDF bytes (e.g. from an upload) into chunks.

    Writes to a temporary file because PyMuPDF needs a real path, then removes
    it. The temp file is always cleaned up, even on error.
    """
    settings = settings or get_settings()
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        docs = load_pdf(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return split_documents(docs, source_name=source_name, settings=settings)


def process_uploaded_file(uploaded_file, settings: Settings | None = None) -> list[Document]:
    """Process a Streamlit ``UploadedFile`` into chunks."""
    return process_pdf_bytes(
        uploaded_file.read(),
        source_name=normalize_name(uploaded_file.name),
        settings=settings,
    )


def normalize_name(file_name: str) -> str:
    """Make a file name safe to use as a vector-store id prefix."""
    return file_name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
