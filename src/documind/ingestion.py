"""PDF ingestion: load a PDF and split it into overlapping text chunks."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# Separators ordered from coarse to fine so the splitter prefers natural
# boundaries: paragraphs, then lines, then sentence ends (with the trailing
# space, so we don't split inside decimals/abbreviations like "3.5" or "U.S."),
# then words. Keeping sentences intact yields cleaner, more complete chunks.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    """Normalise whitespace so chunks are tidy and embed consistently."""
    text = text.replace("\r", "")
    text = _WS_RE.sub(" ", text)        # collapse runs of spaces/tabs
    text = _NL_RE.sub("\n\n", text)     # cap blank-line runs
    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
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
    raw_chunks = _build_splitter(settings).split_documents(docs)

    chunks: list[Document] = []
    for chunk in raw_chunks:
        cleaned = _clean(chunk.page_content)
        # Drop empty/boilerplate fragments so retrieval sees only real content.
        if len(cleaned) < settings.min_chunk_chars:
            continue
        chunk.page_content = cleaned
        chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_index"] = len(chunks)
        chunks.append(chunk)

    logger.info("Split %s into %d clean chunk(s)", source_name, len(chunks))
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


def document_stats(chunks: list[Document]) -> dict[str, int]:
    """Quick stats for a processed document: pages, chunks, words, characters."""
    pages = {c.metadata.get("page") for c in chunks if isinstance(c.metadata.get("page"), int)}
    text = " ".join(c.page_content for c in chunks)
    return {
        "pages": (max(pages) + 1) if pages else 0,
        "chunks": len(chunks),
        "words": len(text.split()),
        "characters": len(text),
    }
