from langchain_core.documents import Document

from documind.config import Settings
from documind.ingestion import _clean, normalize_name, split_documents


def test_normalize_name_replaces_unsafe_chars():
    assert normalize_name("My File-v1.2.pdf") == "My_File_v1_2_pdf"


def test_split_documents_tags_source_and_chunks():
    long_text = ("This is a sentence. " * 200).strip()
    docs = [Document(page_content=long_text, metadata={"page": 0})]
    chunks = split_documents(docs, source_name="doc_pdf")

    assert len(chunks) > 1  # long text should split into multiple chunks
    assert all(c.metadata["source"] == "doc_pdf" for c in chunks)
    assert all("page" in c.metadata for c in chunks)
    # Chunks are sequentially indexed for clean ordering/citation.
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_split_documents_empty_input():
    assert split_documents([], source_name="empty_pdf") == []


def test_clean_normalises_whitespace():
    messy = "Hello    world\r\n\n\n\nNext   line   \n"
    cleaned = _clean(messy)
    assert "    " not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Hello world")


def test_tiny_and_empty_chunks_are_dropped():
    s = Settings(chunk_size=1000, chunk_overlap=0, min_chunk_chars=40)
    docs = [Document(page_content="tiny", metadata={"page": 0})]  # below min length
    assert split_documents(docs, source_name="d_pdf", settings=s) == []
