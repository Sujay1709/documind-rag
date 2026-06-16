from langchain_core.documents import Document

from documind.ingestion import normalize_name, split_documents


def test_normalize_name_replaces_unsafe_chars():
    assert normalize_name("My File-v1.2.pdf") == "My_File_v1_2_pdf"


def test_split_documents_tags_source_and_chunks():
    long_text = ("This is a sentence. " * 200).strip()
    docs = [Document(page_content=long_text, metadata={"page": 0})]
    chunks = split_documents(docs, source_name="doc_pdf")

    assert len(chunks) > 1  # long text should split into multiple chunks
    assert all(c.metadata["source"] == "doc_pdf" for c in chunks)
    assert all("page" in c.metadata for c in chunks)


def test_split_documents_empty_input():
    assert split_documents([], source_name="empty_pdf") == []
