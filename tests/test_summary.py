from langchain_core.documents import Document

from documind import summary
from documind.config import Settings
from documind.ingestion import document_stats


def _settings(tmp_path, budget=100):
    return Settings(summaries_file=tmp_path / "summaries.json", summary_char_budget=budget)


def test_summary_input_respects_char_budget(tmp_path):
    s = _settings(tmp_path, budget=20)
    chunks = [
        Document(page_content="x" * 50, metadata={}),
        Document(page_content="y" * 50, metadata={}),
    ]
    text = summary.summary_input_text(chunks, settings=s)
    assert len(text) <= 20


def test_build_messages_marks_doc_untrusted(tmp_path):
    s = _settings(tmp_path)
    msgs = summary.build_messages(
        "report_pdf", [Document(page_content="hello", metadata={})], settings=s
    )
    assert msgs[0]["role"] == "system"
    assert "<<<DOC" in msgs[1]["content"] and "DOC>>>" in msgs[1]["content"]
    assert "untrusted" in msgs[1]["content"].lower()


def test_save_get_clear_roundtrip(tmp_path):
    s = _settings(tmp_path)
    stats = {"pages": 3, "chunks": 12, "words": 900, "characters": 5000}
    summary.save("report_pdf", "### Overview\nA report.", stats, settings=s)

    got = summary.get("report_pdf", settings=s)
    assert got is not None
    assert "Overview" in got.markdown
    assert got.stats["pages"] == 3

    assert summary.get("missing_pdf", settings=s) is None
    summary.clear(s)
    assert summary.get("report_pdf", settings=s) is None


def test_document_stats():
    chunks = [
        Document(page_content="one two three", metadata={"page": 0}),
        Document(page_content="four five", metadata={"page": 1}),
    ]
    stats = document_stats(chunks)
    assert stats["pages"] == 2
    assert stats["chunks"] == 2
    assert stats["words"] == 5
