"""Pipeline tests that stub out Ollama/Chroma so they run without a server."""

from documind import pipeline, vectorstore
from documind.config import get_settings


def test_retrieve_handles_empty_store(monkeypatch):
    monkeypatch.setattr(
        vectorstore, "query", lambda *a, **k: {"documents": [[]], "metadatas": [[]]}
    )
    result = pipeline.retrieve("anything")
    assert result.is_empty
    assert result.context == ""


def test_retrieve_forwards_source_filter(monkeypatch):
    captured = {}

    def fake_query(prompt, n_results=None, source=None, settings=None):
        captured["source"] = source
        return {"documents": [[]], "metadatas": [[]]}

    monkeypatch.setattr(vectorstore, "query", fake_query)
    pipeline.retrieve("q", source="report_pdf")
    assert captured["source"] == "report_pdf"


def test_retrieve_reranks_and_builds_context(monkeypatch):
    fake = {
        "documents": [["alpha chunk", "beta chunk", "gamma chunk"]],
        "metadatas": [[{"source": "a"}, {"source": "b"}, {"source": "c"}]],
    }
    monkeypatch.setattr(vectorstore, "query", lambda *a, **k: fake)

    from documind.reranker import RankedChunk

    def fake_rerank(query, documents, metadatas=None, top_k=None):
        return [
            RankedChunk(text=documents[1], metadata=metadatas[1], score=0.9),
            RankedChunk(text=documents[0], metadata=metadatas[0], score=0.5),
        ]

    monkeypatch.setattr(pipeline, "rerank", fake_rerank)

    result = pipeline.retrieve("question", settings=get_settings())
    assert not result.is_empty
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "beta chunk"
    assert "beta chunk" in result.context and "alpha chunk" in result.context
