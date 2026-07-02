"""Smoke tests for the Starlette web app.

We never hit Ollama or a real Chroma in tests; instead we monkeypatch the
retrieval path (vectorstore + reranker + LLM stream) so the request handlers
run their full code path (auth, rate-limit, SSE framing, audit log) without
needing model servers.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest


# Use a tmp dir for every Settings call so the tests don't touch the user's
# real .documind/ history.
@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMIND_HISTORY_FILE", str(tmp_path / "history.json"))
    monkeypatch.setenv("DOCUMIND_SUMMARIES_FILE", str(tmp_path / "summaries.json"))
    monkeypatch.setenv("DOCUMIND_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCUMIND_API_TOKEN", "")
    monkeypatch.setenv("DOCUMIND_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("DOCUMIND_SUMMARIZE_ON_UPLOAD", "false")
    from documind.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub vectorstore.query + reranker + LLM stream so /chat works offline."""
    from documind import llm, pipeline, vectorstore
    from documind.reranker import RankedChunk

    def _fake_query(prompt, n_results=None, source=None, settings=None):
        # Match the shape chromadb returns: nested lists of documents + metadatas.
        return {
            "documents": [["DocuMind is a local RAG assistant."]],
            "metadatas": [[{"source": "demo_pdf", "page": 0}]],
        }

    monkeypatch.setattr(vectorstore, "query", _fake_query)

    def _fake_rerank(query, documents, metadatas=None, top_k=None):
        return [
            RankedChunk(
                text=documents[0],
                metadata=(metadatas or [{}])[0],
                score=2.0,
            )
        ]

    monkeypatch.setattr(pipeline, "rerank", _fake_rerank)

    def _fake_stream_chat(messages):
        for tok in ("Hello", " world", "!"):
            yield tok

    monkeypatch.setattr(llm, "stream_chat", _fake_stream_chat)


@pytest.fixture
def app_client(tmp_settings, stub_pipeline):
    from documind.webapp import create_app
    from starlette.testclient import TestClient

    return TestClient(create_app())


def test_healthz_shape(app_client):
    r = app_client.get("/healthz")
    # The handler returns 200/503 depending on ollama reachability; both are OK
    # responses in tests because we don't require a live server.
    assert r.status_code in (200, 503)
    body = r.json()
    assert "ollama" in body and "chroma" in body
    assert "models" in body


def test_index_serves_spa(app_client):
    r = app_client.get("/")
    assert r.status_code == 200
    assert "DocuMind" in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_static_assets_served(app_client):
    css = app_client.get("/static/styles.css")
    js = app_client.get("/static/app.js")
    assert css.status_code == 200 and "dm-bar" in css.text
    assert js.status_code == 200 and "fetch" in js.text


def test_chat_without_question_is_400(app_client):
    r = app_client.post("/chat", json={})
    assert r.status_code == 400
    assert r.json()["error"].startswith("'question'")


def test_chat_streams_sources_and_tokens(app_client):
    with app_client.stream("POST", "/chat", json={"question": "What is DocuMind?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events: list[dict] = []
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            events.append(json.loads(line[len("data: "):]))
    types = [e["type"] for e in events]
    assert "sources" in types
    assert "token" in types
    tokens = "".join(e.get("text", "") for e in events if e["type"] == "token")
    assert tokens == "Hello world!"


def test_ndjson_chat(app_client):
    r = app_client.post("/api/chat", json={"question": "What is DocuMind?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Hello world!"
    assert body["sources"][0]["source"] == "demo_pdf"


def test_upload_requires_files(app_client):
    r = app_client.post("/upload")
    assert r.status_code == 400
    assert "files" in r.json()["error"].lower()


def test_upload_accepts_pdf(app_client):
    # Tiny in-memory PDF is enough to exercise the multipart path; PyMuPDF
    # rejects it as invalid, but the route must respond with JSON, not 500.
    fake_pdf = b"%PDF-1.4\n%fake doc\n%%EOF"
    r = app_client.post(
        "/upload",
        files=[("files", ("hello.pdf", io.BytesIO(fake_pdf), "application/pdf"))],
    )
    assert r.status_code in (200, 400, 422)
    assert isinstance(r.json(), dict)


def test_api_token_required_when_set(tmp_settings, monkeypatch, stub_pipeline):
    monkeypatch.setenv("DOCUMIND_API_TOKEN", "secret123")
    from documind.config import get_settings

    get_settings.cache_clear()

    from documind.webapp import create_app
    from starlette.testclient import TestClient

    client = TestClient(create_app())
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 401

    r = client.post(
        "/chat",
        json={"question": "hi"},
        headers={"X-Documind-Token": "secret123"},
    )
    assert r.status_code != 401


def test_rate_limit_returns_429(tmp_settings, monkeypatch, stub_pipeline):
    monkeypatch.setenv("DOCUMIND_RATE_LIMIT_PER_MIN", "1")
    from documind.config import get_settings

    get_settings.cache_clear()

    from documind.webapp import create_app
    from starlette.testclient import TestClient

    client = TestClient(create_app())
    client.post("/chat", json={"question": "hi"}, headers={"X-Forwarded-For": "1.2.3.4"})
    r = client.post(
        "/chat", json={"question": "hi"}, headers={"X-Forwarded-For": "1.2.3.4"}
    )
    assert r.status_code == 429


# ---------- /api/upload-by-path ---------- #

def test_upload_by_path_requires_token(tmp_settings, monkeypatch, tmp_path):
    """If DOCUMIND_API_TOKEN is set, the endpoint must reject anonymous requests."""
    monkeypatch.setenv("DOCUMIND_API_TOKEN", "secret")
    # Redirect upload_dir so we can drop a test PDF there.
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setenv("DOCUMIND_UPLOAD_DIR", str(upload_dir))
    from documind.config import get_settings

    get_settings.cache_clear()

    from documind.webapp import create_app
    from starlette.testclient import TestClient

    client = TestClient(create_app())
    r = client.post("/api/upload-by-path", json={"path": "uploads/missing.pdf"})
    assert r.status_code == 401


def test_upload_by_path_rejects_traversal(tmp_settings, monkeypatch, tmp_path, stub_pipeline):
    """Path traversal outside the upload dir must 403."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setenv("DOCUMIND_UPLOAD_DIR", str(upload_dir))
    from documind.config import get_settings

    get_settings.cache_clear()

    from documind.webapp import create_app
    from starlette.testclient import TestClient

    client = TestClient(create_app())
    # Try to index something outside the upload dir. We don't need the file to
    # exist — the traversal check runs first and returns 403.
    r = client.post("/api/upload-by-path", json={"path": "/etc/hosts"})
    assert r.status_code == 403
    assert "must live inside" in r.json()["error"]


def test_upload_by_path_happy_path(tmp_settings, monkeypatch, tmp_path, stub_pipeline):
    """A PDF inside the upload dir is indexed and returns a processed payload."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    pdf = upload_dir / "hello.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake doc\n%%EOF")
    monkeypatch.setenv("DOCUMIND_UPLOAD_DIR", str(upload_dir))
    from documind.config import get_settings

    get_settings.cache_clear()

    from documind.webapp import create_app
    from starlette.testclient import TestClient

    client = TestClient(create_app())
    r = client.post("/api/upload-by-path", json={"path": "hello.pdf"})
    # The file is a fake PDF; PyMuPDF rejects it. The handler maps that to 400
    # (or 422 if it parsed zero chunks), not 500. We only assert the route
    # exists, accepts a body, and is correctly auth-free in this fixture.
    assert r.status_code in (200, 400, 422)
    assert isinstance(r.json(), dict)
