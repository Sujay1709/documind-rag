"""Smoke test for scripts/smoke_webapp.py without an actual server.

We stub urllib so the script's network calls return canned responses, then
verify it aggregates failures correctly and exits non-zero on the first bad
assertion.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "smoke_webapp.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fake(responses):
    """Return a fake ``urlopen`` that dispatches by (method, path)."""

    def _dispatch(req_or_url, timeout=10):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1].split("?")[0]
        method = "GET" if isinstance(req_or_url, str) else req_or_url.get_method()
        status, body = responses[(method, path)]
        raw = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")

        if status >= 400:
            class _Err(urllib.error.HTTPError):
                def __init__(self):
                    super().__init__(url, status, "err", {}, None)

                def read(self):
                    return raw

            raise _Err()

        class _Resp:
            def __init__(self):
                self.status = status

            def read(self):
                return raw

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return None

        return _Resp()

    return _dispatch


def test_smoke_script_passes_against_healthy_stub(monkeypatch, capsys):
    mod = _load()
    responses = {
        ("GET", "/healthz"): (200, {"ollama": True, "chroma": True, "status": "ok"}),
        ("GET", "/api/sources"): (200, {"sources": ["a_pdf"]}),
        ("GET", "/api/history"): (200, {"entries": []}),
        ("GET", "/"): (200, "<html><body>hi</body></html>"),
        ("POST", "/chat"): (400, {"error": "'question' is required"}),
        ("POST", "/api/chat"): (400, {"error": "'question' is required"}),
    }
    monkeypatch.setattr("urllib.request.urlopen", _make_fake(responses))
    rc = mod.main()
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "All smoke checks passed" in captured.out


def test_smoke_script_fails_when_endpoint_missing(monkeypatch, capsys):
    mod = _load()
    responses = {
        ("GET", "/healthz"): (500, "boom"),
        ("GET", "/api/sources"): (200, {"sources": []}),
        ("GET", "/api/history"): (200, {"entries": []}),
        ("GET", "/"): (200, "<html></html>"),
        ("POST", "/chat"): (400, {"error": "x"}),
        ("POST", "/api/chat"): (400, {"error": "x"}),
    }
    monkeypatch.setattr("urllib.request.urlopen", _make_fake(responses))
    rc = mod.main()
    captured = capsys.readouterr()
    assert rc == 1
    assert "✗" in captured.err
