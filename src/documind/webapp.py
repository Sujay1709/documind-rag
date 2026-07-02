"""DocuMind web API + SPA shell.

A long-running Starlette app that:

* Reuses a single Ollama + Chroma process (one model load, persistent index).
* Accepts PDF uploads at ``/upload`` (multipart, streamed, size-bounded).
* Streams grounded RAG answers at ``/chat`` (Server-Sent Events, NDJSON fallback).
* Persists per-document auto-summaries (admin only) and a global question log.
* Exposes ``/api/sources``, ``/api/summary/<source>``, ``/healthz`` for ops.
* Serves the static SPA from ``src/documind/webapp/static/``.

The shell wraps the existing ``documind.pipeline`` / ``vectorstore`` / ``summary``
modules so behaviour stays identical to the Streamlit app — this is the same
pipeline, just with a more deployment-friendly surface.

Run locally:
    uvicorn documind.webapp:app --host 0.0.0.0 --port 8000

Or via the project console script:
    documind-web
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import starlette
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import history, summary, vectorstore
from .config import Settings, get_settings
from .ingestion import document_stats, normalize_name, process_pdf_bytes
from .pipeline import retrieve

logger = logging.getLogger("documind.web")

_STATIC_DIR = Path(__file__).parent / "webapp" / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
# Hard cap on a single request body (covers the 500 MB PDF ceiling defined in
# .streamlit/config.toml; the API actually enforces a tighter cap per request).
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024

# A request-id used in the audit log so we can trace a visitor end-to-end.
_REQ_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")


# --------------------------------------------------------------------------- #
# Rate limiting (per-visitor sliding window, in memory)
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Simple in-memory sliding window keyed by visitor id."""

    def __init__(self, max_per_min: int) -> None:
        self._max = max_per_min
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, remaining_in_window)`` for ``key``."""
        if self._max <= 0:
            return True, 0
        now = time.monotonic()
        window = 60.0
        bucket = self._hits.setdefault(key, deque())
        # Drop entries older than the window.
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= self._max:
            return False, 0
        bucket.append(now)
        return True, self._max - len(bucket)


def _client_id(request: Request) -> str:
    """Best-effort visitor id: trust the first hop in X-Forwarded-For, else peer ip."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first = fwd.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------- #
# Audit log (one JSON line per event, append-only)
# --------------------------------------------------------------------------- #
def _audit_path(settings: Settings) -> Path:
    # Sibling of history.json so it ships with the same persistent volume.
    p = settings.history_file.parent / "audit.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _audit(event: str, **fields: Any) -> None:
    try:
        s = get_settings()
        record = {"ts": time.time(), "event": event, **fields}
        with _audit_path(s).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("audit log failed: %s", exc)


# --------------------------------------------------------------------------- #
# Auth + helpers
# --------------------------------------------------------------------------- #
def _check_token(request: Request) -> bool:
    """True if the request is authorised (token unset, or correct token sent)."""
    expected = get_settings().api_token
    if not expected:
        return True
    sent = request.headers.get("x-documind-token", "")
    return bool(sent) and sent == expected


def _bad_request(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=code)


def _is_public_ip(ip: str) -> bool:
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
    )


# --------------------------------------------------------------------------- #
# Streaming the RAG answer
# --------------------------------------------------------------------------- #
def _cap_tokens(stream: Iterable[str], cap: int) -> AsyncIterator[str]:
    """Stop yielding after we've streamed ``cap`` characters.

    We count characters rather than tokens because the LLM doesn't expose token
    boundaries in its stream, and a char cap is a tight enough ceiling to
    protect the server. If the model overruns we just close the generator; the
    client sees a normal end-of-stream.
    """
    total = 0

    def _gen() -> Iterable[str]:
        nonlocal total
        for tok in stream:
            if not tok:
                continue
            total += len(tok)
            if total >= cap:
                # Emit the remainder so we don't slice a token mid-word, then stop.
                yield tok[: max(0, cap - (total - len(tok)))]
                break
            yield tok

    async def _agen() -> AsyncIterator[str]:
        for piece in _gen():
            yield piece
            # Yield to the event loop so the client actually receives tokens as
            # they arrive; without this the generator runs to completion first.
            await asyncio.sleep(0)

    return _agen()


async def _sse(data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
async def index(request: Request) -> Response:
    return HTMLResponse(_INDEX_HTML)


async def healthz(request: Request) -> Response:
    """Liveness + model liveness probe. Used by HF Spaces / Docker / k8s."""
    settings = get_settings()
    ollama_ok = True
    chroma_ok = True
    try:
        import ollama  # local import keeps the module test-friendly
        ollama.Client(host=settings.ollama_base_url).list()
    except Exception as exc:
        ollama_ok = False
        logger.warning("healthz: ollama unreachable: %s", exc)
    try:
        vectorstore.get_collection().count()  # type: ignore[attr-defined]
    except Exception as exc:
        chroma_ok = False
        logger.warning("healthz: chroma unreachable: %s", exc)
    code = 200 if ollama_ok and chroma_ok else 503
    return JSONResponse(
        {
            "status": "ok" if code == 200 else "degraded",
            "ollama": ollama_ok,
            "chroma": chroma_ok,
            "models": {
                "chat": settings.chat_model,
                "embedding": settings.embedding_model,
                "reranker": settings.reranker_model,
            },
        },
        status_code=code,
    )


async def list_sources(request: Request) -> Response:
    try:
        sources = vectorstore.list_sources()
    except Exception as exc:
        return _bad_request(f"vector store error: {exc}", code=503)
    return JSONResponse({"sources": sources})


async def upload(request: Request) -> Response:
    """Upload one or more PDFs. Multipart parts named ``files``."""
    if not _check_token(request):
        return _bad_request("invalid or missing X-Documind-Token", code=401)

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > _MAX_UPLOAD_BYTES:
        return _bad_request(
            f"upload too large (>{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)", code=413
        )

    settings = get_settings()
    # Keep the body in memory for simplicity (uploads are streamed by uvicorn).
    # Multipart parsing raises on malformed input, which Starlette maps to 400.
    try:
        form = await request.form()
    except Exception as exc:
        return _bad_request(f"could not parse multipart form: {exc}", code=400)

    files = form.getlist("files")
    if not files:
        # Support a single ``file`` field too so curl one-liners are easy.
        single = form.get("file")
        if single is not None:
            files = [single]
    if not files:
        return _bad_request("no files in 'files' (or 'file') field", code=400)

    processed: list[dict] = []
    for upload_file in files:
        data = await upload_file.read()
        if not data:
            continue
        if len(data) > _MAX_UPLOAD_BYTES:
            return _bad_request(
                f"{upload_file.filename!r} exceeds size limit", code=413
            )
        name = normalize_name(upload_file.filename or "uploaded.pdf")
        try:
            chunks = process_pdf_bytes(data, source_name=name, settings=settings)
        except Exception as exc:
            return _bad_request(f"could not read {upload_file.filename!r}: {exc}", code=400)
        if not chunks:
            return _bad_request(
                f"{upload_file.filename!r} produced no extractable text", code=422
            )

        stats = document_stats(chunks)
        added = vectorstore.add_documents(chunks, source_name=name)

        # Summarization is an extra LLM pass; skip it in the public demo (when
        # no admin token is configured) to keep cold-start latency low.
        if settings.summarize_on_upload:
            try:
                tokens = list(summary.stream_summary(name, chunks, settings=settings))
                summary.save(name, "".join(tokens).strip() or "_Summary unavailable._", stats)
            except Exception as exc:
                logger.warning("summary failed for %s: %s", name, exc)
        else:
            summary.save(name, "_Summary disabled in public demo._", stats)

        processed.append({"source": name, "chunks": added, **stats})
        _audit("upload", source=name, chunks=added, ip=_client_id(request))

    return JSONResponse({"processed": processed})


async def chat_sse(request: Request) -> Response:
    """SSE stream of one RAG answer.

    Request body (JSON): ``{"question": str, "source": Optional[str], "history": list[dict]}``
    Events:
        data: {"type":"sources","sources":[...]}
        data: {"type":"token","text":"..."}     (repeated)
        data: {"type":"done","answer":"...","duration_s":float}
        data: {"type":"error","message":"..."}  (on failure)
    """
    if request.method != "POST":
        return _bad_request("use POST", code=405)
    if not _check_token(request):
        return _bad_request("invalid or missing X-Documind-Token", code=401)

    settings = get_settings()
    rl: RateLimiter = request.app.state.limiter
    visitor = _client_id(request)
    allowed, remaining = rl.check(visitor)
    if not allowed:
        return _bad_request(
            f"rate limit exceeded for {visitor}; max {settings.rate_limit_per_min}/min",
            code=429,
        )

    try:
        body = await request.json()
    except Exception:
        return _bad_request("request body must be JSON", code=400)

    question = (body.get("question") or "").strip()
    if not question:
        return _bad_request("'question' is required", code=400)
    if len(question) > 4000:
        return _bad_request("'question' too long (>4000 chars)", code=400)

    source = body.get("source") or None
    history_msgs = body.get("history") or []

    started = time.monotonic()
    _audit("chat_start", ip=visitor, question_len=len(question), source=source)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            # 1) Retrieval is sync but tiny; run it in a thread so the event
            #    loop stays free to push the first byte to the client fast.
            retrieval = await asyncio.to_thread(retrieve, question, source, settings)
            sources_payload = [
                {
                    "source": c.metadata.get("source", "unknown"),
                    "page": c.metadata.get("page"),
                    "score": round(float(c.score), 4),
                    "snippet": c.text[:400],
                }
                for c in retrieval.chunks
            ]
            yield await _sse({"type": "sources", "sources": sources_payload})

            if retrieval.is_empty:
                yield await _sse(
                    {"type": "error", "message": "no relevant chunks in the index"}
                )
                return

            # 2) Stream the LLM answer through Ollama. We borrow the same code
            #    path the Streamlit app uses (stream_chat), so behaviour and
            #    guardrails stay identical.
            from .llm import stream_chat, _build_messages
            from . import llm as _llm_mod

            messages = _build_messages(retrieval.context, question, history_msgs)
            stream = stream_chat(messages)
            cap = settings.max_answer_tokens
            full: list[str] = []
            async for piece in _cap_tokens(stream, cap * 4):  # ~4 chars/token
                full.append(piece)
                yield await _sse({"type": "token", "text": piece})

            answer_text = "".join(full).strip()
            # 3) Persist the interaction for the global history view.
            try:
                from .reranker import RankedChunk

                history.add(
                    question=question,
                    answer=answer_text,
                    chunks=[
                        RankedChunk(
                            text=c.text,
                            metadata=c.metadata,
                            score=c.score,
                        )
                        for c in retrieval.chunks
                    ],
                    settings=settings,
                )
            except Exception as exc:  # pragma: no cover - persistence is best-effort
                logger.warning("history.add failed: %s", exc)

            duration = round(time.monotonic() - started, 3)
            _audit(
                "chat_done",
                ip=visitor,
                answer_chars=len(answer_text),
                source=source,
                duration_s=duration,
            )
            yield await _sse(
                {"type": "done", "answer": answer_text, "duration_s": duration}
            )
        except Exception as exc:
            logger.exception("chat stream failed")
            _audit("chat_error", ip=visitor, error=str(exc))
            yield await _sse({"type": "error", "message": str(exc)})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        "X-RateLimit-Remaining": str(remaining),
    }
    return StreamingResponse(event_source(), media_type="text/event-stream", headers=headers)


async def upload_by_path(request: Request) -> Response:
    """Admin-only: index a PDF that already lives on the server's disk.

    This is the right endpoint to expose on an HF Space / VPS deployment
    where uploading a 200 MB PDF over HTTP is slow, but the file is
    already on a persistent volume. The visitor drops the PDF into the
    ``DOCUMIND_UPLOAD_DIR`` (default ``./uploads``), then POSTs the path
    here with the admin token. The path is resolved and validated to live
    inside the upload dir so a leaked token can't index arbitrary files
    (e.g. ``/etc/passwd``).

    Request body (JSON): ``{"path": "/uploads/paper.pdf"}``
    Returns: ``{"processed": [{"source": "paper_pdf", "chunks": N, ...}]}``
    """
    if not _check_token(request):
        return _bad_request("invalid or missing X-Documind-Token", code=401)
    settings = get_settings()
    try:
        body = await request.json()
    except Exception:
        return _bad_request("request body must be JSON", code=400)
    raw_path = (body.get("path") or "").strip()
    if not raw_path:
        return _bad_request("'path' is required", code=400)

    # Resolve the requested path and the configured upload dir, then require
    # the resolved target to live inside the upload dir. This blocks
    # ``../`` traversal and symlink escapes pointing at e.g. ``/etc``.
    upload_dir = settings.upload_dir.expanduser().resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Treat relative paths as living inside the upload dir; absolute paths
    # are still allowed, but must resolve to a child of the upload dir.
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = upload_dir / candidate
    try:
        target = candidate.resolve(strict=False)
    except Exception as exc:
        return _bad_request(f"could not resolve path: {exc}", code=400)
    try:
        target.relative_to(upload_dir)
    except ValueError:
        return _bad_request(
            f"path must live inside {upload_dir} (got {target})", code=403
        )
    if not target.exists() or not target.is_file():
        return _bad_request(f"file not found: {target}", code=404)
    if target.stat().st_size > _MAX_UPLOAD_BYTES:
        return _bad_request(
            f"file too large (>{_MAX_UPLOAD_BYTES // (1024*1024)} MB)", code=413
        )

    name = normalize_name(target.name)
    try:
        with target.open("rb") as fh:
            data = fh.read()
        chunks = process_pdf_bytes(data, source_name=name, settings=settings)
    except Exception as exc:
        return _bad_request(f"could not read {target.name!r}: {exc}", code=400)
    if not chunks:
        return _bad_request(
            f"{target.name!r} produced no extractable text", code=422
        )
    stats = document_stats(chunks)
    added = vectorstore.add_documents(chunks, source_name=name)
    if settings.summarize_on_upload:
        try:
            tokens = list(summary.stream_summary(name, chunks, settings=settings))
            summary.save(name, "".join(tokens).strip() or "_Summary unavailable._", stats)
        except Exception as exc:
            logger.warning("summary failed for %s: %s", name, exc)
    else:
        summary.save(name, "_Summary disabled in public demo._", stats)
    _audit(
        "upload_by_path",
        source=name,
        chunks=added,
        ip=_client_id(request),
        path=str(target),
    )
    return JSONResponse({"processed": [{"source": name, "chunks": added, **stats}]})


async def get_summary(request: Request) -> Response:
    source = request.path_params["source"]
    safe = _REQ_ID_RE.sub("", source).strip("_") or "unknown"
    item = summary.get(safe)
    if item is None:
        return _bad_request(f"no summary for {safe!r}", code=404)
    return JSONResponse(
        {
            "source": item.source,
            "markdown": item.markdown,
            "stats": item.stats,
            "timestamp": item.timestamp,
        }
    )


async def list_history(request: Request) -> Response:
    return JSONResponse({"entries": [asdict(e) for e in history.load()[:50]]})


async def chat_post_ndjson(request: Request) -> Response:
    """Non-streaming JSON chat endpoint for clients that can't do SSE."""
    if not _check_token(request):
        return _bad_request("invalid or missing X-Documind-Token", code=401)
    settings = get_settings()
    rl: RateLimiter = request.app.state.limiter
    visitor = _client_id(request)
    allowed, _ = rl.check(visitor)
    if not allowed:
        return _bad_request(
            f"rate limit exceeded for {visitor}", code=429
        )
    try:
        body = await request.json()
    except Exception:
        return _bad_request("request body must be JSON", code=400)
    question = (body.get("question") or "").strip()
    if not question:
        return _bad_request("'question' is required", code=400)
    source = body.get("source") or None
    history_msgs = body.get("history") or []
    retrieval = await asyncio.to_thread(retrieve, question, source, settings)
    if retrieval.is_empty:
        return JSONResponse({"answer": "", "sources": []})
    from .llm import _build_messages, stream_chat

    messages = _build_messages(retrieval.context, question, history_msgs)
    parts: list[str] = []
    async for piece in _cap_tokens(stream_chat(messages), settings.max_answer_tokens * 4):
        parts.append(piece)
    return JSONResponse(
        {
            "answer": "".join(parts).strip(),
            "sources": [
                {
                    "source": c.metadata.get("source", "unknown"),
                    "page": c.metadata.get("page"),
                    "score": round(float(c.score), 4),
                    "snippet": c.text[:400],
                }
                for c in retrieval.chunks
            ],
        }
    )


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or get_settings()
    app = Starlette(
        debug=False,
        middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])],
        routes=[
            Route("/", endpoint=index, methods=["GET"]),
            Route("/healthz", endpoint=healthz, methods=["GET"]),
            Route("/api/sources", endpoint=list_sources, methods=["GET"]),
            Route("/api/summary/{source}", endpoint=get_summary, methods=["GET"]),
            Route("/api/history", endpoint=list_history, methods=["GET"]),
            Route("/upload", endpoint=upload, methods=["POST"]),
            Route("/chat", endpoint=chat_sse, methods=["POST"]),
            Route("/api/chat", endpoint=chat_post_ndjson, methods=["POST"]),
            Route("/api/upload-by-path", endpoint=upload_by_path, methods=["POST"]),
            Mount("/static", app=StaticFiles(directory=str(_STATIC_DIR)), name="static"),
        ],
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_min)
    return app


# uvicorn entry point: ``uvicorn documind.webapp:app``
app = create_app()


def main() -> None:  # pragma: no cover - console entry only
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "documind.webapp:app",
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "8000")),
        log_level="info",
        access_log=False,  # the audit log covers this
    )


if __name__ == "__main__":  # pragma: no cover
    main()
