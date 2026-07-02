---
title: DocuMind
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DocuMind — chat with your PDFs (public demo)

A long-running RAG service on top of [Ollama](https://ollama.com) +
[ChromaDB](https://www.trychroma.com) + a cross-encoder re-ranker. The Space
serves a single-page web app at `/`, an SSE chat stream at `/chat`, and a
multipart PDF upload at `/upload`. Index and history persist to `/data` so
they survive restarts.

> First boot downloads the language and embedding models, so the Space may take
> a few minutes to become responsive. On free CPU hardware, answers are slower
> than on a local GPU — this Space is intended as a public demo.

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/` | GET | The chat UI (single-page app). |
| `/healthz` | GET | Liveness probe; 200 if Ollama + Chroma are both reachable. |
| `/api/sources` | GET | List of indexed document names. |
| `/api/summary/{source}` | GET | Cached auto-summary for a document. |
| `/api/history` | GET | Most recent Q&A history (JSON). |
| `/upload` | POST | Multipart PDF upload; persists chunks into Chroma. |
| `/chat` | POST | Server-Sent Events stream of a RAG answer. |
| `/api/chat` | POST | Same as `/chat` but returns a single JSON object. |

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `DOCUMIND_API_TOKEN` | _empty_ | If set, requests must send `X-Documind-Token`. |
| `DOCUMIND_RATE_LIMIT_PER_MIN` | `20` | Per-visitor chat cap; `0` disables. |
| `DOCUMIND_SUMMARIZE_ON_UPLOAD` | `true` | Disable to skip the post-upload LLM pass. |
| `DOCUMIND_CHAT_MODEL` | `llama3.2:3b` | Pulled on boot. |
| `DOCUMIND_EMBEDDING_MODEL` | `nomic-embed-text` | Pulled on boot. |

Source & docs: https://github.com/Sujay1709/documind-rag
