# 📄 DocuMind

[![CI](https://github.com/Sujay1709/documind-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujay1709/documind-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Built with Starlette + Streamlit](https://img.shields.io/badge/built%20with-Starlette-FF4B4B.svg)](https://www.starlette.io/)

**Local, privacy-first RAG assistant — chat with your PDFs entirely on your own machine.**

DocuMind ingests PDF documents, indexes them in a local vector store, and answers
your questions with a local LLM. Nothing leaves your computer: embeddings,
retrieval, re-ranking, and generation all run locally via [Ollama](https://ollama.com)
and [ChromaDB](https://www.trychroma.com).

This is a substantially rebuilt successor to an earlier single-file prototype.
See [What's new](#whats-new) for the differences.

---

## Features

- **Warm, interactive landing page** (beige theme) with an animated hero, feature
  cards, a how-it-works strip, clickable example-question chips, and live stat chips.
- **Live processing feedback** — a step-by-step status (reading → chunking →
  embedding → summarizing) while a document is indexed.
- **Auto document briefing** — on processing, the model generates an overview, a
  table of contents, and key highlights, shown as the opening chat message; each
  document also displays page/word/chunk stats. Briefings are cached per document.
- **Large uploads** — PDFs up to **500 MB** per file (configurable).
- **Hardened, document-only answers** — the model is instructed to answer solely
  from your uploaded documents, to treat retrieved text as untrusted data (so
  instructions hidden inside a PDF can't hijack it), and never to reveal its
  system prompt. Everything runs locally, so no document data leaves your machine.
- **Chat interface** with conversation history and follow-up questions.
- **Persistent history store** — every question, answer, and its sources are saved
  to disk and browsable from the sidebar, surviving app restarts.
- **Multiple PDFs** indexed at once, with per-document tracking and a "clear" control.
- **Cross-encoder re-ranking** so the most relevant chunks (not just the most
  similar embeddings) are sent to the model.
- **Source citations** — every answer shows the chunks, source file, page, and
  relevance score it was grounded in.
- **Streaming answers** rendered token-by-token.
- **Fully local & private** — uses Ollama for both embeddings and generation.
- **Configurable** via environment variables (chunk size, models, retrieval depth…).
- **RAG evaluation harness** — measure retrieval (hit@k, recall, MRR) and answer
  quality (token-F1, keyword recall, faithfulness, optional LLM judge). See
  [EVAL.md](EVAL.md).
- **Multi-config benchmark** — same dataset, isolated config overrides, side-by-side
  Markdown report. Use it to defend choices like chunk size and top_k. Run with
  `make benchmark` (see [`scripts/benchmark.py`](scripts/benchmark.py)).
- **Public web app (`documind-web`)** — a long-running Starlette/uvicorn server
  that keeps the LLM warm between requests, exposes `/upload` (multipart PDF),
  `/chat` (SSE stream), `/api/chat` (single JSON), `/healthz`, and a single-page
  app at `/`. Per-visitor rate limit, optional `DOCUMIND_API_TOKEN` gate, JSON-line
  audit log, and a Dockerfile that bundles Ollama for one-command deploys.
- **Tested** core pipeline, with linting, Docker, and CI included.
- **Live smoke test** — every PR runs `scripts/smoke_webapp.py` against a
  freshly-booted uvicorn process (see `.github/workflows/smoke.yml`) so a
  broken route surfaces before deploy. Catch the "did the import break?"
  regressions that pure unit tests miss.

## Public web app (recommended)

DocuMind ships a long-running **Starlette + uvicorn** web app (`documind-web`) that keeps the LLM and vector store warm between requests. It's the right surface for HF Spaces, Render, Fly.io, Cloud Run, or your own VPS.

```bash
# Local
./start-web.sh                                 # http://localhost:8000

# Container (Ollama bundled)
docker compose up --build                      # http://localhost:8000

# Hugging Face Space — see DEPLOY.md
```

Endpoints: `/` (SPA), `/upload` (multipart PDF), `/chat` (SSE stream), `/api/chat` (single JSON), `/healthz`, `/api/sources`. The same `documind.pipeline` powers both surfaces, so the Streamlit app stays available for local use.

See **[DEPLOY.md](DEPLOY.md)** for HF Spaces, Render, Fly.io, and Cloud Run steps.

---

## Using DocuMind (local Streamlit UI)

- **Upload & process** one or more PDFs from the sidebar to index them.
- **Ask** questions in the chat; answers stream in with an expandable Sources
  panel (file, page, relevance score).
- **Revisit a document:** click any indexed document in the sidebar to reopen it
  with its related chat history restored. While a document is open, questions are
  scoped to just that file. Hit **Exit document** to return to all-document chat.
- **History:** every interaction is saved and browsable from the sidebar History
  panel, surviving restarts.

## Screenshots

![Landing](docs/landing.png)

![Chat with sources](docs/chat.png)

These are baked from `docs/landing.svg` and `docs/chat.svg` so the README is
never broken when the app isn't running. To replace them with screenshots
of the live app, follow [`docs/README.md`](docs/README.md).

## Architecture

```
PDF ──▶ ingestion ──▶ chunks ──▶ embeddings (Ollama) ──▶ Chroma vector store
                                                              │
question ─────────────────────────────────────────▶ similarity search
                                                              │
                                              cross-encoder re-ranking
                                                              │
                                          top-k context ──▶ LLM (Ollama) ──▶ streamed answer + citations
```

| Module | Responsibility |
|--------|----------------|
| `config.py` | Typed, env-driven settings (Pydantic). |
| `ingestion.py` | Load PDFs and split into overlapping, source-tagged chunks. |
| `vectorstore.py` | Chroma collection: upsert, query, list sources, reset. |
| `reranker.py` | Cross-encoder re-ranking of retrieved candidates. |
| `llm.py` | Prompt construction and streaming generation via Ollama. |
| `pipeline.py` | Orchestrates retrieve → re-rank → generate. |
| `app.py` | Streamlit chat UI. |

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/download)** running locally, with the models pulled:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

## Quick start

```bash
git clone https://github.com/Sujay1709/documind-rag.git
cd documind-rag

python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt

streamlit run src/documind/app.py
```

Open http://localhost:8501, upload one or more PDFs, click **Process documents**,
then ask questions.

Looking for the resume / interview narrative? See [`RESUME.md`](RESUME.md).

## Configuration

Copy `.env.example` to `.env` and override any of the settings (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMIND_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DOCUMIND_CHAT_MODEL` | `llama3.2:3b` | Generation model |
| `DOCUMIND_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `DOCUMIND_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `DOCUMIND_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DOCUMIND_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DOCUMIND_N_RESULTS` | `30` | Candidates retrieved before re-ranking |
| `DOCUMIND_TOP_K_RERANK` | `12` | Chunks kept after re-ranking |
| `DOCUMIND_PERSIST_DIR` | `./.documind/chroma` | Vector store location |
| `DOCUMIND_HISTORY_FILE` | `./.documind/history.json` | Persistent Q&A history file |
| `DOCUMIND_API_TOKEN` | _empty_ | If set, `/upload` and `/chat` require `X-Documind-Token` |
| `DOCUMIND_RATE_LIMIT_PER_MIN` | `20` | Per-visitor chat cap; `0` disables |
| `DOCUMIND_SUMMARIZE_ON_UPLOAD` | `true` | Skip the post-upload summary in public demos |
| `DOCUMIND_MAX_ANSWER_TOKENS` | `1500` | Char cap (×4) per streamed answer |
| `DOCUMIND_UPLOAD_DIR` | `./uploads` | Admin path-upload reads from this dir only |
| `DOCUMIND_MAX_HISTORY` | `200` | Max history entries retained |
| `DOCUMIND_MAX_UPLOAD_MB` | `500` | Upload size shown in the UI (keep in sync with the server limit below) |

### Upload size

PDFs can be up to **500 MB** per file. This is enforced by Streamlit via
`maxUploadSize` in [`.streamlit/config.toml`](.streamlit/config.toml). To change
it, update **both** that value and `DOCUMIND_MAX_UPLOAD_MB` (the latter only sets
the size shown in the upload section).

### Security & privacy

DocuMind is built so your document data can't leak:

- **Local-only.** Embeddings, retrieval, re-ranking, and generation all run on
  your machine through Ollama — no document content is sent to any external API.
- **Document-grounded answers.** The model is instructed to answer *only* from the
  retrieved context and to say it doesn't know otherwise — it won't pad answers
  with outside knowledge.
- **Prompt-injection resistant.** Retrieved text is wrapped and labelled as
  *untrusted data*, and the system prompt tells the model to ignore any embedded
  instructions (e.g. "ignore previous instructions", "reveal your prompt",
  "send this data somewhere").
- **No prompt disclosure.** The model is told never to reveal its system prompt or
  configuration.

## Run with Docker

```bash
docker compose up --build
# then pull models into the ollama container:
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

App: http://localhost:8501

## Deploy to the cloud (free)

DocuMind can run as a public Hugging Face Space (Docker) with Ollama bundled in
the container — see **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions. The
Space files live in [`deploy/hf-spaces/`](deploy/hf-spaces/).

## Development

```bash
pip install -e ".[dev]"
make test     # run the test suite
make lint     # ruff
make fmt      # auto-format + fix
```

## What's new

Compared with the original single-file prototype, DocuMind:

- **Fixes a broken re-ranker.** The original passed a single concatenated
  string to the cross-encoder and referenced an undefined global `prompt`; it
  could not actually rank anything. Re-ranking now takes an explicit query and a
  list of chunks.
- **Adds real citations.** Answers are grounded in displayed source/page/score.
- **Supports multiple documents** and a chat history for follow-ups.
- **Splits a 140-line script into a tested, typed package** with config,
  Docker, and CI.
- **Externalises configuration** instead of hard-coding URLs and model names.

## License

MIT — see [LICENSE](LICENSE).
