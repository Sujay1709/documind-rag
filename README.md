# 📄 DocuMind

**Local, privacy-first RAG assistant — chat with your PDFs entirely on your own machine.**

DocuMind ingests PDF documents, indexes them in a local vector store, and answers
your questions with a local LLM. Nothing leaves your computer: embeddings,
retrieval, re-ranking, and generation all run locally via [Ollama](https://ollama.com)
and [ChromaDB](https://www.trychroma.com).

This is a substantially rebuilt successor to an earlier single-file prototype.
See [What's new](#whats-new) for the differences.

---

## Features

- **Chat interface** with persistent conversation history and follow-up questions.
- **Multiple PDFs** indexed at once, with per-document tracking and a "clear index" control.
- **Cross-encoder re-ranking** so the most relevant chunks (not just the most
  similar embeddings) are sent to the model.
- **Source citations** — every answer shows the chunks, source file, page, and
  relevance score it was grounded in.
- **Streaming answers** rendered token-by-token.
- **Fully local & private** — uses Ollama for both embeddings and generation.
- **Configurable** via environment variables (chunk size, models, retrieval depth…).
- **Tested** core pipeline, with linting, Docker, and CI included.

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
git clone https://github.com/USERNAME/documind-rag.git
cd documind-rag

python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt

streamlit run src/documind/app.py
```

Open http://localhost:8501, upload one or more PDFs, click **Process documents**,
then ask questions.

## Configuration

Copy `.env.example` to `.env` and override any of the settings (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMIND_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DOCUMIND_CHAT_MODEL` | `llama3.2:3b` | Generation model |
| `DOCUMIND_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `DOCUMIND_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `DOCUMIND_CHUNK_SIZE` | `400` | Characters per chunk |
| `DOCUMIND_CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `DOCUMIND_N_RESULTS` | `10` | Candidates retrieved before re-ranking |
| `DOCUMIND_TOP_K_RERANK` | `3` | Chunks kept after re-ranking |
| `DOCUMIND_PERSIST_DIR` | `./.documind/chroma` | Vector store location |

## Run with Docker

```bash
docker compose up --build
# then pull models into the ollama container:
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

App: http://localhost:8501

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
