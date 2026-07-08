# Deploying DocuMind

DocuMind ships with two surfaces that share one pipeline:

| Surface | Use it for | Default port |
|---------|-----------|--------------|
| `documind-web` (Starlette/uvicorn) | A long-running public web app. Best for HF Spaces, Render, Fly.io, Cloud Run, your own VPS. Models stay loaded; the LLM streams tokens to the browser via SSE. | `8000` |
| `documind` (Streamlit) | Local single-user exploration. Reruns on every interaction. | `8501` |

`start-web.sh` launches the web app; `streamlit run src/documind/app.py` keeps working for the legacy UI. The web app is the recommended path for any public demo.

## What's the same

Both surfaces call the same `documind.pipeline` code: same chunks, same cross-encoder re-ranker, same Ollama chat model, same guardrail system prompt. The only difference is the HTTP transport.

## What's new in the web app

- **Long-running process** — one Ollama client, one Chroma collection, one cross-encoder in memory. No re-loads.
- **SSE chat stream** — `/chat` pushes a `sources` event, then `token` events, then a `done` event. The browser renders as tokens arrive.
- **PDF upload** — `/upload` accepts one or more PDFs in a single multipart request and indexes them on the spot.
- **Visitor rate limit** — `DOCUMIND_RATE_LIMIT_PER_MIN` caps requests per IP (or first hop in `X-Forwarded-For`). Default `20`; `0` disables.
- **Optional API token** — set `DOCUMIND_API_TOKEN` to require `X-Documind-Token` on `/upload` and `/chat`. Useful when you don't want anonymous traffic.
- **Health probe** — `/healthz` returns 200 if both Ollama and Chroma are reachable, 503 otherwise. The Dockerfile's `HEALTHCHECK` uses it.
- **Audit log** — every upload, chat start, chat end, and error is appended to `.documind/audit.log` as JSON lines.

## Endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/` | GET | The chat UI (single-page app). |
| `/healthz` | GET | Liveness probe. |
| `/api/sources` | GET | List of indexed document names. |
| `/api/summary/{source}` | GET | Cached auto-summary for a document. |
| `/api/history` | GET | Most recent Q&A history. |
| `/upload` | POST | Multipart PDF upload. |
| `/chat` | POST | Server-Sent Events stream of a RAG answer. |
| `/api/chat` | POST | Same as `/chat` but returns a single JSON object. |
| `/api/upload-by-path` | POST | Admin-only: index a PDF that already lives on the server (under `DOCUMIND_UPLOAD_DIR`). |
| `/static/*` | GET | SPA assets. |

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `DOCUMIND_API_TOKEN` | _empty_ | If set, requests must send `X-Documind-Token`. |
| `DOCUMIND_RATE_LIMIT_PER_MIN` | `20` | Per-visitor chat cap; `0` disables. |
| `DOCUMIND_SUMMARIZE_ON_UPLOAD` | `true` | Disable to skip the post-upload LLM pass. |
| `DOCUMIND_MAX_ANSWER_TOKENS` | `1500` | Char cap (×4) per streamed answer. |
| `DOCUMIND_UPLOAD_DIR` | `./uploads` | Admin path-upload reads from this dir only. |
| `DOCUMIND_OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama listens. |
| `DOCUMIND_CHAT_MODEL` | `llama3.2:3b` | Pulled on boot. |
| `DOCUMIND_EMBEDDING_MODEL` | `nomic-embed-text` | Pulled on boot. |

## Server-side path uploads (HF Spaces, VPS)

On a public demo, uploading a 200 MB PDF over HTTPS from a visitor's browser
is slow and may time out. For an HF Space, drop the file into the Space's
persistent storage instead and index it by path:

```bash
# 1. Create the dir on the Space's persistent volume (one time).
mkdir -p /data/uploads

# 2. Drop the PDF there (e.g. via the HF Studio file browser or `huggingface-cli upload`).
cp paper.pdf /data/uploads/paper.pdf

# 3. Index it via the admin endpoint.
curl -X POST https://<your-space>.hf.space/api/upload-by-path \
     -H "X-Documind-Token: $DOCUMIND_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"path": "paper.pdf"}'
```

The endpoint validates the resolved path lives inside `DOCUMIND_UPLOAD_DIR`
(default `./uploads`) and rejects `..` traversal, so a leaked token can't
index arbitrary files. Relative paths resolve against the upload dir.

The SPA exposes a small *Admin sign-in* button in the header that lets you
paste the token client-side; the path-upload box only appears once a token
is set.

---

## Hugging Face Spaces (free, Static — recommended)

A Static Space serves only the HTML/JS/CSS bundle; no container, no
model server, no cold start. The whole RAG pipeline runs in the
visitor's browser via WebAssembly:

```
Browser                    HF CDN
  │                          │
  ├── index.html (3.8 KB)    │  ← static
  ├── styles.css (9.1 KB)    │  ← static
  ├── js/pdf.js (PDF.js)     │  ← static
  ├── js/chunker.js          │  ← static
  ├── js/rag-pipeline.js     │  ← static
  │   └─ Xenova/all-MiniLM-L6-v2 (WASM embedder, ~25 MB)
  │   └─ Xenova/cross-encoder-ms-marco (WASM reranker, ~25 MB)
  ├── js/store.js (IndexedDB) │  ← static
  ├── js/llm.js              │  ← static
  └── js/app.js              │  ← static
         │
         └── only outbound call: → Groq (or any OpenAI-compatible)
                                  chat/completions endpoint
                                  streaming, SSE
```

The total bundle is 60 KB. Models are pulled from Hugging Face's CDN
on first use and cached by the browser. Documents and embeddings live
in IndexedDB on the visitor's machine.

### Steps

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Owner: your account · Space name: `documind` · License: MIT
   - **SDK: Static** → *Blank* template · Hardware: *Static (free)* → **Create Space**

2. **Push the bundle.** A Static Space is a git repo with `index.html` at
   the root. The cleanest path is to push the contents of `deploy/hf-spaces-static/`:

   ```bash
   # clone your new (empty) Space
   git clone https://huggingface.co/spaces/<your-hf-username>/documind
   cd documind

   # copy the SPA from this repo
   cp -r ../rag-app/deploy/hf-spaces-static/* .

   git add -A
   git commit -m "Deploy DocuMind static SPA"
   git push
   ```

3. **Open the Space.** The static page loads in seconds. The first
   PDF you upload triggers a one-time 50 MB model load (the
   embedder + re-ranker), then everything runs locally.

4. **Get a free LLM key.** The visitor (or you) pastes a Groq API
   key in the Admin sign-in panel. Free tier, no credit card, sign
   up at https://console.groq.com. The key lives in the browser's
   localStorage; it's never sent anywhere except the configured
   LLM endpoint.

### Why static, not Docker?

- **Truly free.** Static Spaces are free with no paid tier; Docker
  Spaces on free CPU hardware are slow (cold-start model pulls, no
  GPU, ~10s per first token).
- **Zero cold start.** No container to spin up. The bundle is on
  HF's CDN, the page loads in <1s.
- **No data leaves the browser** except the LLM prompt. Documents
  and embeddings are in IndexedDB; even the cited-context sent to
  the LLM is only the 12 chunks the model actually needs.
- **Caveat:** the LLM call still hits Groq's servers, so this is
  not strictly "local." It's "local for the data, cloud for the
  inference" — the right tradeoff for a public Handshake demo.

### Self-host (any static host)

The bundle is plain static files. Drop the `deploy/hf-spaces-static/`
contents into Netlify, Vercel, GitHub Pages, S3+CloudFront, or a
USB stick. Same code, same behavior. CORS is not an issue because
the Groq call is browser-direct.

---

## Hugging Face Spaces (Docker — the original, kept for reference)

This deploys the whole app — web UI **and** a bundled Ollama server — into a
single Docker Space, so it's reachable from a public URL anywhere.

> **Heads-up on the free tier.** Spaces' free hardware is CPU-only (2 vCPU, 16 GB
> RAM). `llama3.2:3b` runs there but answers are slow, and unless you add
> **persistent storage**, the models re-download on every cold start (a few
> minutes). For a snappier demo, upgrade the Space hardware or attach a persistent
> disk. For production speed you'd point `DOCUMIND_OLLAMA_BASE_URL` at a GPU-backed
> Ollama host instead.

### What's in `deploy/hf-spaces/`

| File | Purpose |
|------|---------|
| `Dockerfile` | Space image: Python + Ollama + the app, listening on port 7860. |
| `start.sh` | Starts Ollama, pulls the models, then launches the web app. |
| `README_SPACE.md` | The Space's `README.md` (contains the required HF metadata header). |

### Steps

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Owner: your account · Space name: `documind` · License: MIT
   - **SDK: Docker** → *Blank* template · Hardware: *CPU basic (free)* → **Create Space**

2. **Get the project into the Space repo.** A Space is a git repo. The simplest way
   is to push this project into it with the Spaces Dockerfile at the repo root:

   ```bash
   # clone your new (empty) Space
   git clone https://huggingface.co/spaces/<your-hf-username>/documind
   cd documind

   # copy the app from your GitHub checkout (adjust the path)
   cp -r ~/rag-app/* ~/rag-app/.streamlit ~/rag-app/.gitignore .

   # HF needs the Dockerfile and README at the repo ROOT:
   cp deploy/hf-spaces/Dockerfile ./Dockerfile
   cp deploy/hf-spaces/README_SPACE.md ./README.md

   git add -A
   git commit -m "Deploy DocuMind to HF Spaces"
   git push
   ```

   (You'll authenticate the push with a Hugging Face access token — create one at
   https://huggingface.co/settings/tokens with *write* scope, and use it as the
   git password.)

3. **Watch it build.** Open the Space → **Logs**. You'll see Ollama start, the
   models download, then the web app launch. When the build finishes, the app
   loads in the Space's **App** tab.

### Optional: faster / persistent

- **Persistent models:** Space → Settings → add **Persistent Storage**. The image
  already writes models to `/data/ollama`, so they'll survive restarts.
- **Bigger/faster hardware:** Space → Settings → Hardware → pick a paid tier.
- **Custom models:** set Space *Variables* `DOCUMIND_CHAT_MODEL` /
  `DOCUMIND_EMBEDDING_MODEL` to change what `start.sh` pulls.

## Local check before deploying

```bash
docker compose up --build           # web app on http://localhost:8000
# or
docker build -f deploy/hf-spaces/Dockerfile -t documind-space .
docker run --rm -p 7860:7860 -v documind_data:/data documind-space
```

## Render.com

1. **New → Web Service → Docker**.
2. Set the **Docker Command** to `./start-web.sh` (the default Dockerfile already
   runs it; you can also pin the port via `PORT=10000` env).
3. Add a **Persistent Disk** mounted at `/data` so the index and history survive
   restarts.
4. Set environment variables (`DOCUMIND_OLLAMA_BASE_URL` etc.).

Render doesn't run Ollama in-process. The simplest layout is two services:
* a free-tier Web Service pointing at **Ollama Cloud** or another Ollama host,
  or
* an "ollama" Background Worker pulling models, plus the web service talking to
  it on a private URL.

## Fly.io

```bash
fly launch --no-deploy
fly volumes create documind_data --size 10
fly secrets set DOCUMIND_OLLAMA_BASE_URL=http://<ollama-host>:11434
fly deploy
```

## Google Cloud Run

Cloud Run is serverless, so each cold start re-pulls Ollama models unless you
mount a disk. For a public demo with hundreds of visitors, HF Spaces' persistent
storage is cheaper. Cloud Run shines if you point `DOCUMIND_OLLAMA_BASE_URL` at
a managed Ollama host (e.g. Anyscale Endpoints) — then the container image is
just the web app, no Ollama inside, and cold starts are seconds.

## What to put on your resume / Handshake

- Public URL of your Space (e.g. `https://huggingface.co/spaces/<you>/documind`).
- The endpoint table above as a *system design* talking point: SSE for streaming,
  rate-limit for fairness, token gate for admin features, persistent disk for
  state.
- The benchmark grid (`make benchmark`) as a *defended default* story:
  "bumping `top_k_rerank` from 8 to 12 lifted faithfulness 0.62 → 0.68 and
  keyword recall 0.75 → 0.88 on our eval set."
