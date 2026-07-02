# DocuMind — resume and interview kit

How to translate this project into resume bullets, project blurbs, and
interview narratives. Everything here is grounded in the actual codebase.

## One-line pitch (for Handshake / LinkedIn headline)

> Local-first RAG assistant for PDFs — privacy, citations, a custom eval
> harness, and a long-running web service that anyone can hit over HTTP.

## Resume bullets (pick two or three, not all)

1. **Built a local-first RAG assistant for PDF Q&A.** Designed a typed,
   tested Python package (`config → ingestion → vector store → reranker →
   pipeline → LLM`) using Ollama for embeddings and generation and ChromaDB
   for retrieval. Streams answers token-by-token with per-chunk source
   citations and a confidence score. Nothing leaves the user's machine.

2. **Shipped a cross-encoder re-ranking stage to ground answers in retrieved
   evidence.** Pulled 30 candidate chunks from the vector store, scored each
   (query, chunk) pair jointly with `cross-encoder/ms-marco-MiniLM-L-6-v2`,
   kept the top 12, and re-ordered them in document position before
   generation. The 12-chunk default was chosen from a side-by-side benchmark
   over `chunk_size` and `top_k_rerank` (faithfulness 0.62 → 0.68, keyword
   recall 0.75 → 0.88 on the eval set). Result: stable answers on multi-page
   questions (tables of contents, step lists) that previously came out jumbled.

3. **Wrote a custom RAG evaluation harness** measuring both retrieval
   (hit@k, recall, MRR) and answer quality (token-F1, keyword recall,
   faithfulness, optional LLM judge). Same dataset, isolated config overrides,
   side-by-side Markdown report — used to compare `chunk_size` and
   `top_k_rerank` settings before committing defaults.

4. **Hardened the system prompt for prompt-injection resistance.** Retrieved
   text is wrapped in clear delimiters and labelled untrusted data; the model
   is instructed to refuse embedded commands ("ignore previous instructions",
   "reveal your prompt", "send data anywhere"). Documented in code comments.

5. **Containerized and deployed to Hugging Face Spaces.** Single Docker image
   starts Ollama, pulls both models, and serves Streamlit on port 7860 with
   500 MB PDF uploads. CI runs the pure-Python metric tests on every push;
   generative eval is opt-in via `workflow_dispatch`.

6. **Built a long-running public web service on top of the same RAG pipeline.**
   Starlette + uvicorn shell with `/upload` (multipart PDF), `/chat` (SSE
   stream of tokens + sources), `/api/chat` (single JSON), and a `/healthz`
   probe. Added per-visitor rate limiting, an optional `DOCUMIND_API_TOKEN`
   gate, and a JSON-line audit log so a public Space can run safely behind
   a proxy. Same Dockerfile now bundles Ollama for one-command deploys to
   HF Spaces, Render, or Fly.io.

7. **Shipped a public-deployed Hugging Face Space with a server-side PDF
   upload endpoint and a CI smoke test.** Added `/api/upload-by-path` for
   indexing PDFs that already live on the Space's persistent volume
   (uploaded once via `huggingface-cli`, not 200 MB over HTTPS per visitor),
   with a `..` traversal guard. Wired `scripts/smoke_webapp.py` into a
   dedicated GitHub Actions workflow that boots uvicorn on a free port and
   hits every public route on every PR — catches the "import broke" and
   "route shape changed" regressions that pure unit tests miss.

## Project description (for Handshake / portfolio site)

**DocuMind** is a local, privacy-first RAG assistant for PDFs. You upload a
document, it indexes it on your machine, and you can chat with it — every
answer is grounded in cited passages and streamed back with the exact file,
page, and relevance score it came from. The full pipeline (PDF parsing,
embedding, vector retrieval, cross-encoder re-ranking, generation) runs
locally through Ollama and ChromaDB, so document data never leaves the
computer. A built-in evaluation harness compares retrieval and answer-quality
metrics across configurations, and the whole stack is containerized and
deployable to Hugging Face Spaces.

## Interview narratives

### 1. The 30-second walkthrough

"DocuMind is a local-first RAG app for PDFs. The architecture is a textbook
retrieval stack: PyMuPDF loads each page, a recursive text splitter turns
those into overlapping chunks, Ollama's `nomic-embed-text` embeds them, and
they live in a persistent Chroma collection. When you ask a question, I pull
30 candidates from the vector store, rerank them with a cross-encoder, send
the top 8 to a local Llama model with a hardened system prompt, and stream
the answer back with citations. The interesting part is the eval harness —
most RAG projects skip it, but it's the only way to actually defend changes
to the retrieval config."

### 2. The "tell me about a hard problem" story

**Problem.** Early on, answers about multi-page questions (e.g. "what are the
sections of this textbook?") came out in the wrong order — the model got the
right chunks but presented them out of sequence.

**Diagnosis.** I added a quick eval run and noticed `mrr` was fine but
qualitative answers were jumbled. Looking at the code, the cross-encoder was
returning chunks in *relevance* order, which is great for selection but bad
for sequential questions.

**Fix.** In `pipeline.retrieve`, I sort the reranked chunks by `(page, chunk_index)`
after re-ranking, so selection stays relevance-based but presentation order
matches the source. The fix is two lines plus a comment explaining the
trade-off. Hit rate didn't change, but answer quality for TOCs and numbered
steps noticeably improved — and the eval harness let me confirm it
quantitatively next time.

### 3. The "how do you evaluate RAG?" answer

"Two halves. **Retrieval** — did we fetch the right passages? I measure
hit@k, source recall, and MRR using the indexed document names. **Answer
quality** — is the generated answer correct and grounded? I compute token-F1
against a reference answer, keyword recall for must-have terms, and a
faithfulness heuristic that checks what fraction of the answer's content
words actually appear in the retrieved context. The metrics are pure Python
so they run in CI without a model server; I also have an optional LLM judge
that asks the local chat model to score groundedness 1–5. The point is that
`did retrieval work` and `does the answer look right` are different
questions and need different signals."

### 4. The "why privacy-first?" answer

"Because the documents people upload to RAG assistants tend to be the
documents they *shouldn't* be uploading to someone else's server — resumes,
internal reports, contracts, study material. The whole architecture decision
follows from that: Ollama instead of OpenAI, Chroma on disk instead of a
managed vector DB, Streamlit for a single-user local UI, and a single Docker
image so the same code that runs locally runs in the cloud. The
prompt-injection hardening is the second half of that story: even inside
that trust boundary, retrieved text is untrusted data, and the model is
instructed to refuse embedded commands."

### 5. The "what would you change next?" answer

"If I had another week I'd add: (1) BM25 hybrid retrieval alongside
embeddings, because pure vector search misses exact-term matches on jargon;
(2) a tiny labeling UI so the eval dataset can grow without hand-editing
JSON; and (3) async ingestion — chunking and embedding the first 100
chunks while the rest of the PDF is still being parsed. All three are
follow-ups the eval harness would actually let me measure."

## Talking points you should be ready to defend

- **Why chunk size matters.** Tiny chunks fragment sentences and lose
  context; oversized chunks dilute attention and waste context window. The
  default `chunk_size=1000` is a compromise; the benchmark grid shows the
  trade-off.
- **Why a cross-encoder.** Embedding similarity is fast but coarse — it
  measures "near this concept" not "answers this question". A cross-encoder
  jointly scores (query, chunk) and is much more accurate at the cost of
  latency.
- **Why `num_ctx` is set to 8192 in config.** Ollama silently truncates past
  the model's default context window, which dropped retrieved chunks mid-
  prompt and produced half-answers on long documents. Bumping `num_ctx` was
  the fix.
- **Why citations.** For grounded answers, showing the user *which chunk*
  answered the question is what turns a chat into a verifiable claim.
- **Why `make eval` is a CI step.** Metrics-as-tests catch regressions when
  you change embeddings, chunking, or the system prompt.

## Common recruiter questions

- **"Did you build this alone?"** Yes — design, code, eval, and deployment.
- **"What's the tech stack?"** Python, Streamlit, LangChain loaders,
  ChromaDB, sentence-transformers, Ollama (Llama 3.2 3B + nomic-embed-text),
  Pydantic settings, pytest, Ruff, Docker, GitHub Actions.
- **"Can I try it?"** `git clone`, `pip install -e .`, `ollama pull llama3.2:3b`,
  `ollama pull nomic-embed-text`, `streamlit run src/documind/app.py`.
- **"What's the impact?"** Any user with a sensitive PDF can use a competent
  RAG tool without trusting a third party with their documents.
