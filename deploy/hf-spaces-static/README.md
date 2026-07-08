---
title: DocuMind
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
---

# DocuMind — chat with your PDFs

Privacy-first RAG that runs entirely in your browser:

- **PDF parsing** — PDF.js extracts per-page text.
- **Embeddings** — Xenova/all-MiniLM-L6-v2 in WebAssembly, same model family as the server.
- **Vector search** — cosine over the embedded chunks, top-30 candidates.
- **Re-ranking** — Xenova/cross-encoder-ms-marco-MiniLM-L-6-v2 in WASM, keeps the top 12.
- **Generation** — streaming chat completions to Groq (Llama 3.1 8B) or any OpenAI-compatible endpoint.

Your PDF, the chunks, and the embeddings stay in your browser's IndexedDB. The only network call is the LLM completion, which carries the question and the cited context, not the document text.

## How to use

1. Open the Space.
2. Click **Admin sign-in** in the header and paste a free Groq API key (no credit card, sign up at <https://console.groq.com>).
3. Drop a PDF, wait for the first-time model load (~50 MB, cached after), and ask questions.

## Endpoints

This is a static site — there is no server. All processing happens in your browser. The only outbound call is the LLM streaming completion.

## Self-host

To deploy your own copy:

1. Create a new HF Space: <https://huggingface.co/new-space>
2. SDK = **Static**, blank template, MIT license.
3. Push this directory to the Space repo.
4. Share the URL.

Source & docs: <https://github.com/Sujay1709/documind-rag>
