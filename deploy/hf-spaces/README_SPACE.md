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

# DocuMind — chat with your PDFs

Local-style RAG (Ollama + ChromaDB + cross-encoder re-ranking) packaged to run on
a Hugging Face Space. Upload PDFs, ask questions, get answers with citations.

> First boot downloads the language and embedding models, so the Space may take a
> few minutes to become responsive. On free CPU hardware, answers are slower than
> on a local GPU — this Space is intended as a public demo.

Source & docs: https://github.com/Sujay1709/documind-rag
