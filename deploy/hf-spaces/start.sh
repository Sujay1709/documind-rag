#!/usr/bin/env bash
# Entrypoint for the Hugging Face Space: start Ollama, ensure models exist,
# then launch the long-running web app on the port Spaces expects (7860).
set -euo pipefail

CHAT_MODEL="${DOCUMIND_CHAT_MODEL:-llama3.2:3b}"
EMBED_MODEL="${DOCUMIND_EMBEDDING_MODEL:-nomic-embed-text}"
PORT="${PORT:-7860}"

echo "▶ Starting Ollama server…"
ollama serve &

echo "▶ Waiting for Ollama to be ready…"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✓ Ollama is up."
    break
  fi
  sleep 2
done

echo "▶ Ensuring models are available (first boot can take a few minutes)…"
ollama pull "$CHAT_MODEL"  || echo "⚠ could not pull $CHAT_MODEL"
ollama pull "$EMBED_MODEL" || echo "⚠ could not pull $EMBED_MODEL"

echo "▶ Launching DocuMind web app on :$PORT…"
exec uvicorn documind.webapp:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --log-level info
