#!/usr/bin/env bash
# Convenience launcher for the long-running web app. Use this on a server so
# you don't have to remember the uvicorn flags.
#
#   ./start-web.sh                       # bind 0.0.0.0:8000
#   PORT=7860 ./start-web.sh             # HF Spaces default port
#   DOCUMIND_API_TOKEN=secret ./start-web.sh
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

cd "$(dirname "$0")"

# Best-effort: start Ollama if it's installed and not already running.
if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS "${DOCUMIND_OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
    echo "▶ Starting Ollama…"
    ollama serve >/tmp/documind-ollama.log 2>&1 &
    OLLAMA_PID=$!
    for _ in $(seq 1 30); do
      if curl -fsS "${DOCUMIND_OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi
  # Pull models the user asked for (idempotent).
  for m in "${DOCUMIND_CHAT_MODEL:-llama3.2:3b}" "${DOCUMIND_EMBEDDING_MODEL:-nomic-embed-text}"; do
    ollama pull "$m" || echo "  (could not pull $m — continuing)"
  done
fi

echo "▶ DocuMind web app on http://${HOST}:${PORT}"
exec uvicorn documind.webapp:app \
  --host "$HOST" --port "$PORT" --workers "$WORKERS" \
  --log-level info
