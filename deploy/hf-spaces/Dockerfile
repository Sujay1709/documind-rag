# Hugging Face Spaces (Docker SDK) image for DocuMind.
# Bundles Ollama + the Streamlit app in a single container that listens on 7860
# (the port Spaces expects). Models are pulled on first boot by start.sh.
#
# Build context = repository root. From the repo root:
#   docker build -f deploy/hf-spaces/Dockerfile -t documind-space .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep all model/app data under a writable, persistable location.
    HOME=/data \
    OLLAMA_MODELS=/data/ollama \
    DOCUMIND_OLLAMA_BASE_URL=http://localhost:11434 \
    DOCUMIND_PERSIST_DIR=/data/.documind/chroma \
    DOCUMIND_HISTORY_FILE=/data/.documind/history.json \
    HF_HOME=/data/hf

# curl is used to install Ollama and for health checks.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (provides the `ollama` binary + server).
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Spaces run as a non-root user (uid 1000) with /data writable.
RUN mkdir -p /data/ollama /data/.documind /data/hf && chmod -R 777 /data

EXPOSE 7860

COPY deploy/hf-spaces/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
