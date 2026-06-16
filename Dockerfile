FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for PyMuPDF / sentence-transformers builds are bundled in wheels,
# so only a minimal image is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8501

# Ollama is expected to run on the host / another container; point at it via
# DOCUMIND_OLLAMA_BASE_URL (see docker-compose.yml).
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/documind/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
