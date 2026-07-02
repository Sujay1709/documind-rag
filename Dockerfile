FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used for the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Default to the long-running web app (Starlette/uvicorn). Override with
# `command:` in docker-compose.yml to keep using the Streamlit UI.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/healthz || exit 1

CMD ["./start-web.sh"]
