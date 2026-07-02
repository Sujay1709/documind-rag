.PHONY: install dev run test lint fmt clean benchmark eval

install:
	pip install -r requirements.txt && pip install -e .

dev:
	pip install -e ".[dev]"

run:
	streamlit run src/documind/app.py

# Long-running web app (Starlette/uvicorn). See ./start-web.sh for the
# container-friendly version that also boots Ollama.
web:
	uvicorn documind.webapp:app --host 0.0.0.0 --port $${PORT:-8000}

test:
	pytest -q

# Live HTTP smoke test against a running uvicorn process (set DOCUMIND_BASE_URL).
smoke:
	python scripts/smoke_webapp.py

# Single-config RAG evaluation (writes eval/report.json).
eval:
	documind-eval --dataset eval/datasets/realistic.json --out eval/report.json

# Multi-config comparison grid (writes eval/benchmark.{json,md}).
# Needs Ollama running with the configured models pulled.
benchmark:
	python scripts/benchmark.py --dataset eval/datasets/realistic.json --out eval/benchmark.json --markdown eval/benchmark.md

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .documind
