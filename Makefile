.PHONY: install dev run test lint fmt clean

install:
	pip install -r requirements.txt && pip install -e .

dev:
	pip install -e ".[dev]"

run:
	streamlit run src/documind/app.py

test:
	pytest -q

eval:
	documind-eval --dataset eval/sample_dataset.json --out eval/report.json

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .documind
