# Evaluating DocuMind (RAG evaluation harness)

DocuMind ships with a small, dependency-free harness that measures both halves of
a RAG system: **did retrieval find the right passages**, and **is the generated
answer correct and grounded**.

## Metrics

**Retrieval** (need `expected_sources` in your dataset)
- `retrieval_hit` — 1 if any expected source was retrieved, else 0.
- `retrieval_recall` — fraction of expected sources retrieved.
- `mrr` — mean reciprocal rank of the first relevant source.

**Answer quality**
- `answer_f1` — token-level F1 vs. your reference answer (`ground_truth`).
- `keyword_recall` — fraction of `expected_keywords` present in the answer.
- `faithfulness` — share of the answer's content words found in the retrieved
  context (a free hallucination proxy; higher = better grounded).
- `llm_judge` *(optional, `--judge`)* — the local chat model grades groundedness
  1–5, normalised to 0–1.

All retrieval/answer metric math is pure Python and unit-tested, so it runs in CI
without a model server. Generating answers (and the LLM judge) needs Ollama.

## Dataset format

A JSON array; every field except `question` is optional:

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": "Paris is the capital of France.",
    "expected_sources": ["geography_pdf"],
    "expected_keywords": ["Paris"]
  }
]
```

`expected_sources` use the **normalised** indexed name (dots/dashes/spaces become
underscores), e.g. `geography.pdf` → `geography_pdf`. See the sidebar's
"Indexed documents" list for exact names.

## Running it

1. Index the documents your dataset asks about (upload them in the app, or via the
   ingestion API) and make sure Ollama is running.
2. Run the harness:

```bash
# all documents
documind-eval --dataset eval/sample_dataset.json --out eval/report.json

# scope retrieval to a single document
documind-eval --dataset eval/sample_dataset.json --source geography_pdf

# also grade with the local LLM judge
documind-eval --dataset eval/sample_dataset.json --judge
```

It prints a Markdown summary and, with `--out`, writes a full JSON report
(per-question answers, retrieved sources, and all metrics).

`make eval` runs it against the bundled sample dataset.

## In CI / the cloud

The metric unit tests run automatically in GitHub Actions. A full generative eval
needs Ollama, so it isn't part of the default PR pipeline; a manual
**workflow_dispatch** job (`.github/workflows/eval.yml`) is provided to run it on
demand. You can also run `documind-eval` inside the deployed Hugging Face Space
(it already has Ollama and the models).
