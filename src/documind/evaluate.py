"""Command-line entry point for the RAG evaluation harness.

Examples:
    documind-eval --dataset eval/sample_dataset.json
    documind-eval --dataset eval/sample_dataset.json --source report_pdf --judge \\
        --out eval/report.json

The documents referenced by your dataset must already be indexed (upload them in
the app, or via the ingestion API) and Ollama must be running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import EvalReport, evaluate_dataset, load_dataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="documind-eval", description="Evaluate DocuMind RAG quality.")
    p.add_argument("--dataset", required=True, help="Path to a JSON eval dataset.")
    p.add_argument("--source", default=None, help="Scope retrieval to this indexed document.")
    p.add_argument("--judge", action="store_true", help="Also score with the LLM judge.")
    p.add_argument("--out", default=None, help="Write the full JSON report to this path.")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> EvalReport:
    args = _parse_args(argv)
    samples = load_dataset(args.dataset)
    report = evaluate_dataset(samples, source=args.source, use_llm_judge=args.judge)

    print(report.to_markdown())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull report written to {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        run(argv)
    except FileNotFoundError as exc:
        print(f"Dataset not found: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
