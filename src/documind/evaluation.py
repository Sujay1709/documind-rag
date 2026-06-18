"""A lightweight, dependency-free RAG evaluation harness.

It scores two things that matter for a retrieval-augmented system:

* **Retrieval quality** — did we fetch the right document/passages?
  (hit@k, source recall, mean reciprocal rank)
* **Answer quality** — is the generated answer correct and grounded?
  (token-F1 vs. a reference answer, keyword recall, and a faithfulness
  heuristic measuring how much of the answer is supported by the context)

All core metrics are pure Python so they can be unit-tested without a model
server. An optional LLM-judge (using the local Ollama chat model) can score
correctness/faithfulness on a 1–5 scale when you want a model-graded view.

Run it with the ``documind-eval`` CLI (see ``documind/evaluate.py``).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .pipeline import answer as pipeline_answer

logger = logging.getLogger(__name__)

# A small stopword set so faithfulness/keyword metrics focus on content words.
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "as", "at", "by", "with", "from", "into", "about", "than", "then",
    "so", "such", "but", "if", "while", "do", "does", "did", "has", "have",
    "had", "i", "you", "he", "she", "they", "we", "what", "which", "who", "how",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class EvalSample:
    """One evaluation item.

    ``ground_truth`` and the ``expected_*`` fields are all optional — provide
    whatever you have; metrics that need a missing field are simply skipped.
    """

    question: str
    ground_truth: str = ""
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


def load_dataset(path: str | Path) -> list[EvalSample]:
    """Load an eval dataset from a JSON array of sample objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in raw]


# --------------------------------------------------------------------------- #
# Core metrics (pure functions)
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a predicted and reference answer (SQuAD-style)."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords/phrases present in ``text``."""
    if not keywords:
        return 0.0
    low = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords)


def faithfulness(answer: str, context: str) -> float:
    """Heuristic groundedness: share of answer content-words found in context.

    A low score flags answers that introduce terms absent from the retrieved
    context (a proxy for hallucination). Not a substitute for the LLM judge,
    but useful and free.
    """
    ans = content_tokens(answer)
    if not ans:
        return 0.0
    ctx = set(content_tokens(context))
    supported = sum(1 for t in ans if t in ctx)
    return supported / len(ans)


def retrieval_hit(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """1.0 if any expected source appears among retrieved sources, else 0.0."""
    if not expected_sources:
        return 0.0
    retrieved = set(retrieved_sources)
    return 1.0 if any(s in retrieved for s in expected_sources) else 0.0


def retrieval_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """Fraction of expected sources that were retrieved."""
    if not expected_sources:
        return 0.0
    retrieved = set(retrieved_sources)
    found = sum(1 for s in expected_sources if s in retrieved)
    return found / len(expected_sources)


def reciprocal_rank(expected_sources: list[str], ranked_sources: list[str]) -> float:
    """1/rank of the first relevant source in the ranked list (MRR component)."""
    if not expected_sources:
        return 0.0
    expected = set(expected_sources)
    for idx, src in enumerate(ranked_sources, start=1):
        if src in expected:
            return 1.0 / idx
    return 0.0


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class SampleResult:
    question: str
    answer: str
    retrieved_sources: list[str]
    metrics: dict[str, float]


@dataclass
class EvalReport:
    results: list[SampleResult]
    aggregate: dict[str, float]
    config: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "aggregate": self.aggregate,
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = ["# DocuMind RAG evaluation", ""]
        lines.append(f"Samples: **{len(self.results)}**")
        if self.config:
            cfg = ", ".join(f"{k}=`{v}`" for k, v in self.config.items())
            lines.append(f"Config: {cfg}")
        lines += ["", "## Aggregate metrics", "", "| Metric | Score |", "| --- | --- |"]
        for k, v in self.aggregate.items():
            lines.append(f"| {k} | {v:.3f} |")
        lines += ["", "## Per-question", ""]
        lines += ["| Question | F1 | Faithful | Hit |", "| --- | --- | --- | --- |"]
        for r in self.results:
            q = (r.question[:50] + "…") if len(r.question) > 50 else r.question
            m = r.metrics
            lines.append(
                f"| {q} | {m.get('answer_f1', 0):.2f} | "
                f"{m.get('faithfulness', 0):.2f} | {m.get('retrieval_hit', 0):.0f} |"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional LLM judge
# --------------------------------------------------------------------------- #
def llm_judge(question: str, answer: str, context: str, settings: Settings | None = None) -> float:
    """Ask the local chat model to grade groundedness 1–5, normalised to 0–1.

    Returns 0.0 if the model is unavailable or the response can't be parsed, so
    a missing Ollama server never crashes an eval run.
    """
    settings = settings or get_settings()
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_base_url)
        resp = client.chat(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict grader. Given a QUESTION, an ANSWER, and "
                        "CONTEXT, rate from 1 to 5 how well the answer is supported by "
                        "the context and answers the question. Reply with ONLY the number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}",
                },
            ],
        )
        text = resp.get("message", {}).get("content", "")
        match = re.search(r"[1-5]", text)
        return (int(match.group()) / 5.0) if match else 0.0
    except Exception as exc:  # pragma: no cover - depends on external server
        logger.warning("LLM judge unavailable: %s", exc)
        return 0.0


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def evaluate_sample(
    sample: EvalSample,
    source: str | None = None,
    use_llm_judge: bool = False,
    settings: Settings | None = None,
) -> SampleResult:
    """Run the pipeline on one sample and compute its metrics."""
    stream, retrieval = pipeline_answer(sample.question, source=source, settings=settings)
    answer_text = "".join(stream)

    ranked_sources = [c.metadata.get("source", "unknown") for c in retrieval.chunks]
    retrieved_unique = list(dict.fromkeys(ranked_sources))  # keep order, dedupe
    context = retrieval.context

    metrics: dict[str, float] = {
        "faithfulness": round(faithfulness(answer_text, context), 4),
    }
    if sample.ground_truth:
        metrics["answer_f1"] = round(token_f1(answer_text, sample.ground_truth), 4)
    if sample.expected_keywords:
        metrics["keyword_recall"] = round(keyword_recall(answer_text, sample.expected_keywords), 4)
    if sample.expected_sources:
        metrics["retrieval_hit"] = retrieval_hit(sample.expected_sources, retrieved_unique)
        metrics["retrieval_recall"] = round(
            retrieval_recall(sample.expected_sources, retrieved_unique), 4
        )
        metrics["mrr"] = round(reciprocal_rank(sample.expected_sources, ranked_sources), 4)
    if use_llm_judge:
        metrics["llm_judge"] = round(llm_judge(sample.question, answer_text, context, settings), 4)

    return SampleResult(
        question=sample.question,
        answer=answer_text,
        retrieved_sources=retrieved_unique,
        metrics=metrics,
    )


def _aggregate(results: list[SampleResult]) -> dict[str, float]:
    """Mean of each metric across samples that reported it."""
    sums: Counter = Counter()
    counts: Counter = Counter()
    for r in results:
        for k, v in r.metrics.items():
            sums[k] += v
            counts[k] += 1
    return {k: round(sums[k] / counts[k], 4) for k in sums if counts[k]}


def evaluate_dataset(
    samples: list[EvalSample],
    source: str | None = None,
    use_llm_judge: bool = False,
    settings: Settings | None = None,
) -> EvalReport:
    """Evaluate every sample and return an aggregated report."""
    settings = settings or get_settings()
    results = [
        evaluate_sample(s, source=source, use_llm_judge=use_llm_judge, settings=settings)
        for s in samples
    ]
    return EvalReport(
        results=results,
        aggregate=_aggregate(results),
        config={
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "top_k_rerank": settings.top_k_rerank,
            "n_results": settings.n_results,
            "source_scope": source or "all",
            "llm_judge": use_llm_judge,
        },
    )
