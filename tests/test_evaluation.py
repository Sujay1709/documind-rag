from documind import evaluation as ev
from documind.evaluation import EvalSample
from documind.pipeline import RetrievalResult
from documind.reranker import RankedChunk


# --- pure metric functions -------------------------------------------------- #
def test_token_f1_identical_and_disjoint():
    assert ev.token_f1("paris is the capital", "paris is the capital") == 1.0
    assert ev.token_f1("paris", "tokyo") == 0.0
    assert ev.token_f1("", "anything") == 0.0


def test_token_f1_partial():
    score = ev.token_f1("the capital is paris", "paris")
    assert 0.0 < score < 1.0


def test_keyword_recall():
    assert ev.keyword_recall("Paris is lovely in 2014", ["Paris", "2014"]) == 1.0
    assert ev.keyword_recall("Paris only", ["Paris", "2014"]) == 0.5
    assert ev.keyword_recall("text", []) == 0.0


def test_faithfulness():
    # Every content word of the answer appears in the context -> fully grounded.
    assert ev.faithfulness("Paris capital", "Paris is the capital of France") == 1.0
    # Hallucinated content word absent from context lowers the score.
    assert ev.faithfulness("Berlin capital", "Paris is the capital") == 0.5
    assert ev.faithfulness("", "anything") == 0.0


def test_retrieval_metrics():
    assert ev.retrieval_hit(["a"], ["b", "a"]) == 1.0
    assert ev.retrieval_hit(["a"], ["b", "c"]) == 0.0
    assert ev.retrieval_recall(["a", "b"], ["a", "x"]) == 0.5
    assert ev.reciprocal_rank(["b"], ["a", "b", "c"]) == 0.5
    assert ev.reciprocal_rank(["z"], ["a", "b"]) == 0.0


# --- runner (pipeline mocked, no Ollama) ------------------------------------ #
def test_evaluate_dataset_with_mocked_pipeline(monkeypatch):
    chunks = [
        RankedChunk(text="Paris is the capital of France.",
                    metadata={"source": "geography_pdf", "page": 0}, score=9.0),
        RankedChunk(text="France is in Europe.",
                    metadata={"source": "geography_pdf", "page": 1}, score=4.0),
    ]

    def fake_answer(question, history=None, source=None, settings=None):
        retrieval = RetrievalResult(
            context="\n\n".join(c.text for c in chunks), chunks=chunks
        )
        return iter(["Paris ", "is ", "the ", "capital."]), retrieval

    monkeypatch.setattr(ev, "pipeline_answer", fake_answer)

    samples = [
        EvalSample(
            question="What is the capital of France?",
            ground_truth="Paris is the capital of France.",
            expected_sources=["geography_pdf"],
            expected_keywords=["Paris"],
        )
    ]
    report = ev.evaluate_dataset(samples)

    assert len(report.results) == 1
    m = report.results[0].metrics
    assert m["retrieval_hit"] == 1.0
    assert m["mrr"] == 1.0
    assert m["keyword_recall"] == 1.0
    assert 0.0 < m["answer_f1"] <= 1.0
    assert m["faithfulness"] == 1.0
    # aggregate present and well-formed
    assert "answer_f1" in report.aggregate
    assert report.config["source_scope"] == "all"


def test_report_markdown_renders(monkeypatch):
    def fake_answer(question, history=None, source=None, settings=None):
        return iter(["ok"]), RetrievalResult(context="ctx ok", chunks=[])

    monkeypatch.setattr(ev, "pipeline_answer", fake_answer)
    report = ev.evaluate_dataset([EvalSample(question="q?")])
    md = report.to_markdown()
    assert "RAG evaluation" in md and "Aggregate metrics" in md
