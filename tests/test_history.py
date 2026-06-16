from documind import history
from documind.config import Settings
from documind.reranker import RankedChunk


def _settings(tmp_path):
    return Settings(history_file=tmp_path / "history.json", max_history=3)


def test_add_and_load_roundtrip(tmp_path):
    s = _settings(tmp_path)
    chunks = [
        RankedChunk(
            text="Paris is the capital.",
            metadata={"source": "geo_pdf", "page": 0},
            score=7.5,
        )
    ]
    history.add("What is the capital of France?", "Paris.", chunks=chunks, settings=s)

    entries = history.load(s)
    assert len(entries) == 1
    e = entries[0]
    assert e.question.startswith("What is the capital")
    assert e.answer == "Paris."
    assert e.sources[0]["source"] == "geo_pdf"
    assert e.documents == ["geo_pdf"]


def test_load_returns_newest_first(tmp_path):
    s = _settings(tmp_path)
    history.add("q1", "a1", settings=s)
    history.add("q2", "a2", settings=s)
    history.add("q3", "a3", settings=s)
    entries = history.load(s)
    assert [e.question for e in entries] == ["q3", "q2", "q1"]


def test_max_history_is_enforced(tmp_path):
    s = _settings(tmp_path)  # max_history=3
    for i in range(5):
        history.add(f"q{i}", f"a{i}", settings=s)
    entries = history.load(s)
    assert len(entries) == 3
    assert [e.question for e in entries] == ["q4", "q3", "q2"]


def test_clear(tmp_path):
    s = _settings(tmp_path)
    history.add("q", "a", settings=s)
    history.clear(s)
    assert history.load(s) == []


def test_load_missing_file_is_empty(tmp_path):
    s = _settings(tmp_path)
    assert history.load(s) == []
