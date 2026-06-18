from documind import vectorstore
from documind.config import Settings


def test_persist_path_is_absolute_and_created(tmp_path):
    rel = tmp_path / "nested" / ".." / "chroma"  # deliberately non-normalised
    s = Settings(persist_dir=rel)
    resolved = vectorstore._persist_path(s)
    assert resolved.is_absolute()
    assert resolved.exists()
    # normalised: the ".." is collapsed away
    assert ".." not in resolved.parts
