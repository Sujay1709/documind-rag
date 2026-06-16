from documind.config import Settings


def test_embeddings_url_built_from_base():
    s = Settings(ollama_base_url="http://localhost:11434/")
    assert s.ollama_embeddings_url == "http://localhost:11434/api/embeddings"


def test_env_override(monkeypatch):
    monkeypatch.setenv("DOCUMIND_CHUNK_SIZE", "512")
    monkeypatch.setenv("DOCUMIND_CHAT_MODEL", "llama3.1:8b")
    s = Settings()
    assert s.chunk_size == 512
    assert s.chat_model == "llama3.1:8b"
