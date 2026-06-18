"""Guardrail tests for document-grounded answering and prompt-injection resistance."""

from documind.config import Settings
from documind.llm import SYSTEM_PROMPT, _build_messages


def test_system_prompt_has_grounding_and_injection_rules():
    p = SYSTEM_PROMPT.lower()
    assert "only" in p  # answer only from context
    assert "untrusted" in p  # treat context as data
    assert "ignore" in p  # refuse "ignore your instructions" style attacks
    assert "reveal" in p  # never reveal the system prompt


def test_build_messages_wraps_context_as_data():
    msgs = _build_messages("SECRET DOC TEXT", "What is in here?", history=None)
    assert msgs[0]["role"] == "system"
    user = msgs[-1]["content"]
    # Context is delimited and explicitly labelled untrusted reference data.
    assert "<<<CONTEXT" in user and "CONTEXT>>>" in user
    assert "untrusted" in user.lower()
    assert "SECRET DOC TEXT" in user
    assert "What is in here?" in user


def test_history_is_preserved_between_system_and_question():
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    msgs = _build_messages("ctx", "q", history=history)
    assert msgs[0]["role"] == "system"
    assert msgs[1:3] == history
    assert msgs[-1]["content"].endswith("Question: q")


def test_default_upload_limit_is_large():
    assert Settings().max_upload_mb >= 200
