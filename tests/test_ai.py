"""The model request must not include sampling parameters by default."""

from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.services import ai


def request_payload(monkeypatch, **overrides):
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test")
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)
    model = ai.get_model().bound  # get_model returns the model with tools bound
    return model._get_request_payload([HumanMessage(content="hi")])


def test_no_temperature_sent_by_default(monkeypatch):
    payload = request_payload(monkeypatch)
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload
    assert payload["model"] == settings.LLM_MODEL


def test_temperature_sent_only_when_configured(monkeypatch):
    payload = request_payload(monkeypatch, LLM_TEMPERATURE=1.0)
    assert payload["temperature"] == 1.0


def test_max_tokens_leaves_room_for_thinking(monkeypatch):
    assert request_payload(monkeypatch)["max_tokens"] == settings.LLM_MAX_TOKENS == 16000


def test_message_text_handles_strings_and_content_blocks():
    blocks = [
        {"type": "thinking", "thinking": "hidden reasoning", "signature": "sig"},
        {"type": "text", "text": "First part."},
        {"type": "tool_use", "id": "t1", "name": "retrieve_web_content", "input": {}},
        {"type": "text", "text": "Second part."},
    ]
    assert ai.message_text("plain") == "plain"
    assert ai.message_text(blocks) == "First part.\n\nSecond part."
    assert ai.message_text(None) == ""


def test_hit_length_limit_reads_stop_reason():
    from langchain_core.messages import AIMessage

    assert ai.hit_length_limit(AIMessage(content="x", response_metadata={"stop_reason": "max_tokens"}))
    assert not ai.hit_length_limit(AIMessage(content="x", response_metadata={"stop_reason": "end_turn"}))
    assert not ai.hit_length_limit(AIMessage(content="x"))
