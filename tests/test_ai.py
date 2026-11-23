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
