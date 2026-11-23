"""Tests for the manual LangSmith run helpers. No test calls LangSmith."""

from datetime import datetime

import langsmith
import pytest

from app.core import langsmith_setup
from app.core.config import settings


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.created: list[dict] = []
        self.updated: list[tuple] = []

    def create_run(self, **kwargs):
        self.created.append(kwargs)

    def update_run(self, run_id, **kwargs):
        self.updated.append((run_id, kwargs))


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(langsmith, "Client", FakeClient)


def test_no_run_when_langsmith_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", None)
    assert langsmith_setup.create_run_and_child(
        name="x", run_type="chain", inputs={}
    ) == (None, None)


def test_create_and_end_run_use_the_real_client_methods(tracing_on):
    client, run = langsmith_setup.create_run_and_child(
        name="Chat Query Processing", run_type="chain", inputs={"query": "hi"}
    )

    assert client.created[0]["id"] == run.id
    assert client.created[0]["name"] == "Chat Query Processing"

    langsmith_setup.end_run(client, run, {"ok": True})
    run_id, kwargs = client.updated[0]
    assert run_id == run.id
    assert kwargs["outputs"] == {"ok": True}
    assert isinstance(kwargs["end_time"], datetime)


def test_end_run_records_errors(tracing_on):
    client, run = langsmith_setup.create_run_and_child(
        name="x", run_type="chain", inputs={}
    )
    langsmith_setup.end_run(client, run, error="boom")
    assert client.updated[0][1]["error"] == "boom"
