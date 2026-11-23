"""Shared test fixtures.

Unit tests must never call a live model or a tracing backend. The autouse
fixture below makes any accidental model call fail loudly instead.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch):
    """Fail the test if code tries to build a real model; skip LangSmith runs."""

    def _blocked(*args, **kwargs):
        raise AssertionError("Unit tests must not call a live model")

    monkeypatch.setattr("app.core.workflow.get_model", _blocked)
    monkeypatch.setattr(
        "app.api.endpoints.chat.create_run_and_child", lambda **kwargs: (None, None)
    )


@pytest.fixture
def client():
    return TestClient(app)
