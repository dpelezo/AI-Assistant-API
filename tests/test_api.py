"""API tests. No test here calls a live model."""

import anthropic
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import settings

DEBUG_ROUTES = [
    "/api/debug/threads",
    "/api/research/debug/latest",
    "/api/research/debug/all",
    "/api/research/debug/report/some-id",
]


class FakeGraph:
    """Stands in for the compiled LangGraph workflow.

    Records the state it receives and replies with a fixed AI message,
    so the test checks our endpoint logic, not the model.
    """

    def __init__(self, reply: str = "Fake answer"):
        self.reply = reply
        self.calls: list[dict] = []

    async def ainvoke(self, state, config=None):
        self.calls.append({"state": state, "config": config})
        return {"messages": [*state["messages"], AIMessage(content=self.reply)]}


@pytest.fixture
def fake_graph(monkeypatch):
    graph = FakeGraph()
    # chat.py binds create_workflow at import time, so patch it where it is used.
    monkeypatch.setattr("app.api.endpoints.chat.create_workflow", lambda: graph)
    return graph


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_thread_and_stores_ai_reply(client, fake_graph):
    r = client.post("/api/query", json={"content": "What is MCP?"})

    assert r.status_code == 200
    body = r.json()
    thread_id = body["thread_id"]
    assert body["messages"] == [
        {"role": "human", "content": "What is MCP?", "tool_calls": None, "is_error": False}
    ]

    # TestClient runs background tasks before returning, so the graph has run.
    assert len(fake_graph.calls) == 1
    sent = fake_graph.calls[0]["state"]["messages"]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[-1], HumanMessage)
    assert sent[-1].content == "What is MCP?"
    assert fake_graph.calls[0]["config"] == {"configurable": {"thread_id": thread_id}}

    thread = client.get(f"/api/thread/{thread_id}").json()
    assert [m["role"] for m in thread["messages"]] == ["human", "ai"]
    assert thread["messages"][-1]["content"] == "Fake answer"


def test_query_continues_existing_thread(client, fake_graph):
    thread_id = client.post("/api/query", json={"content": "First"}).json()["thread_id"]
    client.post("/api/query", json={"content": "Second", "thread_id": thread_id})

    sent = fake_graph.calls[-1]["state"]["messages"]
    assert [m.content for m in sent[1:]] == ["First", "Fake answer", "Second"]


def test_model_error_shows_friendly_message(client, monkeypatch):
    class FailingGraph:
        async def ainvoke(self, state, config=None):
            response = httpx.Response(
                400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            )
            raise anthropic.BadRequestError(
                "Your credit balance is too low to access the Anthropic API.",
                response=response,
                body=None,
            )

    monkeypatch.setattr("app.api.endpoints.chat.create_workflow", FailingGraph)
    thread_id = client.post("/api/query", json={"content": "Hi"}).json()["thread_id"]

    reply = client.get(f"/api/thread/{thread_id}").json()["messages"][-1]
    assert reply["role"] == "ai"
    assert reply["is_error"] is True
    assert reply["content"] == "The assistant is temporarily unavailable. Please try again later."
    assert "credit" not in reply["content"]


def test_error_notices_are_not_sent_back_to_the_model(client, monkeypatch, fake_graph):
    class FailingGraph:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("boom")

    monkeypatch.setattr("app.api.endpoints.chat.create_workflow", FailingGraph)
    thread_id = client.post("/api/query", json={"content": "First"}).json()["thread_id"]

    # The thread already exists, so the next message reuses its stored graph.
    from app.api.endpoints.chat import active_threads

    active_threads[thread_id] = fake_graph
    client.post("/api/query", json={"content": "Second", "thread_id": thread_id})

    sent = [m.content for m in fake_graph.calls[-1]["state"]["messages"][1:]]
    assert sent == ["First", "Second"]


def test_reply_with_thinking_blocks_returns_only_the_text(client, monkeypatch):
    class ThinkingGraph:
        async def ainvoke(self, state, config=None):
            reply = AIMessage(content=[
                {"type": "thinking", "thinking": "hidden reasoning", "signature": "sig"},
                {"type": "text", "text": "MCP is the **Model Context Protocol**."},
            ])
            return {"messages": [*state["messages"], reply]}

    monkeypatch.setattr("app.api.endpoints.chat.create_workflow", ThinkingGraph)
    thread_id = client.post("/api/query", json={"content": "What is MCP?"}).json()["thread_id"]

    r = client.get(f"/api/thread/{thread_id}")
    assert r.status_code == 200
    reply = r.json()["messages"][-1]
    assert reply["content"] == "MCP is the **Model Context Protocol**."
    assert "hidden reasoning" not in r.text


def test_unknown_thread_returns_404(client):
    assert client.get("/api/thread/does-not-exist").status_code == 404


@pytest.mark.parametrize("path", DEBUG_ROUTES)
def test_debug_routes_hidden_in_production(client, monkeypatch, path):
    monkeypatch.setattr(settings, "DEBUG", False)
    assert client.get(path).status_code == 404


def test_debug_threads_available_in_debug_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "DEBUG", True)
    r = client.get("/api/debug/threads")
    assert r.status_code == 200
    assert "active_threads" in r.json()
