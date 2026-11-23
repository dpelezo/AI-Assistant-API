# app/models/state.py
from typing import Any, TypedDict


class MessageDict(TypedDict):
    """TypedDict for message dictionaries."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None


class ThreadState(TypedDict):
    """TypedDict for thread state."""
    messages: list[Any]  # Can be AIMessage or HumanMessage


class MemoryItem(TypedDict):
    """TypedDict for memory items."""
    thread_id: str
    state: ThreadState