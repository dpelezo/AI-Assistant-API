# app/models/state.py
from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.messages import AIMessage, HumanMessage


class MessageDict(TypedDict):
    """TypedDict for message dictionaries."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]]


class ThreadState(TypedDict):
    """TypedDict for thread state."""
    messages: List[Any]  # Can be AIMessage or HumanMessage


class MemoryItem(TypedDict):
    """TypedDict for memory items."""
    thread_id: str
    state: ThreadState