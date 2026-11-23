from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):
    """Model for user queries."""
    content: str = Field(..., description="Query content for the AI assistant")
    thread_id: str | None = Field(None, description="Optional thread ID for conversation continuity")

class Message(BaseModel):
    """Model for chat messages."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None

class ConversationResponse(BaseModel):
    """Model for conversation responses."""
    thread_id: str
    messages: list[Message]