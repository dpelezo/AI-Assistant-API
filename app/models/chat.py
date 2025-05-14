from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Query(BaseModel):
    """Model for user queries."""
    content: str = Field(..., description="Query content for the AI assistant")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")

class Message(BaseModel):
    """Model for chat messages."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ConversationResponse(BaseModel):
    """Model for conversation responses."""
    thread_id: str
    messages: List[Message]