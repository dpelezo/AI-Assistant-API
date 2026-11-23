# app/services/ai.py
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langsmith import traceable

from app.core.config import settings
from app.services.web_search import retrieve_web_content


@traceable # Auto-trace this function
def get_model():
    """
    Get a configured ChatAnthropic model with tools bound.
    
    The proper way to set a system message is to include it in messages
    when calling the model, not as a parameter during initialization.
    """
    return ChatAnthropic(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        api_key=settings.ANTHROPIC_API_KEY,
        # Remove the system_prompt from here
    ).bind_tools(
        [retrieve_web_content],
        tool_choice="auto"
    )

def message_text(content: Any) -> str:
    """Return the plain text of a model reply.

    Current Claude models think adaptively, so a reply can be a list of
    content blocks (thinking, text, tool_use) instead of a string. Only the
    text blocks are shown to users.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n\n".join(p for p in parts if p.strip())
    return "" if content is None else str(content)


TRUNCATION_NOTE = (
    "\n\n*This answer reached the length limit and was cut off. "
    "Ask a narrower question for a complete answer.*"
)


def hit_length_limit(message: Any) -> bool:
    """True when the model stopped because it reached max_tokens."""
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("stop_reason") == "max_tokens"


def get_system_message():
    """Get a standard system message to encourage tool use."""
    return SystemMessage(content=(
        "You are a helpful assistant with web search capabilities. "
        "When asked about current events, news, or facts, ALWAYS use your retrieve_web_content tool "
        "to provide up-to-date information. Do not rely on your training data for current information."
    ))