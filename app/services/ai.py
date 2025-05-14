# app/services/ai.py
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from app.services.web_search import retrieve_web_content
from app.core.config import settings
from langsmith import traceable


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
        api_key=settings.ANTHROPIC_API_KEY,
        # Remove the system_prompt from here
    ).bind_tools(
        [retrieve_web_content],
        tool_choice="auto"
    )

def get_system_message():
    """Get a standard system message to encourage tool use."""
    return SystemMessage(content=(
        "You are a helpful assistant with web search capabilities. "
        "When asked about current events, news, or facts, ALWAYS use your retrieve_web_content tool "
        "to provide up-to-date information. Do not rely on your training data for current information."
    ))