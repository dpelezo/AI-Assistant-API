# app/core/workflow.py
import logging
from typing import Literal

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.services.ai import get_model, hit_length_limit, message_text
from app.services.web_search import retrieve_web_content

logger = logging.getLogger(__name__)

# Determine whether to continue or end
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # Check for tool calls
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    logger.debug(f"Checking tool calls: {has_tool_calls}")
    if has_tool_calls:
        logger.debug(f"Tool calls found: {last_message.tool_calls}")
        return "tools"
    return END

# Function to generate model responses
def call_model(state: MessagesState):
    messages = state["messages"]
    logger.debug(f"Calling model with {len(messages)} messages")
    model = get_model()
    response = model.invoke(messages)
    logger.debug(f"Model response: {message_text(response.content)[:100]}...")
    stop_reason = (response.response_metadata or {}).get("stop_reason")
    logger.info(f"Model call finished: stop_reason={stop_reason} usage={response.usage_metadata}")
    if hit_length_limit(response):
        logger.warning("Model reply hit max_tokens and was cut off; consider raising LLM_MAX_TOKENS")
    if hasattr(response, "tool_calls"):
        logger.debug(f"Tool calls in response: {response.tool_calls}")
    return {"messages": [response]}

# Create the workflow
def create_workflow():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([retrieve_web_content]))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()
