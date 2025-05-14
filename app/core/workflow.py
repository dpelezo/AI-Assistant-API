# app/core/workflow.py
from typing import Literal
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.services.ai import get_model
from app.services.web_search import retrieve_web_content

# Determine whether to continue or end
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # Check for tool calls
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    print(f"Checking tool calls: {has_tool_calls}")
    if has_tool_calls:
        print(f"Tool calls found: {last_message.tool_calls}")
        return "tools"
    return END

# Function to generate model responses
def call_model(state: MessagesState):
    messages = state["messages"]
    print(f"Calling model with {len(messages)} messages")
    model = get_model()
    response = model.invoke(messages)
    print(f"Model response: {response.content[:100]}...")
    if hasattr(response, "tool_calls"):
        print(f"Tool calls in response: {response.tool_calls}")
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