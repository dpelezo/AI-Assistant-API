# app/api/endpoints/chat.py
import os
import asyncio
import datetime
import logging
from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.api.dependencies import get_active_threads, get_workflow_factory
from app.services.memory import get_checkpointer
from app.services.ai import get_system_message
from app.core.langsmith_setup import trace_decorator, create_run_and_child, end_run

# Configure logging
logger = logging.getLogger(__name__)

# Get dependencies
checkpointer = get_checkpointer()
active_threads = get_active_threads()
create_workflow = get_workflow_factory()

# Add a separate thread storage for message history
thread_messages = {}

# Models
class Query(BaseModel):
    content: str = Field(..., description="Query content for the AI assistant")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ConversationResponse(BaseModel):
    thread_id: str
    messages: List[Message]

router = APIRouter()

@router.post("/query", response_model=ConversationResponse)
async def process_query(query: Query, background_tasks: BackgroundTasks):
    """Process a query and return the response."""
    # Initialize or retrieve thread
    thread_id = query.thread_id or str(uuid4())
    logger.info(f"Processing query for thread: {thread_id}")
    
    # Create LangSmith run for the entire query process
    ls_client, ls_run = create_run_and_child(
        name="Chat Query Processing",
        run_type="chain",
        inputs={"query": query.content, "thread_id": thread_id}
    )
    
    # Create thread-specific workflow if needed
    if thread_id not in active_threads:
        active_threads[thread_id] = create_workflow()
        # Initialize thread messages with empty list if new
        thread_messages[thread_id] = []
    
    # Get the workflow
    workflow_app = active_threads[thread_id]
    
    # Add query to thread
    human_message = HumanMessage(content=query.content)
    
    # Store message in our thread_messages dictionary
    if thread_id not in thread_messages:
        thread_messages[thread_id] = []
    
    # Add the human message to our message store
    thread_messages[thread_id].append({
        "type": "human",
        "content": query.content,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # Process in background
    def process_in_background():
        try:
            # Get current messages for this thread
            messages = thread_messages[thread_id]
            
            # Create a child run for the workflow execution
            ls_child_client, ls_child_run = None, None
            if ls_client and ls_run:
                ls_child_client, ls_child_run = create_run_and_child(
                    name="Chat Workflow Execution",
                    run_type="chain",
                    inputs={"message_count": len(messages), "last_message": query.content}
                )
            
            # Initialize state with system message and all messages
            system_message = get_system_message()
            initial_messages = [SystemMessage(content=system_message.content)]
            
            # Add all previous messages to the initial state
            for msg in messages:
                if msg["type"] == "human":
                    initial_messages.append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    # Create AI message with proper tool_calls if present
                    if "tool_calls" in msg and msg["tool_calls"]:
                        initial_messages.append(AIMessage(
                            content=msg["content"],
                            tool_calls=msg["tool_calls"]
                        ))
                    else:
                        initial_messages.append(AIMessage(content=msg["content"]))
            
            # Make sure latest human message is there
            if not any(m.content == query.content for m in initial_messages if isinstance(m, HumanMessage)):
                initial_messages.append(HumanMessage(content=query.content))
            
            initial_state = {"messages": initial_messages}
            logger.info(f"Prepared initial state with {len(initial_messages)} messages")
            
            # Run the workflow synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                logger.info("Running workflow in background")
                # Generate a config for the thread
                thread_config = {"configurable": {"thread_id": thread_id}}
                
                # Enable LangSmith tracing for this specific invocation
                if "LANGSMITH_TRACING" not in os.environ:
                    os.environ["LANGSMITH_TRACING"] = "true"
                
                # Invoke the workflow
                final_state = loop.run_until_complete(workflow_app.ainvoke(
                    initial_state,
                    config=thread_config
                ))
                
                logger.info("Workflow completed, extracting responses")
                
                # Extract AI message and any tool calls
                ai_response = None
                tool_calls_found = []
                
                # Extract the AI response from the final state
                if "messages" in final_state:
                    # Inspect each message for debugging
                    for i, msg in enumerate(final_state["messages"]):
                        if isinstance(msg, AIMessage):
                            tool_calls_info = ""
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                tool_calls_info = f" with {len(msg.tool_calls)} tool calls"
                                tool_calls_found.extend(msg.tool_calls)
                            logger.info(f"Message {i}: AI message{tool_calls_info}")
                        elif isinstance(msg, HumanMessage):
                            logger.info(f"Message {i}: Human message")
                        else:
                            logger.info(f"Message {i}: {type(msg)} message")
                    
                    # Find the last AI message
                    for msg in reversed(final_state["messages"]):
                        if isinstance(msg, AIMessage):
                            # Debug log the message structure
                            logger.info(f"Found AI message: {msg}")
                            if hasattr(msg, "__dict__"):
                                logger.info(f"AI message attributes: {msg.__dict__}")
                            
                            # Record the response
                            ai_response = msg
                            
                            # Add this message to our thread store
                            ai_message_data = {
                                "type": "ai",
                                "content": msg.content,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            
                            # Check for additional_kwargs that might contain tool_calls
                            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                                logger.info(f"Found additional_kwargs: {msg.additional_kwargs}")
                                if "tool_calls" in msg.additional_kwargs:
                                    ai_message_data["tool_calls"] = msg.additional_kwargs["tool_calls"]
                                    logger.info(f"Found tool calls in additional_kwargs: {msg.additional_kwargs['tool_calls']}")
                                    tool_calls_found.extend(msg.additional_kwargs["tool_calls"])
                            
                            # Check and add tool calls if present
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                logger.info(f"Found tool calls in AI response: {msg.tool_calls}")
                                ai_message_data["tool_calls"] = msg.tool_calls
                                if not tool_calls_found:  # Avoid duplicates
                                    tool_calls_found.extend(msg.tool_calls)
                            
                            # Store in our thread_messages
                            thread_messages[thread_id].append(ai_message_data)
                            logger.info(f"Added AI response to thread {thread_id}" + 
                                      (f" with {len(msg.tool_calls) if hasattr(msg, 'tool_calls') and msg.tool_calls else 0} tool calls"))
                            break
                else:
                    logger.warning("No messages found in final state")
                
                # Record workflow results in LangSmith
                if ls_child_client and ls_child_run:
                    end_run(ls_child_client, ls_child_run, {
                        "ai_response": ai_response.content if ai_response else None,
                        "tool_calls_count": len(tool_calls_found),
                        "tool_calls": tool_calls_found
                    })
                
                # Try to save the thread state to checkpointer
                try:
                    # Save current thread_messages to checkpointer
                    checkpointer.put(
                        thread_config,
                        {"messages": thread_messages[thread_id]},
                        metadata={"thread_id": thread_id, "type": "chat_thread"},
                        new_versions=[{
                            "version": "1.0",
                            "timestamp": datetime.datetime.now().isoformat()
                        }]
                    )
                    logger.info(f"Saved thread {thread_id} to checkpointer")
                except Exception as e:
                    logger.error(f"Error saving thread to checkpointer: {str(e)}")
                    
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Create an error message to add to the thread
            error_message = {
                "type": "ai",
                "content": f"Error processing your request: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            }
            thread_messages[thread_id].append(error_message)
            logger.info(f"Added error message to thread {thread_id}")
            
            # Record error in LangSmith
            if ls_child_client and ls_child_run:
                end_run(ls_child_client, ls_child_run, error=str(e))
        
        # Complete the parent run in LangSmith
        if ls_client and ls_run:
            end_run(ls_client, ls_run, {
                "thread_id": thread_id,
                "success": True
            })
    
    background_tasks.add_task(process_in_background)
    
    # Return immediately with thread_id and the human message
    return ConversationResponse(
        thread_id=thread_id,
        messages=[Message(role="human", content=query.content)]
    )

@router.get("/thread/{thread_id}", response_model=ConversationResponse)
async def get_thread(thread_id: str):
    """Get a thread by ID."""
    print(f"Getting thread: {thread_id}")
    
    # First check our thread_messages dictionary
    if thread_id in thread_messages and thread_messages[thread_id]:
        print(f"Found thread {thread_id} in memory with {len(thread_messages[thread_id])} messages")
        messages = []
        
        # Convert our stored messages to API format
        for msg in thread_messages[thread_id]:
            if msg["type"] == "human":
                messages.append(Message(role="human", content=msg["content"]))
            elif msg["type"] == "ai":
                # Handle tool_calls - explicitly check and log
                tool_calls = None
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls = msg["tool_calls"]
                    print(f"Found tool calls in message: {tool_calls}")
                
                messages.append(Message(
                    role="ai", 
                    content=msg["content"],
                    tool_calls=tool_calls
                ))
        
        print(f"Returning {len(messages)} messages, last message tool_calls: {messages[-1].tool_calls if messages and messages[-1].role == 'ai' else None}")
        return ConversationResponse(
            thread_id=thread_id,
            messages=messages
        )
    
    # If not in memory, try to get from checkpointer
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    try:
        thread_data = checkpointer.get(thread_config)
        print(f"Retrieved thread from checkpointer: {thread_data is not None}")
        
        if thread_data and "messages" in thread_data:
            messages = []
            
            # Convert stored messages to API format
            for msg in thread_data["messages"]:
                if msg["type"] == "human":
                    messages.append(Message(role="human", content=msg["content"]))
                elif msg["type"] == "ai":
                    tool_calls = msg.get("tool_calls")
                    messages.append(Message(
                        role="ai", 
                        content=msg["content"],
                        tool_calls=tool_calls
                    ))
            
            # Also update our in-memory storage
            thread_messages[thread_id] = thread_data["messages"]
            
            return ConversationResponse(
                thread_id=thread_id,
                messages=messages
            )
    except Exception as e:
        print(f"Error retrieving thread from checkpointer: {str(e)}")
    
    # Thread not found
    print(f"Thread not found: {thread_id}")
    raise HTTPException(status_code=404, detail="Thread not found")

@router.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread by ID."""
    print(f"Deleting thread: {thread_id}")
    
    # Remove from active threads
    if thread_id in active_threads:
        del active_threads[thread_id]
    
    # Remove from our thread_messages store
    if thread_id in thread_messages:
        del thread_messages[thread_id]
        print(f"Removed thread {thread_id} from memory")
    
    # Try to remove from checkpointer
    thread_config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpointer.delete(thread_config)
        print(f"Successfully deleted thread {thread_id} from checkpointer")
    except Exception as e:
        print(f"Error deleting thread from checkpointer: {str(e)}")
    
    return {"status": "success", "message": "Thread deleted"}

@router.get("/debug/threads")
async def debug_list_threads():
    """Debug endpoint to list all active threads."""
    return {
        "active_threads": list(active_threads.keys()),
        "thread_messages": {
            thread_id: {
                "message_count": len(messages),
                "last_update": messages[-1]["timestamp"] if messages else None,
                "preview": [
                    {
                        "type": msg["type"],
                        "content_preview": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                        "has_tool_calls": "tool_calls" in msg and bool(msg["tool_calls"])
                    }
                    for msg in messages[-2:] if messages  # Show last 2 messages
                ]
            }
            for thread_id, messages in thread_messages.items()
        }
    }