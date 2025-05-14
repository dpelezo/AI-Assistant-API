# app/services/memory.py
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, Optional

# Create a single MemorySaver instance
checkpointer = MemorySaver()

def get_checkpointer():
    """Get the MemorySaver instance."""
    return checkpointer