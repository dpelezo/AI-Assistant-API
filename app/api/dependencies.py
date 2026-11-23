# app/api/dependencies.py
from fastapi import HTTPException

from app.core.config import settings
from app.core.workflow import create_workflow

# Store active threads
active_threads = {}

def get_active_threads():
    """Get the active threads dictionary."""
    return active_threads

def get_workflow_factory():
    """Get a factory function to create workflows."""
    return create_workflow

def require_debug() -> None:
    if not settings.DEBUG:
        raise HTTPException(status_code=404)
