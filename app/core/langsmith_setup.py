# app/core/langsmith_setup.py
import logging
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger(__name__)

def setup_langsmith(project_name: str | None = None):
    """
    Set up LangSmith tracing for the application.
    
    Args:
        project_name: Optional project name to use for LangSmith traces.
                     If not provided, will use the LANGSMITH_PROJECT from settings.
    """
    # Check if LangSmith API key is set
    if not settings.LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_API_KEY not set. LangSmith tracing will not be enabled.")
        return False
    
    # Set default project name if not provided
    project = project_name or settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_PROJECT"] = project
    
    # Enable tracing
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_TRACING"] = str(settings.LANGSMITH_TRACING).lower()
    
    # Configure LangSmith to trace in background (better for production)
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = str(settings.LANGSMITH_CALLBACKS_BACKGROUND).lower()
    
    # Log configuration
    logger.info(f"LangSmith tracing enabled for project: {project}")
    return True

def wrap_functions_with_langsmith():
    """
    Wrap OpenAI client and other functions with LangSmith tracing.
    This should be called after setup_langsmith().
    
    Returns:
        bool: True if wrapping was successful, False otherwise
    """
    try:
        # Import LangSmith wrappers
        import openai
        from langsmith.wrappers import wrap_openai
        
        # Check if OpenAI client is being used
        if hasattr(openai, "Client"):
            # Monkey patch OpenAI to use wrapped client
            original_client = openai.Client
            openai.Client = lambda *args, **kwargs: wrap_openai(original_client(*args, **kwargs))
            logger.info("OpenAI client wrapped with LangSmith tracing")
        else:
            logger.warning("OpenAI Client not found, skipping wrapper")
            
        # Additional wrappers could be added here
        
        return True
    except ImportError as e:
        logger.warning(f"Failed to import LangSmith wrappers: {e}")
        logger.warning("Install with: pip install langsmith")
        return False
    except Exception:
        logger.exception("Error wrapping functions with LangSmith")
        return False

def create_run_and_child(
    name: str, 
    run_type: str, 
    inputs: dict[str, Any], 
    project_name: str | None = None
):
    """
    Create a custom run in LangSmith with proper tracing.
    
    Args:
        name: Name of the run
        run_type: Type of the run (e.g., "chain", "llm", "tool")
        inputs: Inputs to the run
        project_name: Optional project name to use for this run
    
    Returns:
        A tuple of (ls_client, run) that can be used for further tracing,
        or (None, None) when LangSmith is not configured.
    """
    if not settings.is_langsmith_enabled:
        return None, None

    try:
        import langsmith as ls

        # Use specified project or fall back to environment variable
        project = project_name or os.environ.get("LANGSMITH_PROJECT", "default")

        client = ls.Client()

        # create_run returns None, so we choose the run ID ourselves
        run_id = uuid4()
        client.create_run(
            name=name,
            inputs=inputs,
            run_type=run_type,
            project_name=project,
            id=run_id,
            start_time=datetime.now(UTC),
        )
        run = SimpleNamespace(id=run_id)

        logger.info(f"Created LangSmith run: {run.id} ({name})")
        return client, run
    except ImportError:
        logger.warning("LangSmith not installed. Install with: pip install langsmith")
        return None, None
    except Exception:
        logger.exception("Error creating LangSmith run")
        return None, None

def end_run(client, run, outputs=None, error=None):
    """
    End a LangSmith run with outputs or error.
    
    Args:
        client: LangSmith client
        run: Run object returned by create_run_and_child
        outputs: Optional outputs to add to the run
        error: Optional error to add to the run
    """
    if not client or not run:
        return
    
    try:
        if error:
            client.update_run(
                run.id,
                error=str(error),
                end_time=datetime.now(UTC),
            )
            logger.info(f"Updated LangSmith run {run.id} with error")
        else:
            client.update_run(
                run.id,
                outputs=outputs or {},
                end_time=datetime.now(UTC),
            )
            logger.info(f"Updated LangSmith run {run.id} with outputs")
    except Exception:
        logger.exception("Error ending LangSmith run")

def trace_decorator(run_type=None, name=None, project_name=None):
    """
    Decorator to trace a function with LangSmith.
    
    Args:
        run_type: Type of the run (default: "chain")
        name: Name of the run (default: function name)
        project_name: Project name (default: from environment)
        
    Example:
        @trace_decorator(run_type="tool", name="Search Tool")
        def search(query):
            # implementation
            return results
    """
    try:
        from langsmith import traceable
        return traceable(
            run_type=run_type or "chain",
            name=name,
            project_name=project_name
        )
    except ImportError:
        # If langsmith not installed, return a no-op decorator
        def no_op_decorator(func):
            return func
        return no_op_decorator