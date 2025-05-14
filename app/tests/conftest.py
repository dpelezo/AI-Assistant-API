# app/tests/conftest.py
"""Test fixtures and configuration."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.services.memory import MemoryService
from app.services.ai import AIService
from app.core.workflow import WorkflowManager


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing."""
    memory_service = MagicMock(spec=MemoryService)
    memory_service.get_thread_state.return_value = None
    return memory_service


@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing."""
    ai_service = MagicMock(spec=AIService)
    ai_service.generate_response.return_value = MagicMock(
        content="This is a test response."
    )
    return ai_service


@pytest.fixture
def mock_workflow_manager():
    """Mock workflow manager for testing."""
    workflow_manager = MagicMock(spec=WorkflowManager)
    workflow_manager.create_workflow.return_value = MagicMock()
    return workflow_manager