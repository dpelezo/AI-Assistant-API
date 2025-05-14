# app/tests/test_api.py
"""Test the API endpoints."""
import pytest
from unittest.mock import patch, MagicMock
import uuid

from app.api.endpoints.chat import router as chat_router


class TestChatEndpoints:
    """Test class for chat endpoints."""
    
    @pytest.mark.asyncio
    @patch("app.api.dependencies.get_memory_service")
    @patch("app.api.dependencies.get_workflow_manager")
    async def test_process_query_new_thread(
        self, 
        mock_get_workflow_manager,
        mock_get_memory_service,
        test_client
    ):
        """Test processing a query with a new thread."""
        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.get_thread_state.return_value = None
        mock_get_memory_service.return_value = mock_memory
        
        mock_workflow = MagicMock()
        mock_workflow_app = MagicMock()
        mock_workflow.create_workflow.return_value = mock_workflow_app
        mock_get_workflow_manager.return_value = mock_workflow
        
        # Test data
        test_query = {
            "content": "Test query",
            "thread_id": None
        }
        
        # Make request
        response = test_client.post("/api/v1/chat/query", json=test_query)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert data["messages"][0]["content"] == "Test query"
        assert mock_memory.get_thread_state.called
        assert mock_workflow.create_workflow.called
    
    @pytest.mark.asyncio
    @patch("app.api.dependencies.get_memory_service")
    async def test_get_thread(
        self,
        mock_get_memory_service,
        test_client
    ):
        """Test getting a thread."""
        # Setup mocks
        mock_memory = MagicMock()
        
        # Create mock messages without using __class__.__name__
        human_message = MagicMock(spec="HumanMessage")
        human_message.content = "Hello"
        # Set type information another way
        type(human_message).__name__ = "HumanMessage"
        
        ai_message = MagicMock(spec="AIMessage")
        ai_message.content = "Hi there"
        ai_message.tool_calls = None
        # Set type information another way
        type(ai_message).__name__ = "AIMessage"
        
        mock_thread_state = {
            "messages": [human_message, ai_message]
        }
        
        mock_memory.get_thread_state.return_value = mock_thread_state
        mock_get_memory_service.return_value = mock_memory
        
        # Generate a thread ID
        thread_id = str(uuid4())
        
        # Make request
        response = test_client.get(f"/api/v1/chat/thread/{thread_id}")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == thread_id
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "human"
        assert data["messages"][1]["role"] == "ai"
    
    @pytest.mark.asyncio
    @patch("app.api.dependencies.get_memory_service")
    async def test_delete_thread(
        self,
        mock_get_memory_service,
        test_client
    ):
        """Test deleting a thread."""
        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.delete_thread.return_value = True
        mock_get_memory_service.return_value = mock_memory
        
        # Generate a thread ID
        thread_id = str(uuid4())
        
        # Make request
        response = test_client.delete(f"/api/v1/chat/thread/{thread_id}")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert mock_memory.delete_thread.called_with(thread_id)