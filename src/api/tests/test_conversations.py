import sys
import os
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, UTC  # Note: Using timezone-aware dates
from unittest.mock import patch, MagicMock
from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.main import app
from api.routes.auth import get_current_user  # Import the actual dependency

client = TestClient(app)

# Mock user for testing
@pytest.fixture
def mock_current_user():
    return {
        "_id": str(ObjectId()),
        "email": "test@example.com",
        "username": "testuser"
    }

# Mock conversation data
@pytest.fixture
def mock_conversation(mock_current_user):
    return {
        "_id": str(ObjectId()),
        "user_id": str(mock_current_user["_id"]),
        "session_id": str(ObjectId()),
        "title": "Test Conversation",
        "created_at": datetime.now(UTC),  # Using timezone-aware datetime
        "updated_at": datetime.now(UTC),  # Using timezone-aware datetime
        "messages": [],
        "is_active": True
    }

# Override the get_current_user dependency for all tests
@pytest.fixture(autouse=True)
def override_dependency(mock_current_user):
    # Create an async function that returns the mock user
    async def mock_get_current_user():
        return mock_current_user

    # Override the dependency
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    # Clean up after tests
    app.dependency_overrides.clear()

class TestConversationsAPI:
    def test_create_conversation(self, mock_current_user):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.insert_one.return_value = None
            
            response = client.post(
                "/conversations/",
                json={
                    "session_id": str(ObjectId()),
                    "title": "New Test Conversation"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "New Test Conversation"
            assert data["user_id"] == str(mock_current_user["_id"])
            assert "id" in data
            assert "created_at" in data
            assert "updated_at" in data
            assert data["message_count"] == 0
            assert data["last_message_preview"] is None

    def test_list_conversations(self, mock_current_user, mock_conversation):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.find.return_value.sort.return_value = [mock_conversation]
            
            response = client.get("/conversations/")
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["id"] == mock_conversation["_id"]
            assert data[0]["title"] == mock_conversation["title"]

    def test_get_conversation_detail(self, mock_current_user, mock_conversation):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.find_one.return_value = mock_conversation
            
            response = client.get(f"/conversations/{mock_conversation['_id']}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == mock_conversation["_id"]
            assert data["title"] == mock_conversation["title"]
            assert "messages" in data

    def test_get_nonexistent_conversation(self, mock_current_user):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.find_one.return_value = None
            
            response = client.get(f"/conversations/{str(ObjectId())}")
            
            assert response.status_code == 404
            assert response.json()["detail"] == "Conversation not found"

    def test_update_conversation(self, mock_current_user, mock_conversation):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            # Create a copy of mock_conversation for the updated version
            updated_conversation = mock_conversation.copy()
            updated_conversation["title"] = "Updated Test Conversation"
            
            # First find_one returns original conversation
            # Second find_one (after update) returns updated conversation
            mock_collection.find_one.side_effect = [mock_conversation, updated_conversation]
            mock_collection.update_one.return_value = MagicMock(modified_count=1)
            
            new_title = "Updated Test Conversation"
            response = client.put(
                f"/conversations/{mock_conversation['_id']}",
                json={"title": new_title}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == new_title

    def test_delete_conversation(self, mock_current_user, mock_conversation):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.update_one.return_value = MagicMock(modified_count=1)
            
            response = client.delete(f"/conversations/{mock_conversation['_id']}")
            
            assert response.status_code == 200
            assert response.json()["message"] == "Conversation deleted successfully"

    def test_delete_nonexistent_conversation(self, mock_current_user):
        with patch("api.routes.conversations.conversations_collection") as mock_collection:
            mock_collection.update_one.return_value = MagicMock(modified_count=0)
            
            response = client.delete(f"/conversations/{str(ObjectId())}")
            
            assert response.status_code == 404
            assert response.json()["detail"] == "Conversation not found or already deleted"

class ConversationMessage(BaseModel):
    """Individual message in a conversation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ConversationResponse(BaseModel):
    """Conversation data returned to client"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None

class MessageResponse(BaseModel):
    """Message data returned to client"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ConversationDetailResponse(BaseModel):
    """Detailed conversation response including messages"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]