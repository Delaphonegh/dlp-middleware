from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging

# Import models and dependencies from existing modules
from .auth import get_current_user, conversations_collection
from .models import ConversationCreate, ConversationResponse, ConversationUpdate

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router with correct prefix
router = APIRouter(prefix="/conversations", tags=["conversations"])

# Conversation management functions
def create_conversation(user_id: str, session_id: str, title: str = "New Conversation") -> dict:
    """Create a new conversation thread for a user"""
    conversation_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    conversation_data = {
        "_id": conversation_id,
        "user_id": user_id,
        "session_id": session_id,  # Link to the session
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],  # Will store the conversation history
        "is_active": True
    }
    
    conversations_collection.insert_one(conversation_data)
    logger.info(f"Created new conversation '{title}' for user {user_id}")
    
    return conversation_data

def get_user_conversations(user_id: str) -> List[dict]:
    """Get all conversations for a user"""
    return list(conversations_collection.find(
        {"user_id": user_id, "is_active": True}
    ).sort("updated_at", -1))  # Most recent first

def get_conversation(conversation_id: str, user_id: str) -> dict:
    """Get a specific conversation"""
    conversation = conversations_collection.find_one({
        "_id": conversation_id,
        "user_id": user_id,
        "is_active": True
    })
    
    return conversation

def add_message_to_conversation(
    conversation_id: str, 
    role: str,  # "user" or "assistant" 
    content: str
) -> dict:
    """Add a message to a conversation"""
    message_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    message = {
        "_id": message_id,
        "role": role,
        "content": content,
        "timestamp": now
    }
    
    # Add message and update conversation timestamp
    conversations_collection.update_one(
        {"_id": conversation_id},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": now}
        }
    )
    
    # Update preview of last message (truncated content)
    preview = content[:50] + "..." if len(content) > 50 else content
    conversations_collection.update_one(
        {"_id": conversation_id},
        {"$set": {"last_message_preview": preview}}
    )
    
    return message

def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Soft delete a conversation"""
    result = conversations_collection.update_one(
        {"_id": conversation_id, "user_id": user_id},
        {"$set": {"is_active": False}}
    )
    
    return result.modified_count > 0

# API Endpoints
@router.post("/", response_model=ConversationResponse)
async def create_new_conversation(
    data: ConversationCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new conversation thread"""
    user_id = str(current_user["_id"])
    
    # Create the conversation
    conversation = create_conversation(
        user_id=user_id,
        session_id=data.session_id,
        title=data.title
    )
    
    return {
        "id": conversation["_id"],
        "user_id": conversation["user_id"],
        "title": conversation["title"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": 0,
        "last_message_preview": None
    }

@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(current_user: dict = Depends(get_current_user)):
    """List all conversations for the current user"""
    user_id = str(current_user["_id"])
    conversations = get_user_conversations(user_id)
    
    result = []
    for conv in conversations:
        result.append({
            "id": conv["_id"],
            "user_id": conv["user_id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "message_count": len(conv.get("messages", [])),
            "last_message_preview": conv.get("last_message_preview")
        })
    
    return result

@router.get("/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation_detail(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get details of a conversation including all messages"""
    user_id = str(current_user["_id"])
    conversation = get_conversation(conversation_id, user_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Convert MongoDB _id to string in messages
    messages = []
    for msg in conversation.get("messages", []):
        msg_copy = msg.copy()
        if "_id" in msg_copy:
            msg_copy["id"] = str(msg_copy.pop("_id"))
        messages.append(msg_copy)
    
    return {
        "id": conversation["_id"],
        "user_id": conversation["user_id"],
        "title": conversation["title"],
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "messages": messages
    }

@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    data: ConversationUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update conversation details (currently just title)"""
    user_id = str(current_user["_id"])
    conversation = get_conversation(conversation_id, user_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Update the title
    conversations_collection.update_one(
        {"_id": conversation_id},
        {"$set": {"title": data.title}}
    )
    
    # Get updated conversation
    updated = get_conversation(conversation_id, user_id)
    
    return {
        "id": updated["_id"],
        "user_id": updated["user_id"],
        "title": updated["title"],
        "created_at": updated["created_at"],
        "updated_at": updated["updated_at"],
        "message_count": len(updated.get("messages", [])),
        "last_message_preview": updated.get("last_message_preview")
    }

@router.delete("/{conversation_id}")
async def delete_user_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation"""
    user_id = str(current_user["_id"])
    success = delete_conversation(conversation_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or already deleted"
        )
    
    return {"message": "Conversation deleted successfully"}