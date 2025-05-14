# app/api/router.py
from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.endpoints import chat, research
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(chat.router, tags=["chat"])
router.include_router(research.router, prefix="/research", tags=["research"])

# Add a simple info endpoint
@router.get("/", tags=["info"])
async def get_api_info():
    """Get basic information about the API."""
    return {
        "name": "AI Assistant API",
        "version": "1.0.0",
        "features": [
            "Chat with AI assistant",
            "Web search capabilities",
            "Deep research functionality",
            "LangSmith observability" if settings.is_langsmith_enabled else "LangSmith observability (disabled)"
        ],
        "endpoints": {
            "chat": f"{settings.API_PREFIX}/query",
            "research": f"{settings.API_PREFIX}/research/query"
        },
        "observability": {
            "enabled": settings.is_langsmith_enabled,
            "project": settings.LANGSMITH_PROJECT if settings.is_langsmith_enabled else None
        }
    }

