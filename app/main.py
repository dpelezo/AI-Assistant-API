# app/main.py
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.router import router
from app.core.config import settings
from app.api.dependencies import active_threads
from app.core.langsmith_setup import setup_langsmith, wrap_functions_with_langsmith

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    
    # Initialize LangSmith if configured
    if settings.is_langsmith_enabled:
        langsmith_enabled = setup_langsmith(settings.LANGSMITH_PROJECT)
        if langsmith_enabled:
            # Wrap OpenAI client with LangSmith tracing
            wrap_success = wrap_functions_with_langsmith()
            if wrap_success:
                logger.info("LangSmith tracing enabled and configured with function wrapping")
            else:
                logger.info("LangSmith tracing enabled but function wrapping failed")
        else:
            logger.warning("LangSmith setup failed despite configuration")
    else:
        logger.info("LangSmith tracing not enabled - set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to enable")

    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    active_threads.clear()

app = FastAPI(
    title="AI Assistant API",
    description="An API for interacting with an AI assistant powered by Claude and Exa Search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "langsmith_enabled": settings.is_langsmith_enabled,
        "version": "1.0.0"
    }

# Register routes
app.include_router(router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)