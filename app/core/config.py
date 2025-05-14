# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API settings
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "supersecretkey"
    
    # Model settings
    ANTHROPIC_API_KEY: str = ""
    EXA_API_KEY: str = ""
    LLM_MODEL: str = "claude-3-5-sonnet-20240620"
    LLM_TEMPERATURE: float = 0.0
    
    # Deep Research settings
    MAX_RESEARCH_ITERATIONS: int = 5
    DEFAULT_RESEARCH_ITERATIONS: int = 3
    
    # LangSmith settings
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "ai-assistant-api"
    LANGSMITH_CALLBACKS_BACKGROUND: bool = True
    
    # Debug settings
    DEBUG: bool = False
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }
    
    @property
    def is_langsmith_enabled(self) -> bool:
        """Check if LangSmith is properly configured."""
        return bool(self.LANGSMITH_API_KEY and self.LANGSMITH_TRACING)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG


settings = Settings()