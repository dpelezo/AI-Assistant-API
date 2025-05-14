# app/utils/logging.py
"""Logging utilities."""
import logging
from logging.config import dictConfig
from app.core.config import settings


def configure_logging():
    """Configure logging for the application."""
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "app": {"handlers": ["console"], "level": log_level, "propagate": False},
            "uvicorn": {"handlers": ["console"], "level": log_level, "propagate": False},
            "fastapi": {"handlers": ["console"], "level": log_level, "propagate": False},
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }
    
    dictConfig(logging_config)
    
    return logging.getLogger("app")