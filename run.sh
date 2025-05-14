#!/bin/bash

# Script to run the API server with various options

# Default values
PORT=8000
RELOAD=true
WORKERS=1
LOG_LEVEL="info"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-reload)
      RELOAD=false
      shift
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port NUMBER        Port to run the server on (default: 8000)"
      echo "  --no-reload          Disable auto-reload for development"
      echo "  --workers NUMBER     Number of worker processes (default: 1)"
      echo "  --log-level LEVEL    Log level (debug|info|warning|error) (default: info)"
      echo "  --help               Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information."
      exit 1
      ;;
  esac
done

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Check if langsmith is configured
if [ -n "$LANGSMITH_API_KEY" ]; then
  echo "LangSmith configured with API key."
  export LANGSMITH_TRACING=true
  echo "LangSmith tracing enabled."
else
  echo "No LangSmith API key found. Observability features will be limited."
fi

# Prepare reload flag
if [ "$RELOAD" = true ]; then
  RELOAD_FLAG="--reload"
else
  RELOAD_FLAG=""
fi

# Print startup information
echo "Starting API server on port $PORT with log level $LOG_LEVEL"
echo "Auto-reload: $RELOAD"
echo "Workers: $WORKERS"

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port $PORT $RELOAD_FLAG --workers $WORKERS --log-level $LOG_LEVEL