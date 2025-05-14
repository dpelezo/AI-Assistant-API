# AI Assistant API

A FastAPI server implementation that integrates Claude AI with Exa Search capabilities using LangGraph for conversation workflow management. Enhanced with LangSmith for observability and debugging.

## Features

- **FastAPI Backend**: Well-structured, high-performance API with clean architecture
- **Claude Integration**: Integration with Anthropic's Claude 3.5 Sonnet model
- **Web Search**: Real-time information retrieval with Exa Search
- **Conversation Management**: Thread-based conversation handling with LangGraph
- **Asynchronous Processing**: Background task processing for improved responsiveness
- **Docker Support**: Container-based deployment for consistency across environments
- **Structured Testing**: Comprehensive test suite with pytest
- **LangSmith Observability**: Complete tracing and debugging of AI workflows and tool calls
- **Deep Research**: Multi-iteration research capability with contextual synthesis

## Project Structure

```
app/
├── api/                    # API layer
│   ├── dependencies.py     # Dependency injection
│   ├── endpoints/          # API endpoints
│   │   ├── chat.py         # Chat-related endpoints
│   │   └── research.py     # Deep research endpoints
│   └── router.py           # API router configuration
├── core/                   # Core application components
│   ├── config.py           # Application settings
│   ├── workflow.py         # LangGraph workflow
│   └── langsmith_setup.py  # LangSmith tracing utilities
├── models/                 # Data models
│   ├── chat.py             # Chat-related models
│   └── state.py            # State management models
├── services/               # Business logic
│   ├── ai.py               # AI model service
│   ├── memory.py           # Memory management
│   ├── web_search.py       # Web search capabilities
│   └── deep_research.py    # Deep research capabilities
├── tests/                  # Test suite
│   ├── conftest.py         # Test configuration
│   └── test_api.py         # API tests
├── utils/                  # Utilities
│   └── logging.py          # Logging configuration
└── main.py                 # Application entry point
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- Anthropic API key
- Exa API key
- LangSmith API key (optional, for observability)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Configuration
DEBUG=True
SECRET_KEY=your_secret_key

# Model Settings
ANTHROPIC_API_KEY=your_anthropic_api_key
MODEL_NAME=claude-3-5-sonnet-20240620
TEMPERATURE=0

# Search Settings
EXA_API_KEY=your_exa_api_key

# LangSmith Settings (optional, for observability)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your_project_name

# Research Settings (optional)
DEFAULT_RESEARCH_ITERATIONS=3
MAX_RESEARCH_ITERATIONS=5

# CORS Settings (optional)
# BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

### Running with Docker

The easiest way to start the server is with Docker Compose:

```bash
# Start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

### Running Locally

You can also run the server locally using the provided script:

```bash
# Make the script executable
chmod +x run.sh

# Run the server
./run.sh

# Run with custom port
./run.sh --port 9000

# Run without auto-reload
./run.sh --no-reload
```

### Manual Setup

If you prefer to set up manually:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

## API Endpoints

### Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Process a query and return a response |
| `/api/thread/{thread_id}` | GET | Retrieve a conversation thread |
| `/api/thread/{thread_id}` | DELETE | Delete a conversation thread |
| `/api/debug/threads` | GET | List all active threads (debug) |
| `/api/debug/thread/{thread_id}` | GET | View raw thread data (debug) |
| `/api/debug/tool-calls/{thread_id}` | GET | View tool calls in a thread (debug) |

### Research Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/research/query` | POST | Start a deep research process |
| `/api/research/status/{research_id}` | GET | Check research status |
| `/api/research/result/{research_id}` | GET | Get research results |
| `/api/research/debug/latest` | GET | View latest research (debug) |
| `/api/research/debug/all` | GET | List all research jobs (debug) |
| `/api/research/debug/report/{research_id}` | GET | Get research report by ID (debug) |

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/` | GET | Root endpoint with API information |
| `/docs` | GET | API documentation (Swagger UI) |
| `/redoc` | GET | API documentation (ReDoc) |

## LangSmith Observability

The application is integrated with LangSmith for comprehensive tracing and debugging:

1. **Setup**: Provide your `LANGSMITH_API_KEY` as an environment variable
2. **Tracing**: All AI interactions, tool calls, and research processes are traced
3. **Dashboard**: View traces at [smith.langchain.com](https://smith.langchain.com)
4. **Debugging**: Debug tool calls and research processes in detail

For more information, see the [LangSmith Integration Guide](docs/langsmith-guide.md).

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest app/tests/test_api.py
```

## Development

### Adding New Endpoints

1. Create a new file in `app/api/endpoints/`
2. Define your router and endpoints
3. Include your router in `app/api/router.py`

### Modifying the AI Model

To change the AI model settings, update the configuration in `app/core/config.py` or set the corresponding environment variables.

## License

MIT