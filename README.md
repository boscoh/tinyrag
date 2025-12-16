# TinyRAG

A lightweight Retrieval Augmented Generation (RAG) starter kit with FastAPI, MCP (Model Context Protocol), and multi-LLM support.

## Features

- **RAG with Semantic Search** - Retrieve relevant information using embeddings
- **Multi-LLM Support** - OpenAI, AWS Bedrock, or local Ollama
- **MCP Integration** - Use Model Context Protocol for tool integration
- **FastAPI Server** - REST API with web interface
- **Docker Ready** - Containerized deployment

## Quick Start

### Prerequisites

- Python 3.13+
- `uv` package manager

### Installation

```bash
uv sync
```

### Configuration

Create a `.env` file:

```bash
LLM_SERVICE=openai
OPENAI_API_KEY=your-api-key-here
```

Supported `LLM_SERVICE` values:
- `openai` - OpenAI API
- `bedrock` - AWS Bedrock
- `ollama` - Local Ollama instance

### Run the Server

```bash
uv run fastapi_server.py --port 8000
```

The server will start and automatically open the browser at `http://localhost:8000`

## API Endpoints

### `/` - Web Interface
The main chat interface.

### `/info` - Service Information
Returns LLM service, chat model, and embedding model details:
```json
{
  "llm_service": "openai",
  "chat_model": "gpt-4o",
  "embed_model": "text-embedding-3-small"
}
```

### `/health` - Health Check
Returns server status.

### `/chat` - Chat API
Process a query using the RAG system.

**Request:**
```json
{
  "query": "Find speakers on machine learning",
  "mode": "assistant",
  "history": []
}
```

**Response:**
```json
{
  "id": "uuid",
  "role": "assistant",
  "status": "success",
  "data": "Response text"
}
```

## Configuration

### Supported Models

**OpenAI:**
- Chat: `gpt-4o`
- Embeddings: `text-embedding-3-small`

**AWS Bedrock:**
- Chat: `amazon.nova-pro-v1:0`
- Embeddings: `amazon.titan-embed-text-v2:0`

**Ollama:**
- Embeddings: `nomic-embed-text`
- Chat tools not supported (embeddings only)

### Environment Variables

- `LLM_SERVICE` - Service to use (required)
- `OPENAI_API_KEY` - OpenAI API key (if using OpenAI)
- `AWS_PROFILE` - AWS profile (if using Bedrock)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials (alternative to profile)
- `CORS_OFF` - Disable CORS if set
- `AWS_REGION` - AWS region (default: us-east-1)

## Development

### Test the MCP Client

```bash
uv run python mcp_client.py
```

This launches an interactive chat interface.

### Generate Embeddings Manually

```bash
uv run python rag.py
```

## Docker

### Build

```bash
docker build -t tinyrag .
```

### Run

```bash
docker run -p 8000:80 -e LLM_SERVICE=openai -e OPENAI_API_KEY=your-key tinyrag
```

Or with environment file:

```bash
docker run -p 8000:80 --env-file .env tinyrag
```

## Project Structure

```
tinyrag/
├── fastapi_server.py    # FastAPI application
├── mcp_server.py        # MCP server implementation
├── mcp_client.py        # MCP client with LLM integration
├── rag.py               # RAG service with embeddings
├── config.py            # Model configuration
├── cli.py               # Command-line interface
├── index.html           # Web UI
└── data/                # Speaker data and embeddings
```

## Architecture

- **FastAPI Server** - REST API and web interface
- **MCP Server** - Tool provider via STDIO
- **MCP Client** - Tool caller with multi-step reasoning
- **RAG Service** - Embeddings and semantic search
- **Chat Client** - Unified LLM interface

## License

MIT
