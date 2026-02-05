# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **FinTech RAG System** - a financial document retrieval-augmented generation system built with Python, LangChain, FastAPI, and ChromaDB. The system processes PDF documents and provides AI-powered question-answering capabilities using Zhipu AI's GLM-4 model.

## Development Commands

```bash
# Install dependencies (requires Poetry)
poetry install

# Run development server (auto-reload on file changes)
poetry run uvicorn src.fintech_rag.api.app:app --reload

# Run directly with Python
cd src/fintech-rag && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Build and run with Docker
docker-compose up --build

# Run Docker without Compose
docker build -t fintech-rag .
docker run -p 8000:8000 -e ZHIPUAI_API_KEY=your_key fintech-rag
```

## Required Environment Variables

Create a `.env` file in the project root:
```
ZHIPUAI_API_KEY=your_api_key_here
STATIC_DIR=static  # Optional, defaults to searching for static/ directory
```

## Architecture

The system follows a **layered architecture** with clear separation of concerns:

### 1. API Layer ([`src/fintech-rag/api/app.py`](src/fintech-rag/api/app.py))
- **FastAPI** web application serving both REST API and web UI
- **LangChain Agent** with tool integration
- Serves static HTML/JS frontend from [`static/`](static/)
- **Endpoints:**
  - `GET /` - Web UI for agent chat
  - `POST /api/query` - Query processing through the agent

### 2. Core RAG System ([`src/fintech-rag/core/`](src/fintech-rag/core/))

**[`huggingFace_rag.py`](src/fintech-rag/core/huggingFace_rag.py)** - Document processing and retrieval:
- Uses **HuggingFace embeddings** (BAAI/bge-base-en-v1.5)
- Processes PDF documents with **PyPDFLoader**
- **RecursiveCharacterTextSplitter** for chunking (chunk_size=500, overlap=50)
- **ChromaDB** vector storage with persistence in `./chroma_db/`
- Similarity search with k=4 retrieval

**[`zhipu_llm.py`](src/fintech-rag/core/zhipu_llm.py)** - Zhipu AI LLM wrapper (exists but not actively used in current agent implementation)

### 3. Vector Database
- **ChromaDB** for persistent document embeddings
- Stored in [`src/fintech-rag/core/chroma_db/`](src/fintech-rag/core/chroma_db/)
- Data persists across runs

### 4. LLM Integration
- **Zhipu AI GLM-4 model** via `langchain_community.chat_models.zhipuai.ChatZhipuAI`
- Model: `glm-4`, temperature configurable (0.1 for RAG, 0.7 for agent)
- Requires valid `ZHIPUAI_API_KEY`

### 5. Frontend ([`static/index.html`](static/index.html))
- Modern chat interface with real-time message updates
- JavaScript-based API client for `/api/query` endpoint
- Includes example queries and error handling

## Key Implementation Details

### RAG Chain Pattern
The RAG system uses LangChain's LCEL (LangChain Expression Language) pattern:
```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Agent Tools
Current tool in [`app.py`](src/fintech-rag/api/app.py:44-50):
- `get_current_datetime()` - Returns current date/time

To add new tools:
1. Define a function decorated with `@tool`
2. Add docstring explaining what the tool does
3. Include the tool in `create_agent(llm, tools=[...])` call

### Static File Serving
The API has robust static file path resolution that:
1. Checks `STATIC_DIR` environment variable
2. Falls back to searching parent directories for `static/` folder
3. Raises error if static directory cannot be found

## Project Structure Notes

- **Empty directories**: [`models/`](src/fintech-rag/models/), [`services/`](src/fintech-rag/services/), [`tests/`](tests/), [`scripts/`](scripts/) - reserved for future expansion
- **Sample document**: [`docs/caching-at-scale-with-redis-updated-2021-12-04.pdf`](docs/caching-at-scale-with-redis-updated-2021-12-04.pdf) for testing RAG functionality
- **Docker setup**: Uses Python 3.11 Alpine, Poetry for dependency management, Uvicorn as ASGI server

## Port Configuration

- Default port: **8000**
- Host: `0.0.0.0` (for Docker compatibility)
- Access at: http://localhost:8000
