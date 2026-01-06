# RAG Query Assistant

A Retrieval Augmented Generation (RAG) system that combines Ollama LLM with Langchain and Chroma vector database to answer questions based on custom documents.

## Overview

This project implements a complete RAG pipeline with:
- **Document Ingestion** - Processes PDF, DOCX, and TXT files
- **Vector Database** - Stores document embeddings using Chroma
- **REST API** - FastAPI backend for querying
- **Web Interface** - Modern, responsive UI for interacting with the system

## Current Capabilities

### What It Can Do
- ✅ Ingest and index multiple documents (PDF, DOCX, TXT)
- ✅ Answer questions based on indexed document content
- ✅ Retrieve relevant source documents for each answer
- ✅ Display source relevance scores (0-100%)
- ✅ Show answer confidence based on source relevance
- ✅ Provide expandable source content snippets
- ✅ Web-based query interface with modern UI

### API Endpoint
**POST** `/query`
```json
{
  "question": "Your question here",
  "top_k": 4
}
```

Response includes:
- Generated answer
- List of source documents with content and relevance scores

## Project Structure

```
ollama-rag/
├── app/                          # Backend API
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Configuration
│   ├── data/                     # Document storage
│   ├── db/                       # Vector database (Chroma)
│   └── services/
│       ├── document_manager.py   # Document ingestion & indexing
│       └── retrieval.py          # Query & response generation
├── frontend/                     # Web UI
│   └── index.html                # Query interface
├── dev/                          # Development utilities
├── requirements.txt
└── README.md
```

## Tech Stack

- **LLM**: Ollama (llama3.2)
- **Framework**: Langchain
- **Vector DB**: Chroma
- **API**: FastAPI
- **Frontend**: HTML/CSS/JavaScript

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place documents in `app/data/`

3. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Access the UI at `http://localhost:8000`

## Development Status

This project is in active development. Early testing phase with core RAG functionality implemented.