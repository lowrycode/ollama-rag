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

### API Endpoints

**POST `/query`**

Retrieve an answer using the indexed documents as context.

Request body:
```json
{
  "question": "Your question here",
  "top_k": 4
}
```
***NOTE:*** *`top_k` controls how many document chunks are used as context and retrieved as sources in response.*

Response:
```json
{
  "answer": "Generated answer text",
  "sources": [
    {
      "file": "/abs/path/document.pdf",
      "content": "Relevant excerpt from document",
      "relevance": 0.82
    }
  ]
}
```

---

**GET `/sync`**

Check whether the vector database is currently in sync with local documents.

Response:
```json
{
  "is_in_sync": true,
  "last_synced_at": "2026-01-10T15:48:37+00:00"
}
```

---

**POST `/sync`**

Trigger a synchronization between local documents and the vector database.
- Detects added, updated, and removed files
- Updates the vector database accordingly

Response:
```json
{
  "is_in_sync": true,
  "last_synced_at": "2026-01-10T15:48:37+00:00"
}
```

---

**GET `/health`**

Simple health check endpoint.

Response:
```json
{
  "status": "ok"
}
```


## Project Structure

```
ollama-rag/
├── app/                          # Backend API
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Configuration
│   ├── .cache/                   # Cache & metadata storage
│   │   └── file_index.json       # Local metadata cache for syncing
│   ├── data/                     # Document storage
│   ├── db/                       # Vector database (Chroma)
│   └── services/
│       ├── document_manager.py   # Document ingestion & indexing
│       └── retrieval.py          # Query & response generation
├── frontend/                     # Web UI
│   └── index.html                # Query interface
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

3. Ensure Ollama is running in the background

4. Start the server from project root:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Access the UI at `http://localhost:8000`

## How to Reset and Load Your Own Documents

The project includes sample documents in the `app/data/` folder (statutory guidance for schools in England, taken from www.gov.uk). The documents have been pre-loaded into the vector database so you can start querying immediately.

To start fresh with your own documents:
1. **Delete the `app/.cache/` and `app/db/` directories:**  
    *These will be re-created automatically by the app when needed.*
2. **Replace the documents in the `app/data/` directory:**  
    *These will be added to the vector database when you run the sync procedure via the UI.*

***NOTE:*** *Depending on the size and number of documents, the sync procedure may take a while! It took around 25 mins to add these documents to the vector database.*

## Deployment

### Local vs Production LLMs

This project uses **Ollama** and is well suited for **local development and private use**.

**Benefits of Ollama (Local Development)**
- Runs entirely on your machine (no external API calls)
- Ideal for working with sensitive or private documents
- No usage costs or rate limits imposed by external providers

**For production deployments**, running a local LLM is not recommended due to:
- High CPU/GPU and memory requirements
- Limited scalability under concurrent user load
- Increased operational complexity (model management, resource limits)

In production, it is preferable to **use a hosted LLM provider** (e.g. OpenAI, Google, Anthropic) for model inference. This requires only swapping one line of code in `app/main.py` inside the lifespan function:

``` python
# Change this:
llm = ChatOllama(model="llama3.2")

# to a different model, e.g. OpenAI GPT-4:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

The rest of the codebase remains unchanged.

For a complete list of models supported by LangChain, see the [LangChain Providers & Integrations Overview](https://docs.langchain.com/oss/python/integrations/providers/overview).
