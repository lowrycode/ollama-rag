from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.document_manager import DocumentManager
from app.services.retrieval import retrieve_context, generate_prompt_string
from langchain_ollama import ChatOllama

# Initialise app with middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict after testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DocumentManager and LLM once
dm = DocumentManager()
dm.update_db_sync()
vector_db = dm.vector_db  # assumes vector DB is ready and loaded
llm = ChatOllama(model="llama3.2")


# Pydantic Schema
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4  # how many docs to retrieve


class SourceItem(BaseModel):
    file: str
    content: str
    relevance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# Endpoint functions
@app.post("/query", response_model=QueryResponse)
async def query_with_context(req: QueryRequest):
    try:
        results = retrieve_context(
            vector_db=vector_db, query=req.question, top_k=req.top_k
        )
        if not results:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join(r["content"] for r in results)
        sources = [
            {
                "file": r["source"],
                "content": r["content"],
                "relevance": r["relevance"],
            }
            for r in results
        ]

        prompt = generate_prompt_string(context, req.question)
        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}
