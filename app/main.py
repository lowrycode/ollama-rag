import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.document_manager import DocumentManager
from app.services.retrieval import retrieve_context, generate_prompt_string
from langchain_ollama import ChatOllama

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO in deployment
    format="%(levelname)-9s %(message)s"
)
THIRD_PARTY = [
    "httpx",
    "httpcore",
    "chromadb",
    "posthog",
    "urllib3",
    "ollama",
]
# Disable third party logs by reassigning level
for name in THIRD_PARTY:
    logging.getLogger(name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# Global variables
dm: DocumentManager | None = None
vector_db = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dm, vector_db, llm

    # Startup logic
    logger.info("Loading Document Manager and LLM...")
    dm = DocumentManager()
    vector_db = dm.vector_db
    llm = ChatOllama(model="llama3.2")
    logger.debug("Finished loading Document Manager and LLM")

    yield  # app is now running
    # No shutdown logic


# Initialise app with middleware
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class SyncCheckResponse(BaseModel):
    is_in_sync: bool


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

    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/sync/check", response_model=SyncCheckResponse)
async def check_sync_status():
    return {"is_in_sync": dm.is_in_sync(show_summary=False)}


@app.post("/sync/update", status_code=201, response_model=SyncCheckResponse)
async def sync_documents():
    dm.sync(show_summary=False)
    return {"is_in_sync": dm.is_in_sync(show_summary=False)}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
