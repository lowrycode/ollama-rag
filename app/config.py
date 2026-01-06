from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "db"
CACHE_DIR = BASE_DIR / ".cache"

# Models / settings
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
DEFAULT_COLLECTION = "simple-rag"

# Chunking options
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
