import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HINDAWI_DATA_DIR = os.path.join(DATA_DIR, "hindawi")
HINDAWI_BOOKS_DIR = os.path.join(HINDAWI_DATA_DIR, "books")

SHAMELA_DATA_DIR = os.path.join(DATA_DIR, "shamela")
SHAMELA_BOOKS_DIR = os.path.join(SHAMELA_DATA_DIR, "books")

HINDAWI_BASE_URL = "https://www.hindawi.org"
SHAMELA_BASE_URL = "https://shamela.ws"

# Shamela API settings (register at https://dev.shamela.ws to get these)
SHAMELA_API_KEY = os.environ.get("SHAMELA_API_KEY", "")
SHAMELA_API_BOOKS_ENDPOINT = os.environ.get(
    "SHAMELA_API_BOOKS_ENDPOINT", "https://dev.shamela.ws/api/books"
)
SHAMELA_API_MASTER_ENDPOINT = os.environ.get(
    "SHAMELA_API_MASTER_PATCH_ENDPOINT", "https://dev.shamela.ws/api/master_patch"
)
SHAMELA_DB_DIR = os.path.join(SHAMELA_DATA_DIR, "db")

# RAG settings
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
COLLECTION_NAME = "arabic_books"
INGEST_DB_PATH = os.path.join(DATA_DIR, "ingest_progress.db")
BM25_INDEX_DIR = os.path.join(DATA_DIR, "bm25_index")
BM25_ENABLED = os.environ.get("BM25_ENABLED", "1") not in ("0", "false", "False")
RRF_K = 60

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
EMBEDDING_DIMENSION = 768

# Reranker (cross-encoder over (query, chunk) pairs).
# bge-reranker-v2-m3 is multilingual and strong on Arabic. ~568M params, ~1.2GB in fp16.
# If VRAM is tight (8GB GPU + LLM + embedder), switch to "BAAI/bge-reranker-base" (~280MB).
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") not in ("0", "false", "False")
RERANK_FETCH_K = 30  # candidates pulled from vector store before reranking

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
MIN_CHUNK_SIZE = 100  # discard chunks smaller than this
EMBED_BATCH_SIZE = 128
TOP_K = 5

# Request settings
REQUEST_DELAY = 2  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # exponential backoff multiplier

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = logging.INFO


def setup_logging():
    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)


def ensure_dirs():
    for d in [HINDAWI_BOOKS_DIR, SHAMELA_BOOKS_DIR, SHAMELA_DB_DIR, CHROMA_DIR]:
        os.makedirs(d, exist_ok=True)
