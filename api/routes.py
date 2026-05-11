"""API route definitions."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ingest.progress import IngestProgress

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str | None = None
    source: str | None = None
    history: list[ChatMessage] | None = None


@router.post("/query")
async def query(req: QueryRequest, request: Request):
    """Full RAG: retrieve relevant chunks and generate an answer."""
    pipeline = request.app.state.pipeline
    result = await pipeline.query(
        query=req.query,
        top_k=req.top_k,
        category=req.category,
        source=req.source,
        history=[m.model_dump() for m in req.history] if req.history else None,
    )
    return result


@router.get("/search")
async def search(
    q: str,
    top_k: int = 5,
    category: str | None = None,
    source: str | None = None,
    request: Request = None,
):
    """Retrieval only — returns relevant chunks without LLM generation."""
    retriever = request.app.state.retriever
    results = retriever.retrieve(
        query=q, top_k=top_k, category=category, source=source
    )
    return {
        "query": q,
        "results": [
            {
                "text": r["text"][:500],
                "score": round(r["score"], 4),
                **r["metadata"],
            }
            for r in results
        ],
    }


@router.get("/stats")
async def stats():
    """Ingestion statistics."""
    progress = IngestProgress()
    s = progress.get_stats()
    progress.close()
    return s


@router.get("/health")
async def health(request: Request):
    """Service health check."""
    generator = request.app.state.generator
    vectorstore = request.app.state.vectorstore

    return {
        "status": "ok",
        "ollama": generator.check_health(),
        "vectors": vectorstore.count(),
    }
