"""FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import RERANK_ENABLED, setup_logging
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.generator import Generator
from rag.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    setup_logging()

    embedder = Embedder()
    vectorstore = VectorStore()
    reranker = Reranker() if RERANK_ENABLED else None
    retriever = Retriever(embedder, vectorstore, reranker=reranker)
    generator = Generator()
    pipeline = RAGPipeline(retriever, generator)

    app.state.retriever = retriever
    app.state.generator = generator
    app.state.pipeline = pipeline
    app.state.vectorstore = vectorstore

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arabic RAG API",
        description="Retrieval-Augmented Generation for Arabic books",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from api.routes import router
    app.include_router(router, prefix="/api")

    return app


app = create_app()
