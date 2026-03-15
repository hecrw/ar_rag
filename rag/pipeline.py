"""RAG pipeline: retrieval + generation."""

from rag.retriever import Retriever
from rag.generator import Generator


class RAGPipeline:
    """Orchestrates retrieval and generation."""

    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    async def query(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        source: str | None = None,
    ) -> dict:
        """Run full RAG: retrieve context, then generate answer."""
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query, top_k=top_k, category=category, source=source
        )

        # Generate answer
        answer = await self.generator.generate(query, chunks)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "text": c["text"][:500],  # truncate for response size
                    "score": round(c["score"], 4),
                    **c["metadata"],
                }
                for c in chunks
            ],
        }
