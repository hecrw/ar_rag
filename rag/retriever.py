"""Retrieval: query embedding + vector search (+ optional reranking)."""

from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.vectorstore import VectorStore
from utils.text_cleaning import clean_arabic_text
from config import RERANK_FETCH_K, TOP_K


class Retriever:
    """Combines embedding + vector search for retrieval, with optional rerank."""

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        reranker: Reranker | None = None,
        rerank_fetch_k: int = RERANK_FETCH_K,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.rerank_fetch_k = rerank_fetch_k

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        category: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        """Retrieve top-k relevant chunks for a query.

        If a reranker is configured, fetch ``rerank_fetch_k`` candidates from
        the vector store first, then cross-encode them down to ``top_k``.
        """
        # Normalize query (strip diacritics for better matching)
        query = clean_arabic_text(query, keep_diacritics=False)

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Build metadata filter
        where = None
        filters = []
        if category:
            filters.append({"category": category})
        if source:
            filters.append({"source": source})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        fetch_k = max(self.rerank_fetch_k, top_k) if self.reranker else top_k
        candidates = self.vectorstore.search(
            query_embedding, top_k=fetch_k, where=where
        )

        if self.reranker:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates
