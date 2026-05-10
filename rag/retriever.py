"""Retrieval: hybrid (dense + BM25) with optional cross-encoder reranking."""

from rag.bm25_retriever import BM25Retriever
from rag.embedder import Embedder
from rag.fusion import rrf_fuse
from rag.reranker import Reranker
from rag.vectorstore import VectorStore
from utils.text_cleaning import clean_arabic_text
from config import RERANK_FETCH_K, RRF_K, TOP_K


class Retriever:
    """Hybrid retrieval: dense embedding + BM25, fused via RRF, then reranked."""

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        reranker: Reranker | None = None,
        bm25: BM25Retriever | None = None,
        rerank_fetch_k: int = RERANK_FETCH_K,
        rrf_k: int = RRF_K,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.bm25 = bm25
        self.rerank_fetch_k = rerank_fetch_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        category: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        query = clean_arabic_text(query, keep_diacritics=False)
        query_embedding = self.embedder.embed_query(query)

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

        dense_hits = self.vectorstore.search(
            query_embedding, top_k=fetch_k, where=where
        )

        if self.bm25 and self.bm25.loaded:
            bm25_hits = self.bm25.search(query, top_k=fetch_k, where=where)
            fused = rrf_fuse([dense_hits, bm25_hits], k=self.rrf_k)
        else:
            fused = dense_hits

        if self.reranker:
            return self.reranker.rerank(query, fused[:fetch_k], top_k=top_k)
        return fused[:top_k]
