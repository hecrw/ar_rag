"""Retrieval: query embedding + vector search."""

from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from utils.text_cleaning import clean_arabic_text
from config import TOP_K


class Retriever:
    """Combines embedding + vector search for retrieval."""

    def __init__(self, embedder: Embedder, vectorstore: VectorStore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        category: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        """Retrieve top-k relevant chunks for a query."""
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

        return self.vectorstore.search(query_embedding, top_k=top_k, where=where)
