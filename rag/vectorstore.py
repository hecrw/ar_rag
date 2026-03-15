"""ChromaDB vector store wrapper."""

import hashlib
import logging

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class VectorStore:
    """Wraps ChromaDB for storing and searching document chunks."""

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}': "
            f"{self.collection.count()} vectors"
        )

    def upsert(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ):
        """Upsert chunks with their embeddings into the collection.

        Each chunk dict must have 'text' and 'metadata' keys.
        """
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Deterministic ID from book_id + text hash
            book_id = chunk["metadata"].get("book_id", "")
            source = chunk["metadata"].get("source", "")
            text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()[:12]
            chunk_id = f"{source}_{book_id}_{text_hash}"

            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Search for similar chunks.

        Returns list of dicts with 'text', 'metadata', 'score'.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine distance → similarity
            })

        return hits

    def delete_book(self, book_id: str, source: str):
        """Delete all chunks for a specific book."""
        self.collection.delete(
            where={"$and": [{"book_id": book_id}, {"source": source}]}
        )

    def count(self) -> int:
        return self.collection.count()
