"""Embedding model wrapper using sentence-transformers."""

import logging

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers model for encoding text."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self._is_e5 = "e5" in model_name.lower()
        logger.info("Embedding model loaded")

    def embed_texts(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
        """Embed document texts. Adds 'passage: ' prefix for E5 models."""
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. Adds 'query: ' prefix for E5 models."""
        text = f"query: {query}" if self._is_e5 else query
        embedding = self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return embedding.tolist()
