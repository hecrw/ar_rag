"""BM25 retrieval over the same chunk corpus stored in the vector store."""

import logging
import os
import pickle

import bm25s

from config import BM25_INDEX_DIR
from utils.text_cleaning import clean_arabic_text

logger = logging.getLogger(__name__)


def _matches(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_matches(meta, w) for w in where["$or"])
    for k, v in where.items():
        if meta.get(k) != v:
            return False
    return True


class BM25Retriever:
    """Tokenized BM25 index persisted alongside the Chroma collection."""

    def __init__(self, index_dir: str = BM25_INDEX_DIR):
        self.index_dir = index_dir
        self.retriever: bm25s.BM25 | None = None
        self.docs: list[dict] | None = None

    @property
    def loaded(self) -> bool:
        return self.retriever is not None and self.docs is not None

    def load(self) -> bool:
        meta_path = os.path.join(self.index_dir, "docs.pkl")
        if not os.path.exists(meta_path):
            logger.warning(
                f"No BM25 index at {self.index_dir}. "
                f"Run `python main.py bm25-build` to create one."
            )
            return False
        with open(meta_path, "rb") as f:
            self.docs = pickle.load(f)
        self.retriever = bm25s.BM25.load(self.index_dir, mmap=True)
        logger.info(f"Loaded BM25 index: {len(self.docs)} docs")
        return True

    def build(self, docs: list[dict]) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        self.docs = docs
        corpus = [clean_arabic_text(d["text"], keep_diacritics=False) for d in docs]
        logger.info(f"Tokenizing {len(corpus)} docs for BM25...")
        tokens = bm25s.tokenize(corpus, stopwords=None, show_progress=True)
        logger.info("Indexing...")
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=True)
        self.retriever.save(self.index_dir)
        with open(os.path.join(self.index_dir, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)
        logger.info(f"BM25 index saved to {self.index_dir}")

    def search(
        self,
        query: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[dict]:
        if not self.loaded:
            return []
        q = clean_arabic_text(query, keep_diacritics=False)
        tokens = bm25s.tokenize([q], stopwords=None, show_progress=False)
        fetch_k = top_k * 5 if where else top_k
        fetch_k = min(fetch_k, len(self.docs))
        if fetch_k <= 0:
            return []
        indices, scores = self.retriever.retrieve(
            tokens, k=fetch_k, show_progress=False
        )
        out = []
        for idx, score in zip(indices[0], scores[0]):
            d = self.docs[int(idx)]
            if where and not _matches(d.get("metadata") or {}, where):
                continue
            out.append({
                "id": d["id"],
                "text": d["text"],
                "metadata": d.get("metadata") or {},
                "score": float(score),
            })
            if len(out) >= top_k:
                break
        return out
