"""Cross-encoder reranker for retrieved chunks."""

import logging

import torch
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reranker:
    """Reorders candidate chunks by cross-encoder relevance score."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        device = _pick_device()
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        logger.info(
            f"Loading reranker: {model_name} on {device} "
            f"({'fp16' if model_kwargs else 'fp32'})"
        )
        self.model = CrossEncoder(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            max_length=512,
        )
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        batch_size: int = 32,
    ) -> list[dict]:
        """Score (query, candidate.text) pairs and return the top_k by score."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        out = []
        for score, cand in ranked[:top_k]:
            out.append({**cand, "score": float(score)})
        return out
