"""LLM generation via Ollama."""

import logging

import httpx

from config import OLLAMA_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """أنت مساعد عربي متخصص في الإجابة على الأسئلة بناءً على النصوص المقدمة.
أجب على السؤال باستخدام المعلومات الموجودة في السياق المقدم فقط.
إذا لم تجد الإجابة في السياق، قل "لم أجد إجابة في النصوص المتاحة".
اذكر المصادر (عنوان الكتاب والمؤلف) عند الإجابة.
استخدم تنسيق Markdown في إجابتك:
- العناوين الفرعية بـ ## أو ###
- النقاط بـ - أو *
- التأكيد بـ **النص العريض** أو *المائل*
- الاقتباسات المباشرة من المصادر بـ > مع ذكر رقم المصدر."""


class Generator:
    """Generates answers using Ollama."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_URL,
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120)

    async def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer given a query and retrieved context chunks."""
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            source_info = f"[{i}] {meta.get('title', '')} — {meta.get('author', '')}"
            context_parts.append(f"{source_info}\n{chunk['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""السياق:
{context}

السؤال: {query}

الإجابة:"""

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def check_health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            r = self.client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False
