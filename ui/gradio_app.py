"""Gradio UI for the Arabic RAG system."""

import logging

import gradio as gr

from config import BM25_ENABLED, RERANK_ENABLED, TOP_K, setup_logging
from rag.bm25_retriever import BM25Retriever
from rag.embedder import Embedder
from rag.generator import Generator
from rag.pipeline import RAGPipeline
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "—")
        author = s.get("author", "—")
        score = s.get("score", 0)
        text = s.get("text", "")
        lines.append(
            f"**[{i}] {title} — {author}** _(score: {score})_\n\n> {text}\n"
        )
    return "\n---\n".join(lines)


def build_app() -> gr.Blocks:
    setup_logging()
    logger.info("Loading RAG components for Gradio UI...")
    embedder = Embedder()
    vectorstore = VectorStore()
    reranker = Reranker() if RERANK_ENABLED else None
    bm25 = None
    if BM25_ENABLED:
        bm25 = BM25Retriever()
        if not bm25.load():
            bm25 = None
    retriever = Retriever(embedder, vectorstore, reranker=reranker, bm25=bm25)
    generator = Generator()
    pipeline = RAGPipeline(retriever, generator)

    ollama_ok = generator.check_health()
    vector_count = vectorstore.count()
    logger.info(
        f"Ready. Ollama reachable: {ollama_ok}, vectors in store: {vector_count}"
    )

    async def respond(message: str, top_k: float, source: str):
        message = (message or "").strip()
        if not message:
            return "Please enter a question.", ""
        src = None if source == "all" else source
        try:
            result = await pipeline.query(
                query=message, top_k=int(top_k), source=src
            )
        except Exception as e:
            logger.exception("Query failed")
            return f"Error: {e}", ""
        return result["answer"], _format_sources(result["sources"])

    with gr.Blocks(title="Arabic RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Arabic RAG\n"
            f"Vectors: **{vector_count}** · "
            f"Ollama: **{'connected' if ollama_ok else 'not reachable'}**"
        )

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="السؤال / Question",
                    placeholder="اكتب سؤالك هنا...",
                    rtl=True,
                    lines=2,
                )
                with gr.Row():
                    submit = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                answer = gr.Textbox(
                    label="الإجابة / Answer", rtl=True, lines=10
                )
            with gr.Column(scale=1):
                top_k = gr.Slider(
                    1, 20, value=TOP_K, step=1, label="Top K (sources)"
                )
                source = gr.Dropdown(
                    ["all", "shamela", "hindawi"],
                    value="all",
                    label="Source filter",
                )

        gr.Markdown("### Retrieved sources")
        sources_md = gr.Markdown()

        submit.click(
            respond,
            inputs=[question, top_k, source],
            outputs=[answer, sources_md],
        )
        question.submit(
            respond,
            inputs=[question, top_k, source],
            outputs=[answer, sources_md],
        )
        clear.click(
            lambda: ("", "", ""),
            outputs=[question, answer, sources_md],
        )

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    app = build_app()
    app.queue().launch(server_name=host, server_port=port, share=share)
