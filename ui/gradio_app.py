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

    async def respond(message: str, history: list, top_k: float, source: str):
        message = (message or "").strip()
        history = history or []
        if not message:
            return history, "", ""
        src = None if source == "all" else source
        try:
            result = await pipeline.query(
                query=message,
                top_k=int(top_k),
                source=src,
                history=history,
            )
            answer = result["answer"]
            sources_md = _format_sources(result["sources"])
        except Exception as e:
            logger.exception("Query failed")
            answer = f"**Error:** {e}"
            sources_md = ""
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return new_history, "", sources_md

    with gr.Blocks(title="Arabic RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Arabic RAG\n"
            f"Vectors: **{vector_count}** · "
            f"Ollama: **{'connected' if ollama_ok else 'not reachable'}**"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=500,
                    rtl=True,
                    show_copy_button=True,
                    label="المحادثة / Conversation",
                )
                question = gr.Textbox(
                    placeholder="اكتب سؤالك هنا...",
                    rtl=True,
                    lines=2,
                    show_label=False,
                )
                with gr.Row():
                    submit = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear chat")
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
            inputs=[question, chatbot, top_k, source],
            outputs=[chatbot, question, sources_md],
        )
        question.submit(
            respond,
            inputs=[question, chatbot, top_k, source],
            outputs=[chatbot, question, sources_md],
        )
        clear.click(
            lambda: ([], "", ""),
            outputs=[chatbot, question, sources_md],
        )

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    app = build_app()
    app.queue().launch(server_name=host, server_port=port, share=share)
