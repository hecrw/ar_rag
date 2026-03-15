"""Ingestion processor: read books → chunk → embed → store."""

import logging

from tqdm import tqdm

from config import EMBED_BATCH_SIZE
from ingest.reader import iter_books, count_books
from ingest.progress import IngestProgress
from rag.chunker import chunk_shamela_book, chunk_hindawi_book
from rag.embedder import Embedder
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class IngestProcessor:
    """Orchestrates the full ingestion pipeline."""

    def __init__(self):
        self.embedder = Embedder()
        self.vectorstore = VectorStore()
        self.progress = IngestProgress()

    def run(self, source: str = "all", limit: int | None = None):
        """Ingest books into the vector store.

        Args:
            source: "shamela", "hindawi", or "all"
            limit: Max books to process (None for all)
        """
        total = count_books(source)
        logger.info(f"Found {total} book files to process")

        stats = self.progress.get_stats()
        logger.info(
            f"Already ingested: {stats['total_books']} books, "
            f"{stats['total_chunks']} chunks"
        )

        processed = 0
        skipped = 0
        errors = 0
        total_chunks = 0

        pbar = tqdm(iter_books(source), total=total, desc="Ingesting books")

        for book_source, book in pbar:
            if limit and processed >= limit:
                break

            book_id = str(book.get("id", ""))
            if not book_id:
                errors += 1
                continue

            # Skip if already ingested
            if self.progress.is_ingested(book_id, book_source):
                skipped += 1
                pbar.set_postfix(done=processed, skip=skipped, err=errors)
                continue

            try:
                # Chunk the book
                if book_source == "shamela":
                    chunks = chunk_shamela_book(book)
                else:
                    chunks = chunk_hindawi_book(book)

                if not chunks:
                    errors += 1
                    continue

                # Embed and store in batches
                for i in range(0, len(chunks), EMBED_BATCH_SIZE):
                    batch = chunks[i : i + EMBED_BATCH_SIZE]
                    texts = [c["text"] for c in batch]
                    embeddings = self.embedder.embed_texts(texts)
                    self.vectorstore.upsert(batch, embeddings)

                # Mark as done
                self.progress.mark_ingested(book_id, book_source, len(chunks))
                processed += 1
                total_chunks += len(chunks)
                pbar.set_postfix(done=processed, skip=skipped, chunks=total_chunks)

            except Exception as e:
                logger.warning(f"Error processing book {book_id}: {e}")
                errors += 1

        pbar.close()
        self.progress.close()

        logger.info(
            f"Ingestion complete: {processed} books processed, "
            f"{total_chunks} chunks embedded, {skipped} skipped, {errors} errors"
        )
        logger.info(f"Total vectors in store: {self.vectorstore.count()}")
