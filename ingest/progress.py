"""SQLite-backed ingestion progress tracker for resume capability."""

import hashlib
import json
import os
import sqlite3

from config import INGEST_DB_PATH


class IngestProgress:
    """Tracks which books have been ingested into the vector store."""

    def __init__(self, db_path: str = INGEST_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ingested_books (
                book_id TEXT,
                source TEXT,
                num_chunks INTEGER,
                file_hash TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (book_id, source)
            )
        """)
        self.conn.commit()

    def is_ingested(self, book_id: str, source: str) -> bool:
        """Check if a book has already been ingested."""
        row = self.conn.execute(
            "SELECT 1 FROM ingested_books WHERE book_id = ? AND source = ?",
            (book_id, source),
        ).fetchone()
        return row is not None

    def mark_ingested(self, book_id: str, source: str, num_chunks: int):
        """Record that a book has been successfully ingested."""
        self.conn.execute(
            """INSERT OR REPLACE INTO ingested_books
               (book_id, source, num_chunks)
               VALUES (?, ?, ?)""",
            (book_id, source, num_chunks),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        rows = self.conn.execute(
            """SELECT source, COUNT(*) as books, SUM(num_chunks) as chunks
               FROM ingested_books GROUP BY source"""
        ).fetchall()

        stats = {"total_books": 0, "total_chunks": 0, "by_source": {}}
        for source, books, chunks in rows:
            stats["by_source"][source] = {"books": books, "chunks": chunks or 0}
            stats["total_books"] += books
            stats["total_chunks"] += chunks or 0

        return stats

    def close(self):
        self.conn.close()
