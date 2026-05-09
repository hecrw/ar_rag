"""Stream books from JSON files one at a time."""

import json
import os
from collections.abc import Iterator

from config import HINDAWI_BOOKS_DIR, SHAMELA_BOOKS_DIR


def iter_books(source: str) -> Iterator[tuple[str, dict]]:
    """Yield (source, book_dict) for each book JSON file.

    Reads one file at a time to keep memory usage low.
    """
    if source in ("shamela", "all"):
        yield from _iter_dir("shamela", SHAMELA_BOOKS_DIR)
    if source in ("hindawi", "all"):
        yield from _iter_dir("hindawi", HINDAWI_BOOKS_DIR)


def _iter_dir(source: str, books_dir: str) -> Iterator[tuple[str, dict]]:
    """Iterate over JSON files in a directory."""
    if not os.path.isdir(books_dir):
        return

    for entry in sorted(os.scandir(books_dir), key=lambda e: e.name):
        if not entry.name.endswith(".json"):
            continue
        try:
            with open(entry.path, "rb") as f:
                book = json.loads(f.read().decode("utf-8", errors="replace"))
            yield source, book
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skipping {entry.name}: {e}")
            continue


def count_books(source: str) -> int:
    """Count total book JSON files for progress bars."""
    total = 0
    if source in ("shamela", "all") and os.path.isdir(SHAMELA_BOOKS_DIR):
        total += sum(1 for f in os.scandir(SHAMELA_BOOKS_DIR) if f.name.endswith(".json"))
    if source in ("hindawi", "all") and os.path.isdir(HINDAWI_BOOKS_DIR):
        total += sum(1 for f in os.scandir(HINDAWI_BOOKS_DIR) if f.name.endswith(".json"))
    return total
