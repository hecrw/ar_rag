"""Arabic text chunking for RAG ingestion."""

import re

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from utils.text_cleaning import clean_arabic_text


# Sentence-ending patterns for Arabic and Latin
_SENTENCE_END = re.compile(r"(?<=[.!?؟।۔])\s+")
_PARAGRAPH_END = re.compile(r"\n\n+")


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Recursively split text into chunks with overlap.

    Tries to split at paragraph boundaries first, then sentences, then
    at any whitespace.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    # Try splitting at paragraph boundaries
    chunks = _split_at_separator(text, _PARAGRAPH_END, max_chars, overlap)
    if chunks:
        return chunks

    # Try splitting at sentence boundaries
    chunks = _split_at_separator(text, _SENTENCE_END, max_chars, overlap)
    if chunks:
        return chunks

    # Fall back to splitting at whitespace near max_chars
    return _split_at_whitespace(text, max_chars, overlap)


def _split_at_separator(
    text: str, pattern: re.Pattern, max_chars: int, overlap: int
) -> list[str] | None:
    """Split text at the given regex separator, keeping chunks under max_chars."""
    parts = pattern.split(text)
    if len(parts) <= 1:
        return None

    chunks = []
    current = ""

    for part in parts:
        candidate = (current + "\n\n" + part).strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current and len(current) >= MIN_CHUNK_SIZE:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            if current and overlap > 0:
                tail = current[-overlap:]
                current = tail + "\n\n" + part
            else:
                current = part

            # If current part alone exceeds max, force-split it
            if len(current) > max_chars:
                sub_chunks = _split_at_whitespace(current, max_chars, overlap)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""

    if current and len(current) >= MIN_CHUNK_SIZE:
        chunks.append(current)

    return chunks if chunks else None


def _split_at_whitespace(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text at whitespace boundaries near max_chars."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) >= MIN_CHUNK_SIZE:
                chunks.append(chunk)
            break

        # Find last whitespace before end
        split_pos = text.rfind(" ", start, end)
        if split_pos <= start:
            split_pos = end  # no whitespace found, hard split

        chunk = text[start:split_pos].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)

        # Move start back by overlap amount
        start = split_pos - overlap if overlap > 0 else split_pos

    return chunks


def chunk_shamela_book(book: dict) -> list[dict]:
    """Chunk a Shamela book into pieces with metadata.

    Shamela books have pages (short text units). We concatenate short pages
    before chunking.
    """
    pages = book.get("pages", [])
    if not pages:
        return []

    # Concatenate all page text with page markers
    full_text = ""
    for page in pages:
        page_text = page.get("text", "")
        footnotes = page.get("footnotes", "")
        if footnotes:
            page_text = page_text + "\n" + footnotes
        if page_text:
            full_text += page_text + "\n\n"

    full_text = clean_arabic_text(full_text, keep_diacritics=False)
    if not full_text:
        return []

    raw_chunks = _split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    return [
        {
            "text": chunk,
            "metadata": {
                "book_id": str(book.get("id", "")),
                "source": "shamela",
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "category": book.get("category", ""),
            },
        }
        for chunk in raw_chunks
    ]


def chunk_hindawi_book(book: dict) -> list[dict]:
    """Chunk a Hindawi book into pieces with metadata.

    Hindawi books have chapters (long text units). Each chapter is chunked
    independently.
    """
    chapters = book.get("chapters", [])
    if not chapters:
        return []

    all_chunks = []
    for chapter in chapters:
        chapter_text = clean_arabic_text(
            chapter.get("text", ""), keep_diacritics=False
        )
        if not chapter_text:
            continue

        raw_chunks = _split_text(chapter_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in raw_chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "book_id": str(book.get("id", "")),
                    "source": "hindawi",
                    "title": book.get("title", ""),
                    "author": book.get("author", ""),
                    "category": book.get("category", ""),
                    "chapter": chapter.get("title", ""),
                },
            })

    return all_chunks
