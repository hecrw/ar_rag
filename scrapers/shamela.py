"""Shamela.ws scraper — extracts books from the downloaded Shamela database.

The Shamela4 desktop app stores data across:
  - master.db (SQLite): book catalog, authors, categories
  - database/store/page/ (Lucene 9.x index): actual page text (7M+ docs)
  - database/store/author/ (Lucene 9.x index): author biographies

A Java helper tool (tools/LuceneDump.java) reads the Lucene indexes.

Setup:
  1. Download: https://archive.org/download/shamela4_official/shamela.full.1446.1.iso
  2. Extract on Windows using shamela.exe (the .bin inside is encrypted)
  3. Copy the extracted 'database/' folder to your project
  4. Run: python main.py shamela --db-path /path/to/shamela4/
"""

import json
import os
import re
import sqlite3
import subprocess

from bs4 import BeautifulSoup
from tqdm import tqdm

from config import SHAMELA_DATA_DIR, SHAMELA_BOOKS_DIR, SHAMELA_BASE_URL
from scrapers.base import BaseScraper
from utils.text_cleaning import clean_arabic_text

# Path to the Java tool and Lucene JARs (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAVA_TOOL_DIR = os.path.join(PROJECT_ROOT, "tools")
JAVA_BIN = "/opt/homebrew/opt/openjdk@21/bin/java"


class ShamelaScraper(BaseScraper):
    """Extracts books from the Shamela4 database (SQLite + Lucene)."""

    CATALOG_FILE = os.path.join(SHAMELA_DATA_DIR, "catalog.json")

    def __init__(self, db_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self._lucene_jars = ""
        self._page_index = ""
        self._author_index = ""

    def _setup_paths(self) -> bool:
        """Validate paths and set up index locations."""
        if not self.db_path:
            self.logger.error(
                "\n"
                "========================================\n"
                "  Shamela database path required!\n"
                "\n"
                "  1. Download the Shamela database ISO\n"
                "  2. Extract on Windows using shamela.exe\n"
                "  3. Copy the extracted folder to your project\n"
                "  4. Run with --db-path:\n"
                "     python main.py shamela --db-path /path/to/shamela4/\n"
                "========================================"
            )
            return False

        # Find master.db
        master_db = os.path.join(self.db_path, "database", "master.db")
        if not os.path.exists(master_db):
            # Maybe they pointed directly to the database/ dir
            master_db = os.path.join(self.db_path, "master.db")
        if not os.path.exists(master_db):
            self.logger.error(f"master.db not found in {self.db_path}")
            return False
        self._master_db = master_db

        # Find Lucene indexes
        base = os.path.dirname(master_db)
        self._page_index = os.path.join(base, "store", "page")
        self._author_index = os.path.join(base, "store", "author")

        if not os.path.isdir(self._page_index):
            self.logger.error(f"Page index not found: {self._page_index}")
            return False

        # Find Lucene JARs
        lucene_dir = os.path.join(self.db_path, "app", "lucene", "1")
        if not os.path.isdir(lucene_dir):
            # Try parent
            lucene_dir = os.path.join(os.path.dirname(self.db_path), "app", "lucene", "1")
        if not os.path.isdir(lucene_dir):
            self.logger.error(
                f"Lucene JARs not found. Expected at: {lucene_dir}\n"
                "Make sure --db-path points to the extracted shamela4/ root folder."
            )
            return False
        self._lucene_jars = os.path.join(lucene_dir, "*")

        # Check Java
        if not os.path.exists(JAVA_BIN):
            self.logger.error(
                f"Java not found at {JAVA_BIN}. "
                "Install with: brew install openjdk@21"
            )
            return False

        # Compile Java tool if needed
        class_file = os.path.join(JAVA_TOOL_DIR, "LuceneDump.class")
        java_file = os.path.join(JAVA_TOOL_DIR, "LuceneDump.java")
        if not os.path.exists(class_file) or (
            os.path.getmtime(java_file) > os.path.getmtime(class_file)
        ):
            self.logger.info("Compiling LuceneDump.java...")
            javac = JAVA_BIN.replace("/java", "/javac")
            result = subprocess.run(
                [javac, "-cp", self._lucene_jars, java_file],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                self.logger.error(f"Failed to compile LuceneDump: {result.stderr}")
                return False

        return True

    def _load_catalog(self) -> list[dict]:
        """Load book catalog with author info from master.db."""
        conn = sqlite3.connect(f"file:{self._master_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        books = []
        cursor = conn.cursor()
        cursor.execute("""
            SELECT b.book_id, b.book_name, b.meta_data,
                   GROUP_CONCAT(a.author_name, ' / ') as authors,
                   GROUP_CONCAT(a.author_id) as author_ids,
                   MIN(a.death_text) as death_text,
                   c.category_name
            FROM book b
            LEFT JOIN author_book ab ON b.book_id = ab.book_id
            LEFT JOIN author a ON ab.author_id = a.author_id
            LEFT JOIN category c ON b.book_category = c.category_id
            WHERE b.hidden = 0 OR b.hidden IS NULL
            GROUP BY b.book_id
            ORDER BY b.book_id
        """)

        for row in cursor.fetchall():
            books.append({
                "book_id": row["book_id"],
                "book_name": row["book_name"] or "",
                "authors": row["authors"] or "",
                "author_ids": row["author_ids"] or "",
                "death_text": row["death_text"] or "",
                "category": row["category_name"] or "",
                "meta_data": row["meta_data"] or "",
            })

        conn.close()
        self.logger.info(f"Loaded {len(books)} books from master.db")
        return books

    def _load_author_bios(self) -> dict[str, str]:
        """Load all author biographies from the Lucene author index."""
        if not os.path.isdir(self._author_index):
            self.logger.warning("Author index not found, skipping bios")
            return {}

        self.logger.info("Loading author biographies from Lucene index...")
        result = subprocess.run(
            [JAVA_BIN, "-Xmx512m",
             "-cp", f"{JAVA_TOOL_DIR}:{self._lucene_jars}",
             "LuceneDump", self._author_index],
            capture_output=True, text=True, timeout=120,
        )

        bios = {}
        for line in result.stdout.strip().split("\n"):
            if not line.startswith("{"):
                continue
            try:
                doc = json.loads(line)
                author_id = doc.get("id", "")
                body = doc.get("body_store", "")
                if author_id and body:
                    bios[author_id] = clean_arabic_text(body)
            except json.JSONDecodeError:
                continue

        self.logger.info(f"Loaded {len(bios)} author biographies")
        return bios

    def _extract_pages_batch(self, book_ids: list[int]) -> dict[str, list[dict]]:
        """Extract pages for a batch of books using the Lucene batch mode.

        Returns dict mapping book_id (str) -> list of page dicts.
        """
        # Prepare prefixes: "{book_id}-"
        prefixes = [f"{bid}-" for bid in book_ids]
        input_data = "\n".join(prefixes)

        result = subprocess.run(
            [JAVA_BIN, "-Xmx2g",
             "-cp", f"{JAVA_TOOL_DIR}:{self._lucene_jars}",
             "LuceneDump", self._page_index, "--batch", "id"],
            input=input_data, capture_output=True, text=True,
            timeout=600,
        )

        pages_by_book: dict[str, list[dict]] = {}
        current_book_id = None

        for line in result.stdout.split("\n"):
            if line.startswith("###BOOK:"):
                current_book_id = line[8:].rstrip("-")
                if current_book_id not in pages_by_book:
                    pages_by_book[current_book_id] = []
                continue

            if not line.startswith("{") or current_book_id is None:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = doc.get("id", "")
            body = doc.get("body", "")
            foot = doc.get("foot", "")

            # Parse page number from id: "{book_id}-{page_num}"
            parts = doc_id.split("-", 1)
            page_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            text = self._clean_nass(body)
            if not text:
                continue

            page_data = {
                "number": page_num,
                "text": text,
            }
            if foot:
                page_data["footnotes"] = clean_arabic_text(foot)

            pages_by_book[current_book_id].append(page_data)

        # Sort pages by number within each book
        for bid in pages_by_book:
            pages_by_book[bid].sort(key=lambda p: p["number"])

        return pages_by_book

    @staticmethod
    def _clean_nass(text: str) -> str:
        """Clean the text content field."""
        if not text:
            return ""
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(separator="\n", strip=True)
        return clean_arabic_text(text)

    def run(self, limit: int | None = None):
        """Extract books from Shamela database and save as JSON.

        Args:
            limit: Max number of books to extract (None for all).
        """
        if not self._setup_paths():
            return

        # Load catalog from SQLite
        catalog = self._load_catalog()
        if not catalog:
            self.logger.error("No books found in master.db")
            return

        # Load author bios from Lucene
        author_bios = self._load_author_bios()

        # Filter out already scraped books
        to_scrape = []
        skipped = 0
        for book in catalog:
            filepath = os.path.join(SHAMELA_BOOKS_DIR, f"{book['book_id']}.json")
            if self.book_exists(filepath):
                skipped += 1
            else:
                to_scrape.append(book)

        if limit:
            to_scrape = to_scrape[:limit]

        self.logger.info(
            f"Will scrape {len(to_scrape)} books ({skipped} already exist)"
        )

        if not to_scrape:
            self.logger.info("Nothing to scrape!")
            return

        # Process in batches to avoid huge memory usage
        BATCH_SIZE = 100
        scraped = 0
        errors = 0

        for batch_start in tqdm(
            range(0, len(to_scrape), BATCH_SIZE),
            desc="Processing book batches",
            total=(len(to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE,
        ):
            batch = to_scrape[batch_start:batch_start + BATCH_SIZE]
            book_ids = [b["book_id"] for b in batch]

            # Extract pages for this batch from Lucene
            try:
                pages_by_book = self._extract_pages_batch(book_ids)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Timeout extracting batch at offset {batch_start}")
                errors += len(batch)
                continue
            except Exception as e:
                self.logger.warning(f"Error extracting batch: {e}")
                errors += len(batch)
                continue

            for book_info in batch:
                bid = str(book_info["book_id"])
                pages = pages_by_book.get(bid, [])

                if not pages:
                    errors += 1
                    continue

                # Build author bio
                author_bio = ""
                if book_info["author_ids"]:
                    for aid in book_info["author_ids"].split(","):
                        aid = aid.strip()
                        if aid in author_bios:
                            author_bio = author_bios[aid]
                            break

                book_data = {
                    "id": bid,
                    "title": book_info["book_name"],
                    "author": book_info["authors"],
                    "author_bio": author_bio,
                    "category": book_info["category"],
                    "description": book_info["meta_data"],
                    "url": f"{SHAMELA_BASE_URL}/book/{bid}",
                    "total_pages": len(pages),
                    "pages": pages,
                }

                filepath = os.path.join(SHAMELA_BOOKS_DIR, f"{bid}.json")
                self.save_json(book_data, filepath)
                scraped += 1

        self.logger.info(
            f"Done: {scraped} extracted, {skipped} skipped, {errors} errors"
        )
