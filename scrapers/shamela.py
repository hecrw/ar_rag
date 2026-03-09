"""Shamela.ws scraper — extracts books from the downloaded Shamela database.

The Shamela desktop database is a collection of .bok files (MS Access .mdb format).
Each .bok file contains one or more books with Arabic text.

Setup:
  1. Download the full Shamela database from:
     https://archive.org/download/shamela4_official/shamela.full.1446.1.iso
  2. Mount or extract the ISO
  3. Point this scraper at the directory containing .bok files:
     python main.py shamela --db-path /path/to/shamela/Books/

The .bok file schema (MS Access / MDB format):
  - Table 'Main': metadata with columns BkId, Bk (title), Betaka (intro)
  - Table 'b{BkId}': page content with columns id, nass (text), page, part
  - Table 't{BkId}': table of contents with columns id, tit, lvl, sub
"""

import os
import re

from tqdm import tqdm

from config import SHAMELA_DATA_DIR, SHAMELA_BOOKS_DIR, SHAMELA_BASE_URL
from scrapers.base import BaseScraper
from utils.text_cleaning import clean_arabic_text


class ShamelaScraper(BaseScraper):
    """Extracts books from the downloaded Shamela database (.bok files)."""

    CATALOG_FILE = os.path.join(SHAMELA_DATA_DIR, "catalog.json")

    def __init__(self, db_path: str = "", **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path

    def _find_bok_files(self) -> list[str]:
        """Find all .bok files in the database directory."""
        if not self.db_path:
            self.logger.error(
                "No database path provided. Use --db-path to specify the "
                "directory containing .bok files from the Shamela download."
            )
            return []

        if not os.path.isdir(self.db_path):
            self.logger.error(f"Directory not found: {self.db_path}")
            return []

        bok_files = []
        for root, _, files in os.walk(self.db_path):
            for f in files:
                if f.lower().endswith(".bok"):
                    bok_files.append(os.path.join(root, f))

        self.logger.info(f"Found {len(bok_files)} .bok files")
        return sorted(bok_files)

    def _parse_bok_file(self, bok_path: str) -> list[dict]:
        """Parse a single .bok file and extract all books from it.

        A .bok file may contain one or more books.
        Returns a list of book dicts.
        """
        from access_parser import AccessParser

        try:
            db = AccessParser(bok_path)
        except Exception as e:
            self.logger.warning(f"Cannot parse {bok_path}: {e}")
            return []

        books = []

        # Read the Main table for book metadata
        try:
            main_table = db.parse_table("Main")
        except Exception:
            # Some .bok files may have different structure
            self.logger.debug(f"No 'Main' table in {bok_path}")
            return []

        if not main_table:
            return []

        # Main table columns: BkId, Bk (title), Betaka (intro), Auth (author), ...
        bk_ids = main_table.get("BkId", [])
        bk_names = main_table.get("Bk", [])
        betakas = main_table.get("Betaka", [])
        authors = main_table.get("Auth", main_table.get("auth", []))

        for i in range(len(bk_ids)):
            book_id = str(bk_ids[i]) if i < len(bk_ids) else ""
            title = self._decode_field(bk_names[i]) if i < len(bk_names) else ""
            betaka = self._decode_field(betakas[i]) if i < len(betakas) else ""

            if not book_id:
                continue

            # Extract author from Auth column or from Betaka text
            author = ""
            if i < len(authors):
                author = self._decode_field(authors[i]).strip()
            if not author and betaka:
                author = self._extract_author_from_betaka(betaka)

            # Read body table: b{BkId}
            body_table_name = f"b{book_id}"
            try:
                body = db.parse_table(body_table_name)
            except Exception:
                self.logger.debug(f"No body table '{body_table_name}' in {bok_path}")
                continue

            if not body:
                continue

            # Extract pages
            pages = []
            nass_col = body.get("nass", body.get("Nass", []))
            id_col = body.get("id", body.get("Id", list(range(1, len(nass_col) + 1))))
            page_col = body.get("page", body.get("Page", []))
            part_col = body.get("part", body.get("Part", []))

            for j in range(len(nass_col)):
                raw_text = self._decode_field(nass_col[j])
                text = self._clean_nass(raw_text)
                if not text:
                    continue

                page_data = {
                    "number": id_col[j] if j < len(id_col) else j + 1,
                    "text": text,
                }
                if j < len(page_col) and page_col[j]:
                    page_data["page"] = page_col[j]
                if j < len(part_col) and part_col[j]:
                    page_data["part"] = part_col[j]

                pages.append(page_data)

            if not pages:
                continue

            # Read TOC table: t{BkId} (optional)
            toc = []
            toc_table_name = f"t{book_id}"
            try:
                toc_table = db.parse_table(toc_table_name)
                if toc_table:
                    tit_col = toc_table.get("tit", toc_table.get("Tit", []))
                    lvl_col = toc_table.get("lvl", toc_table.get("Lvl", []))
                    for k in range(len(tit_col)):
                        heading = self._decode_field(tit_col[k])
                        if heading:
                            toc.append({
                                "title": heading,
                                "level": lvl_col[k] if k < len(lvl_col) else 0,
                            })
            except Exception:
                pass

            book_data = {
                "id": book_id,
                "title": title,
                "author": author,
                "description": clean_arabic_text(betaka) if betaka else "",
                "url": f"{SHAMELA_BASE_URL}/book/{book_id}",
                "total_pages": len(pages),
                "pages": pages,
                "source_file": os.path.basename(bok_path),
            }
            if toc:
                book_data["toc"] = toc

            books.append(book_data)

        return books

    @staticmethod
    def _extract_author_from_betaka(betaka: str) -> str:
        """Extract author name from the Betaka (description) field.

        The Betaka often contains a line like:
          المؤلف: ابن كثير
        or
          المؤلف : أبو حامد الغزالي
        """
        # Match "المؤلف" followed by optional space, colon, then the name
        match = re.search(r"المؤلف\s*:\s*(.+?)(?:\n|$)", betaka)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _decode_field(value) -> str:
        """Decode a field value from the MDB database."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            # Try UTF-8 first, then Windows-1256 (common Arabic encoding)
            for encoding in ("utf-8", "cp1256", "iso-8859-6"):
                try:
                    return value.decode(encoding)
                except (UnicodeDecodeError, AttributeError):
                    continue
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _clean_nass(text: str) -> str:
        """Clean the nass (text content) field from Shamela."""
        if not text:
            return ""
        # Remove HTML tags if present
        if "<" in text and ">" in text:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(separator="\n", strip=True)
        return clean_arabic_text(text)

    def run(self, limit: int | None = None):
        """Parse .bok files and save each book as JSON.

        Args:
            limit: Max number of books to extract (None for all).
        """
        if not self.db_path:
            self.logger.error(
                "\n"
                "========================================\n"
                "  Shamela database path required!\n"
                "\n"
                "  1. Download the Shamela database:\n"
                "     https://archive.org/download/shamela4_official/shamela.full.1446.1.iso\n"
                "  2. Mount or extract the ISO\n"
                "  3. Run with --db-path:\n"
                "     python main.py shamela --db-path /path/to/shamela/Books/\n"
                "========================================"
            )
            return

        bok_files = self._find_bok_files()
        if not bok_files:
            return

        scraped = 0
        skipped = 0

        for bok_path in tqdm(bok_files, desc="Processing .bok files"):
            books = self._parse_bok_file(bok_path)

            for book in books:
                if limit and scraped >= limit:
                    self.logger.info(f"Reached limit of {limit} books")
                    self.logger.info(
                        f"Done: {scraped} extracted, {skipped} skipped"
                    )
                    return

                filepath = os.path.join(SHAMELA_BOOKS_DIR, f"{book['id']}.json")
                if self.book_exists(filepath):
                    skipped += 1
                    continue

                self.save_json(book, filepath)
                scraped += 1

        self.logger.info(
            f"Done: {scraped} extracted, {skipped} skipped (already existed)"
        )
