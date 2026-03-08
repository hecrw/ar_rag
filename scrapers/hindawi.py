import os
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from config import HINDAWI_BASE_URL, HINDAWI_DATA_DIR, HINDAWI_BOOKS_DIR
from scrapers.base import BaseScraper
from utils.text_cleaning import clean_arabic_text, clean_html_text


class HindawiScraper(BaseScraper):
    """Scraper for hindawi.org Arabic books library."""

    CATALOG_FILE = os.path.join(HINDAWI_DATA_DIR, "catalog.json")
    TOTAL_LISTING_PAGES = 215  # ~20 books per page

    def scrape_catalog(self) -> list[dict]:
        """Scrape the full book catalog from listing pages.

        Returns list of {id, title, url} dicts.
        """
        # Resume from saved catalog if available
        existing = self.load_json(self.CATALOG_FILE)
        if existing:
            self.logger.info(f"Loaded existing catalog with {len(existing)} books")
            return existing

        self.logger.info("Scraping Hindawi book catalog...")
        catalog = []
        consecutive_failures = 0

        for page_num in tqdm(range(1, self.TOTAL_LISTING_PAGES + 1), desc="Catalog pages"):
            url = f"{HINDAWI_BASE_URL}/books/{page_num}/"
            response = self.fetch(url)
            if response is None:
                consecutive_failures += 1
                self.logger.warning(f"Failed to fetch listing page {page_num}")
                if consecutive_failures >= 5:
                    self.logger.info("5 consecutive failures — assuming end of catalog")
                    break
                continue

            consecutive_failures = 0
            soup = BeautifulSoup(response.text, "lxml")
            book_items = soup.find_all("li", class_="bookCover")

            if not book_items:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                continue

            for item in book_items:
                link = item.find("a", href=re.compile(r"/books/\d+/"))
                if not link:
                    continue

                href = link.get("href", "")
                match = re.search(r"/books/(\d+)/", href)
                if not match:
                    continue

                book_id = match.group(1)

                # Title is in the img alt attribute: "كتاب بعنوان <title>"
                img = link.find("img")
                title = ""
                if img and img.get("alt"):
                    alt = img["alt"]
                    # Remove prefix "كتاب بعنوان " (Book titled)
                    title = re.sub(r"^كتاب بعنوان\s*", "", alt).strip()
                if not title:
                    title = link.get_text(strip=True)

                catalog.append({
                    "id": book_id,
                    "title": title,
                    "url": f"{HINDAWI_BASE_URL}/books/{book_id}/",
                })

            # Save incrementally every 50 pages
            if page_num % 50 == 0:
                self.save_json(catalog, self.CATALOG_FILE)

        self.logger.info(f"Found {len(catalog)} books in catalog")
        self.save_json(catalog, self.CATALOG_FILE)
        return catalog

    def scrape_book(self, book_id: str, book_url: str, book_title: str) -> dict | None:
        """Scrape a single book: metadata + all chapters."""
        filepath = os.path.join(HINDAWI_BOOKS_DIR, f"{book_id}.json")
        if self.book_exists(filepath):
            self.logger.debug(f"Skipping {book_id} (already scraped)")
            return None

        # Fetch book main page for metadata
        response = self.fetch(book_url)
        if response is None:
            self.logger.warning(f"Failed to fetch book page: {book_url}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Extract metadata
        author = ""
        author_el = soup.find(class_="author")
        if author_el:
            author = author_el.get_text(strip=True)
        else:
            # Find contributor links, skip generic "المساهمون" (Contributors) label
            for a in soup.find_all("a", href=re.compile(r"/contributors/")):
                name = a.get_text(strip=True)
                if name and name != "المساهمون":
                    author = name
                    break

        description = ""
        # The content div on book pages holds the description
        content_div = soup.find(class_="content")
        if content_div:
            # Get first paragraph as description
            first_p = content_div.find("p")
            if first_p:
                description = clean_arabic_text(first_p.get_text(strip=True))

        category = ""
        # Find the book's category from the details section
        details_el = soup.find(class_="details")
        if details_el:
            cat_link = details_el.find("a", href=re.compile(r"/categories/"))
            if cat_link:
                category = cat_link.get_text(strip=True)
                # Remove trailing number (e.g., "مسرحيات٢٣٤" -> "مسرحيات")
                category = re.sub(r"[\d٠-٩]+$", "", category).strip()

        # Find chapter links
        chapter_links = []
        for a in soup.find_all("a", href=re.compile(rf"/books/{book_id}/\d+/")):
            href = a.get("href", "")
            match = re.search(rf"/books/{book_id}/(\d+)/", href)
            if match:
                ch_num = int(match.group(1))
                ch_title = a.get_text(strip=True)
                if ch_num not in [c["number"] for c in chapter_links]:
                    chapter_links.append({
                        "number": ch_num,
                        "title": ch_title,
                        "url": f"{HINDAWI_BASE_URL}/books/{book_id}/{ch_num}/",
                    })

        # Sort by chapter number
        chapter_links.sort(key=lambda c: c["number"])

        # If no chapter links found, try sequential discovery
        if not chapter_links:
            chapter_links = self._discover_chapters(book_id)

        # Scrape each chapter
        chapters = []
        for ch_info in chapter_links:
            ch_data = self._scrape_chapter(ch_info)
            if ch_data:
                chapters.append(ch_data)

        if not chapters:
            self.logger.warning(f"No chapters found for book {book_id}")
            return None

        book_data = {
            "id": book_id,
            "title": book_title,
            "author": author,
            "category": category,
            "description": description,
            "url": book_url,
            "chapters": chapters,
        }

        self.save_json(book_data, filepath)
        return book_data

    def _discover_chapters(self, book_id: str) -> list[dict]:
        """Try sequential chapter numbers until 404."""
        chapters = []
        for ch_num in range(1, 200):  # reasonable upper bound
            url = f"{HINDAWI_BASE_URL}/books/{book_id}/{ch_num}/"
            response = self.fetch(url)
            if response is None:
                break
            chapters.append({
                "number": ch_num,
                "title": "",
                "url": url,
            })
        return chapters

    def _scrape_chapter(self, ch_info: dict) -> dict | None:
        """Scrape a single chapter page."""
        response = self.fetch(ch_info["url"])
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Try to find the chapter content area
        content = (
            soup.find("article")
            or soup.find(class_="chapter-content")
            or soup.find(class_="content")
            or soup.find(id="content")
        )

        if content is None:
            # Fallback: get the main body text
            content = soup.find("main") or soup.find("body")

        text = clean_html_text(content) if content else ""

        # Get chapter title from the page if not already known
        title = ch_info.get("title", "")
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        return {
            "number": ch_info["number"],
            "title": title,
            "text": text,
            "url": ch_info["url"],
        }

    def run(self, limit: int | None = None):
        """Run the full scraping pipeline.

        Args:
            limit: Max number of books to scrape (None for all).
        """
        catalog = self.scrape_catalog()

        if limit:
            catalog = catalog[:limit]

        self.logger.info(f"Scraping {len(catalog)} books from Hindawi...")

        scraped = 0
        skipped = 0

        for book in tqdm(catalog, desc="Books"):
            filepath = os.path.join(HINDAWI_BOOKS_DIR, f"{book['id']}.json")
            if self.book_exists(filepath):
                skipped += 1
                continue

            result = self.scrape_book(book["id"], book["url"], book["title"])
            if result:
                scraped += 1

        self.logger.info(
            f"Done: {scraped} scraped, {skipped} skipped (already existed)"
        )
