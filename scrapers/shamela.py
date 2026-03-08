import json
import os
import re
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

from config import SHAMELA_BASE_URL, SHAMELA_DATA_DIR, SHAMELA_BOOKS_DIR, HEADERS
from scrapers.base import BaseScraper
from utils.text_cleaning import clean_arabic_text


class ShamelaScraper(BaseScraper):
    """Scraper for shamela.ws Islamic sciences library.

    Uses requests for catalog pages (authors index) and Playwright
    for book content pages (which are behind Cloudflare protection).
    """

    CATALOG_FILE = os.path.join(SHAMELA_DATA_DIR, "catalog.json")
    COOKIES_FILE = os.path.join(SHAMELA_DATA_DIR, "cookies.json")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    def _init_browser(self):
        """Initialize Playwright browser for Cloudflare-protected pages.

        Opens a visible browser window. On first run, the user must
        solve the Cloudflare challenge (click the checkbox). The session
        cookies are then saved and reused.
        """
        if self._browser is not None:
            return

        from playwright.sync_api import sync_playwright

        self.logger.info(
            "Starting browser for Shamela scraping. "
            "If a Cloudflare challenge appears, please solve it manually."
        )

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=False)

        # Load saved cookies if available
        storage_state = None
        if os.path.exists(self.COOKIES_FILE):
            storage_state = self.COOKIES_FILE
            self.logger.info("Loading saved cookies")

        self._context = self._browser.new_context(
            locale="ar-SA",
            timezone_id="Asia/Riyadh",
            storage_state=storage_state,
        )
        self._page = self._context.new_page()

    def _save_cookies(self):
        """Save browser cookies for reuse across sessions."""
        if self._context:
            state = self._context.storage_state()
            os.makedirs(os.path.dirname(self.COOKIES_FILE), exist_ok=True)
            with open(self.COOKIES_FILE, "w") as f:
                json.dump(state, f)

    def _close_browser(self):
        """Clean up browser resources."""
        if self._browser:
            self._save_cookies()
            self._browser.close()
            self._playwright.stop()
            self._browser = None
            self._context = None
            self._page = None
            self._playwright = None

    def _fetch_with_browser(self, url: str, wait_selector: str = ".nass") -> str | None:
        """Fetch a page using Playwright, waiting for content to load.

        Returns the page HTML on success, None on failure.
        """
        self._init_browser()
        self._rate_limit()
        self._last_request_time = time.time()

        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for content or detect Cloudflare challenge
            try:
                self._page.wait_for_selector(wait_selector, timeout=10000)
            except Exception:
                # Check if we're on Cloudflare challenge
                title = self._page.title()
                if "لحظة" in title or "moment" in title.lower():
                    self.logger.info(
                        "Cloudflare challenge detected. Please solve it in the browser window..."
                    )
                    # Wait for user to solve challenge (up to 120 seconds)
                    try:
                        self._page.wait_for_selector(wait_selector, timeout=120000)
                        # Save cookies after solving challenge
                        self._save_cookies()
                    except Exception:
                        self.logger.warning(f"Timed out waiting for challenge resolution: {url}")
                        return None
                else:
                    # Page loaded but no matching selector - might be 404 or different structure
                    return None

            return self._page.content()

        except Exception as e:
            self.logger.warning(f"Browser error fetching {url}: {e}")
            return None

    def scrape_catalog(self) -> list[dict]:
        """Build catalog by scraping author pages to collect all book IDs.

        Uses requests (no Cloudflare on author pages).
        Returns list of {id, title, author, url} dicts.
        """
        existing = self.load_json(self.CATALOG_FILE)
        if existing:
            self.logger.info(f"Loaded existing catalog with {len(existing)} books")
            return existing

        self.logger.info("Scraping Shamela book catalog via authors index...")

        # Step 1: Get all author IDs
        author_ids = self._scrape_author_ids()
        self.logger.info(f"Found {len(author_ids)} authors")

        # Step 2: For each author, get their book list
        catalog = []
        seen_book_ids = set()

        for author_id, author_name in tqdm(author_ids, desc="Authors"):
            books = self._scrape_author_books(author_id, author_name)
            for book in books:
                if book["id"] not in seen_book_ids:
                    seen_book_ids.add(book["id"])
                    catalog.append(book)

            # Save incrementally every 100 authors
            if len(catalog) % 500 == 0 and catalog:
                self.save_json(catalog, self.CATALOG_FILE)

        self.logger.info(f"Found {len(catalog)} unique books in catalog")
        self.save_json(catalog, self.CATALOG_FILE)
        return catalog

    def _scrape_author_ids(self) -> list[tuple[str, str]]:
        """Scrape the authors index page for (author_id, author_name) pairs."""
        url = f"{SHAMELA_BASE_URL}/authors"
        response = self.fetch(url)
        if response is None:
            self.logger.error("Failed to fetch authors index")
            return []

        soup = BeautifulSoup(response.text, "lxml")
        authors = []
        seen = set()

        for a in soup.find_all("a", href=re.compile(r"/author/\d+")):
            href = a.get("href", "")
            match = re.search(r"/author/(\d+)", href)
            if match:
                author_id = match.group(1)
                if author_id not in seen:
                    seen.add(author_id)
                    author_name = a.get_text(strip=True)
                    authors.append((author_id, author_name))

        return authors

    def _scrape_author_books(self, author_id: str, author_name: str) -> list[dict]:
        """Get all books from an author's page."""
        url = f"{SHAMELA_BASE_URL}/author/{author_id}"
        response = self.fetch(url)
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        books = []
        seen = set()

        for a in soup.find_all("a", href=re.compile(r"/book/\d+")):
            href = a.get("href", "")
            match = re.search(r"/book/(\d+)", href)
            if match:
                book_id = match.group(1)
                title = a.get_text(strip=True)
                if title and book_id not in seen:
                    seen.add(book_id)
                    books.append({
                        "id": book_id,
                        "title": title,
                        "author": author_name,
                        "url": f"{SHAMELA_BASE_URL}/book/{book_id}",
                    })

        return books

    def scrape_book(self, book_id: str, book_title: str, author: str) -> dict | None:
        """Scrape a single book using the browser."""
        filepath = os.path.join(SHAMELA_BOOKS_DIR, f"{book_id}.json")
        if self.book_exists(filepath):
            self.logger.debug(f"Skipping {book_id} (already scraped)")
            return None

        pages = []
        page_num = 1
        consecutive_failures = 0

        while consecutive_failures < 3:
            url = f"{SHAMELA_BASE_URL}/book/{book_id}/{page_num}"
            html = self._fetch_with_browser(url)

            if html is None:
                consecutive_failures += 1
                if page_num == 1:
                    self.logger.warning(f"Cannot access book {book_id}")
                    return None
                page_num += 1
                continue

            consecutive_failures = 0
            text = self._extract_page_text(html)

            if not text:
                page_num += 1
                consecutive_failures += 1
                continue

            pages.append({
                "number": page_num,
                "text": text,
                "url": url,
            })

            page_num += 1

        if not pages:
            self.logger.warning(f"No pages found for book {book_id}")
            return None

        # Extract title from first page if possible
        first_html = self._fetch_with_browser(
            f"{SHAMELA_BASE_URL}/book/{book_id}/1", wait_selector="h1"
        )
        actual_title = book_title
        if first_html:
            soup = BeautifulSoup(first_html, "lxml")
            h1 = soup.find("h1")
            if h1:
                actual_title = h1.get_text(strip=True) or book_title

        book_data = {
            "id": book_id,
            "title": actual_title,
            "author": author,
            "url": f"{SHAMELA_BASE_URL}/book/{book_id}",
            "total_pages": len(pages),
            "pages": pages,
        }

        self.save_json(book_data, filepath)
        return book_data

    def _extract_page_text(self, html: str) -> str:
        """Extract text content from a Shamela book page."""
        soup = BeautifulSoup(html, "lxml")

        # Primary selector: div.nass contains the book text
        nass_divs = soup.find_all(class_="nass")
        if nass_divs:
            parts = []
            for nass in nass_divs:
                paragraphs = nass.find_all("p")
                if paragraphs:
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text:
                            parts.append(text)
                else:
                    text = nass.get_text(strip=True)
                    if text:
                        parts.append(text)
            return clean_arabic_text("\n".join(parts))

        # Fallback: try the wrapper div
        wrapper = soup.find(id="wrapper")
        if wrapper:
            return clean_arabic_text(wrapper.get_text(separator="\n", strip=True))

        return ""

    def run(self, limit: int | None = None):
        """Run the full scraping pipeline.

        Args:
            limit: Max number of books to scrape (None for all).
        """
        try:
            catalog = self.scrape_catalog()

            if limit:
                catalog = catalog[:limit]

            self.logger.info(f"Scraping {len(catalog)} books from Shamela...")

            scraped = 0
            skipped = 0

            for book in tqdm(catalog, desc="Books"):
                filepath = os.path.join(SHAMELA_BOOKS_DIR, f"{book['id']}.json")
                if self.book_exists(filepath):
                    skipped += 1
                    continue

                result = self.scrape_book(
                    book["id"], book["title"], book.get("author", "")
                )
                if result:
                    scraped += 1

            self.logger.info(
                f"Done: {scraped} scraped, {skipped} skipped (already existed)"
            )
        finally:
            self._close_browser()
