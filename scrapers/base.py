import json
import logging
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import HEADERS, REQUEST_DELAY, REQUEST_TIMEOUT, MAX_RETRIES, BACKOFF_FACTOR


class BaseScraper:
    """Base scraper with session management, rate limiting, and retry logic."""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = self._create_session()
        self._last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(HEADERS)

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _rate_limit(self):
        """Enforce delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def fetch(self, url: str) -> requests.Response | None:
        """Fetch a URL with rate limiting and error handling.

        Returns the Response on success, or None on failure.
        """
        self._rate_limit()
        self._last_request_time = time.time()

        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"HTTP error fetching {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None

    @staticmethod
    def save_json(data: dict | list, filepath: str):
        """Save data as JSON with Arabic text preserved."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(filepath: str) -> dict | list | None:
        """Load JSON file if it exists."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def book_exists(filepath: str) -> bool:
        """Check if a book JSON file already exists (for resume)."""
        return os.path.exists(filepath)
