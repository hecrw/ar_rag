import argparse
import sys

from config import setup_logging, ensure_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Arabic RAG Web Scraper - Scrape Hindawi.org and Shamela.ws"
    )
    parser.add_argument(
        "target",
        choices=["hindawi", "shamela", "all"],
        help="Which site to scrape",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of books to scrape (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="",
        help="Path to Shamela .bok files directory (required for shamela target)",
    )

    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if args.target in ("hindawi", "all"):
        from scrapers.hindawi import HindawiScraper

        scraper = HindawiScraper(delay=args.delay)
        print(f"\n{'='*50}")
        print("Starting Hindawi.org scraper")
        print(f"{'='*50}\n")
        scraper.run(limit=args.limit)

    if args.target in ("shamela", "all"):
        from scrapers.shamela import ShamelaScraper

        scraper = ShamelaScraper(db_path=args.db_path, delay=args.delay)
        print(f"\n{'='*50}")
        print("Starting Shamela.ws database extractor")
        print(f"{'='*50}\n")
        scraper.run(limit=args.limit)

    print("\nDone!")


if __name__ == "__main__":
    main()
