import argparse
import sys

from config import setup_logging, ensure_dirs


def cmd_scrape(args):
    """Run web scrapers."""
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


def cmd_ingest(args):
    """Ingest scraped books into the vector store."""
    from ingest.processor import IngestProcessor

    print(f"\n{'='*50}")
    print("Starting ingestion pipeline")
    print(f"{'='*50}\n")

    processor = IngestProcessor()
    processor.run(source=args.source, limit=args.limit)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn

    print(f"\n{'='*50}")
    print(f"Starting Arabic RAG API on {args.host}:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_ui(args):
    """Launch the Gradio chat UI."""
    from ui.gradio_app import launch

    print(f"\n{'='*50}")
    print(f"Starting Gradio UI on {args.host}:{args.port}")
    print(f"{'='*50}\n")

    launch(host=args.host, port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(description="Arabic RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape books from websites")
    scrape_parser.add_argument(
        "target", choices=["hindawi", "shamela", "all"], help="Which site to scrape"
    )
    scrape_parser.add_argument("--limit", type=int, default=None)
    scrape_parser.add_argument("--delay", type=float, default=2.0)
    scrape_parser.add_argument("--db-path", type=str, default="")

    # Legacy: support old `python main.py hindawi` syntax
    for target in ("hindawi", "shamela", "all"):
        legacy = subparsers.add_parser(target, help=f"(Legacy) Scrape {target}")
        legacy.add_argument("--limit", type=int, default=None)
        legacy.add_argument("--delay", type=float, default=2.0)
        legacy.add_argument("--db-path", type=str, default="")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest books into vector store"
    )
    ingest_parser.add_argument(
        "--source",
        choices=["shamela", "hindawi", "all"],
        default="all",
        help="Which source to ingest (default: all)",
    )
    ingest_parser.add_argument("--limit", type=int, default=None)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the RAG API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio chat UI")
    ui_parser.add_argument("--host", type=str, default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true", help="Create a public link")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    ensure_dirs()

    if args.command == "scrape":
        cmd_scrape(args)
    elif args.command in ("hindawi", "shamela", "all"):
        # Legacy compatibility
        args.target = args.command
        cmd_scrape(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "ui":
        cmd_ui(args)


if __name__ == "__main__":
    main()
