#!/usr/bin/env python
"""Wipe the vector store and re-ingest PDFs from scratch.

Usage:
    python scripts/reingest.py                      # uses the default PDFs below
    python scripts/reingest.py --keep path/to.pdf   # ingest without wiping first
    python scripts/reingest.py path/to/a.pdf ...    # wipe, then ingest these
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ingest import discover_pdfs, ingest_pdf  # noqa: E402
from app.main import DOCUMENTS_DIR, VECTOR_STORE_DIR, retriever  # noqa: E402

DEFAULT_PDFS = [
    os.path.join("documents", "2026-04-20_Macroeconomic-Forecast-April-2026.pdf"),
    "/Users/Denis/Desktop/Books/fundamentals-of-data-engineering.pdf",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", help="PDF files to ingest.")
    parser.add_argument("--keep", action="store_true",
                        help="Do not wipe the store first (dedup still applies).")
    parser.add_argument("--all", action="store_true",
                        help="Ingest every known PDF, not just the one in DOCUMENTS_DIR.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    log = logging.getLogger("reingest")

    if args.pdfs:
        paths = args.pdfs
    elif args.all:
        paths = [p for p in DEFAULT_PDFS if os.path.isfile(p)]
    else:
        paths = discover_pdfs(DOCUMENTS_DIR)
    if not paths:
        log.error(
            "No PDFs found. Pass paths explicitly, or drop a PDF into %s.", DOCUMENTS_DIR
        )
        return 1

    wipe = not args.keep
    if wipe:
        log.warning("Wiping %s (%d existing chunks)", VECTOR_STORE_DIR, retriever.vector_store.count())
        retriever.clear()

    for path in paths:
        if not os.path.isfile(path):
            log.error("Skipping missing file: %s", path)
            continue
        started = time.time()
        stats = ingest_pdf(path, retriever)
        log.info(
            "%s -> extracted %d, indexed %d, duplicates %d, empty %d, store now %d (%.1fs)",
            os.path.basename(path), stats["chunks_extracted"], stats["indexed"],
            stats["skipped_duplicates"], stats["skipped_empty"], stats["total_chunks"],
            time.time() - started,
        )

    log.info("Done: %s", retriever.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())