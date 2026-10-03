"""PDF ingestion: discovery, extraction and indexing.

Kept separate from the HTTP layer so the same code path serves startup
auto-indexing and the `scripts/reingest.py` maintenance script.
"""

import logging
import os
from typing import Any, Dict, List

from .retriever import HybridRetriever
from .splitter import extract_text_and_images, ocr_image

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")
PDF_SUFFIX = ".pdf"


def discover_pdfs(documents_dir: str = DOCUMENTS_DIR) -> List[str]:
    """Return absolute paths of PDFs in `documents_dir`, sorted for stable ordering."""
    if not os.path.isdir(documents_dir):
        logger.warning("DOCUMENTS_DIR %s does not exist; nothing to auto-index", documents_dir)
        return []
    return [
        os.path.join(documents_dir, name)
        for name in sorted(os.listdir(documents_dir))
        if name.lower().endswith(PDF_SUFFIX)
        and os.path.isfile(os.path.join(documents_dir, name))
    ]


def ingest_pdf(pdf_path: str, retriever: HybridRetriever, display_name: str = None) -> Dict[str, Any]:
    """Blocking: parse, OCR image-only chunks, and index. Expects a worker thread."""
    display_name = display_name or os.path.basename(pdf_path)

    chunks = extract_text_and_images(pdf_path, source_name=display_name)
    for chunk in chunks:
        if not chunk.get("text") and chunk.get("images"):
            ocr_texts = [t for t in (ocr_image(img) for img in chunk["images"]) if t]
            if ocr_texts:
                chunk["text"] = "\n".join(ocr_texts)

    stats = retriever.add_documents(chunks)
    logger.info("Ingested %s: %s", display_name, stats)
    return {"status": "ok", "filename": display_name, "chunks_extracted": len(chunks), **stats}


def ensure_indexed(retriever: HybridRetriever, documents_dir: str = DOCUMENTS_DIR) -> Dict[str, Any]:
    """Index every PDF in `documents_dir` if the vector store is empty. Startup helper."""
    if retriever.vector_store.count() > 0:
        logger.info("Vector store already holds %d chunks; skipping auto-index",
                    retriever.vector_store.count())
        return {"indexed_documents": 0}

    pdfs = discover_pdfs(documents_dir)
    if not pdfs:
        logger.warning(
            "Vector store is empty and no PDFs found in %s. "
            "Drop a PDF there and restart, or run scripts/reingest.py.",
            documents_dir,
        )
        return {"indexed_documents": 0}

    logger.info("Vector store empty; auto-indexing %d PDF(s) from %s", len(pdfs), documents_dir)
    for path in pdfs:
        ingest_pdf(path, retriever)
    return {"indexed_documents": len(pdfs)}