import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import pymupdf
from pathlib import Path

# find_tables() emits a one-off layout-package recommendation on stderr; not actionable here.
pymupdf.no_recommend_layout()

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+")
_WHITESPACE = re.compile(r"[ \t]+")


def _join_within_unit(parts: List[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _split_oversized(piece: str, size: int) -> List[str]:
    """Break a single paragraph that exceeds `size` on sentence, then word, boundaries."""
    sentences = [s for s in _SENTENCE_SPLIT.split(piece) if s and s.strip()]
    if not sentences:
        sentences = [piece]

    out: List[str] = []
    buf = ""
    for sentence in sentences:
        sentence = sentence.strip()
        # A single "sentence" longer than a whole chunk: fall back to word wrapping.
        while len(sentence) > size:
            if buf:
                out.append(buf)
                buf = ""
            out.append(sentence[:size])
            sentence = sentence[size:]
        sentence = sentence.lstrip()
        if not sentence:
            continue
        if not buf:
            buf = sentence
        elif len(buf) + 1 + len(sentence) <= size:
            buf = _join_within_unit([buf, sentence])
        else:
            out.append(buf)
            buf = sentence
    if buf:
        out.append(buf)
    return out


def _overlap_tail(text: str, overlap: int) -> str:
    """Take the trailing `overlap` characters, snapped forward to a clean boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    boundary = _SENTENCE_SPLIT.search(tail)
    if boundary and boundary.end() < len(tail):
        return tail[boundary.end():].strip()
    space = tail.find(" ")
    return (tail[space + 1:] if space != -1 else tail).strip()


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into chunks that respect paragraph, sentence and word boundaries."""
    if not text or not text.strip():
        return []
    if overlap >= size:
        overlap = size // 5

    units: List[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
        else:
            units.extend(_split_oversized(paragraph, size))

    chunks: List[str] = []
    buf = ""
    for unit in units:
        candidate = f"{buf}\n\n{unit}" if buf else unit
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = _overlap_tail(buf, overlap)
        else:
            tail = ""
        buf = f"{tail}\n\n{unit}".strip() if tail else unit
        if len(buf) > size:
            pieces = _split_oversized(buf, size)
            chunks.extend(pieces[:-1])
            buf = pieces[-1] if pieces else ""
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _clean_cell(value: Optional[str]) -> str:
    return _WHITESPACE.sub(" ", (value or "").replace("\n", " ")).strip()


def _serialize_table(rows: List[List[Optional[str]]], page_no: int, ordinal: int) -> Optional[str]:
    """Render an extracted table as pipe-delimited text, one source row per line."""
    cleaned = []
    for row in rows:
        cells = [_clean_cell(c) for c in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < MIN_TABLE_ROWS:
        return None
    if max(len(r) for r in cleaned) < MIN_TABLE_COLS:
        return None
    body = "\n".join(" | ".join(cells) for cells in cleaned)
    return f"[Table {ordinal} | page {page_no}]\n{body}"


def _chunk_serialized_table(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Split a serialized table on row boundaries, repeating the header in each chunk."""
    lines = text.split("\n")
    header, rows = lines[0], lines[1:]
    chunks: List[str] = []
    buf: List[str] = []
    for row in rows:
        candidate = "\n".join([header] + buf + [row])
        if buf and len(candidate) > size:
            chunks.append("\n".join([header] + buf))
            buf = [row]
        else:
            buf.append(row)
    if buf:
        chunks.append("\n".join([header] + buf))
    return chunks


def _extract_table_chunks(page: "pymupdf.Page", page_no: int, start_ordinal: int) -> List[Dict[str, Any]]:
    """Detect tables on a page and emit them as dedicated chunks."""
    try:
        found = page.find_tables()
    except Exception:
        return []
    if not found or not found.tables:
        return []

    chunks: List[str] = []
    for offset, table in enumerate(found.tables):
        try:
            rows = table.extract()
        except Exception:
            continue
        serialized = _serialize_table(rows, page_no, start_ordinal + offset)
        if serialized:
            chunks.extend(_chunk_serialized_table(serialized))

    return [
        {"text": body, "page": page_no, "type": "table", "ordinal": start_ordinal + i}
        for i, body in enumerate(chunks)
    ]


def _extract_page_images(doc: "pymupdf.Document", page: "pymupdf.Page", page_index: int, tmpdir: str) -> List[str]:
    images: List[str] = []
    for img_i, img in enumerate(page.get_images(full=True)):
        try:
            pix = pymupdf.Pixmap(doc, img[0])
            if pix.n >= 5:  # CMYK -> RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            img_path = os.path.join(tmpdir, f"p{page_index + 1}_img{img_i}.png")
            pix.save(img_path)
            pix = None
            images.append(img_path)
        except Exception:
            continue
    return images


def extract_text_and_images(pdf_path: str, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return chunks with keys: `text`, `page`, `type`, `images`, `meta`.

    `source_name` overrides the name recorded in chunk metadata, so uploads can be
    labelled with their original filename rather than a temporary on-disk name.
    """
    source_name = source_name or Path(pdf_path).name
    doc = pymupdf.open(pdf_path)
    out_chunks: List[Dict[str, Any]] = []
    tmpdir = tempfile.mkdtemp(prefix="pdf_assets_")
    table_ordinal = 0

    try:
        for p in range(len(doc)):
            page = doc[p]
            page_no = p + 1
            meta = {"source": source_name, "page": page_no}

            table_chunks = _extract_table_chunks(page, page_no, table_ordinal)
            table_ordinal += len(table_chunks)
            for chunk in table_chunks:
                out_chunks.append({
                    "text": chunk["text"],
                    "page": page_no,
                    "type": "table",
                    "images": [],
                    "meta": {**meta, "type": "table", "table_ordinal": chunk["ordinal"]},
                })

            text = page.get_text("text")
            images = _extract_page_images(doc, page, p, tmpdir)

            for body in _chunk_text(text):
                out_chunks.append({
                    "text": body,
                    "page": page_no,
                    "type": "text",
                    "images": images,
                    "meta": {**meta, "type": "text"},
                })

            if not text.strip() and images:
                out_chunks.append({
                    "text": "",
                    "page": page_no,
                    "type": "image",
                    "images": images,
                    "meta": {**meta, "type": "image"},
                })
    finally:
        doc.close()

    return out_chunks


def ocr_image(image_path: str) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ""