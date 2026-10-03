Macroeconomic Forecast RAG

Overview
- PDF text + table extraction, boundary-aware chunking, hybrid retrieval (BM25 + vector), Grok generation.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set `GROK_API_KEY`, `GROK_API_URL`, and optional `GROK_MODEL` in `.env`.

3. Run locally:

```bash
uvicorn app.main:app --reload
```

Then open http://localhost:8000. On first run the PDFs in `DOCUMENTS_DIR` are
indexed automatically (about 20 seconds for the 59-page forecast), after which
the chat is ready.

Documents

The app is **read-only**. There is no upload endpoint — the corpus is whatever
PDFs sit in `DOCUMENTS_DIR` (default: the project root). On startup, if the
vector store is empty, every PDF in that directory is indexed automatically.

To add or remove documents, drop PDFs into `DOCUMENTS_DIR` and re-index:

```bash
python scripts/reingest.py                     # wipe, then index every PDF in DOCUMENTS_DIR
python scripts/reingest.py --keep              # index without wiping (dedup applies)
python scripts/reingest.py path/to/other.pdf   # wipe, then index a specific file
```

Re-indexing an already-present document is a no-op: chunks are deduplicated by
content hash, so repeats report `skipped_duplicates` instead of growing the
collection. An empty store plus an empty directory logs a warning and starts
with no documents.

Docker
- Build: `docker build -t rag-grok-app .`
- Run: `docker run -p 8000:8000 --env-file .env rag-grok-app`

API
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Chunk counts by type |
| `POST` | `/chat` | Ask a question; returns `response`, `sources`, `used_llm`, `error` |

`/upload` was removed. Adding documents means editing `DOCUMENTS_DIR` and
re-indexing, not pushing files through the browser.

Architecture
- `app/splitter.py`: extracts PDF text and tables, creates boundary-aware chunks.
- `app/ingest.py`: PDF discovery, ingestion, and startup auto-indexing.
- `app/embeddings.py`: loads the embedding model from `EMBEDDING_MODEL` and converts text to dense vectors.
- `app/vector_store.py`: persists embeddings and metadata to a local Chroma vector database.
- `app/retriever.py`: content-hash dedup, sparse BM25 plus dense retrieval, combined scores.
- `app/generator.py`: sends the best contexts to Grok. Raises `GrokError` on failure rather than returning a fake answer.
- `app/main.py`: FastAPI endpoints for querying and serving the UI.

Configuration
| Variable | Default | Purpose |
|----------|---------|---------|
| `GROK_API_KEY` | — | Required for generation |
| `GROK_API_URL` | Groq chat completions | Generation endpoint |
| `GROK_MODEL` | `llama-3.3-70b-versatile` | Generation model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `RAG_LAMBDA` | `0.5` | Weight of dense vs BM25 scores |
| `RAG_DENSE_MULTIPLIER` | `4` | Dense candidate pool = `top_k × this` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking window |
| `MAX_CONTEXT_CHARS` | `12000` | Prompt context budget |
| `DOCUMENTS_DIR` | `.` | Directory scanned for PDFs at startup |
| `VECTOR_STORE_DIR` | `./data/chroma` | Vector DB location |

Notes about PDFs with images and tables
- Tables detected by `find_tables()` are indexed as dedicated chunks, one row per
  line, so row labels stay attached to their numbers.
- **Borderless tables merge columns.** Statistical tables without ruling lines
  come back with adjacent columns concatenated. Row membership is preserved;
  individual column boundaries are not.
- **Charts are not extracted.** Chart geometry is vector drawing data, invisible
  to `get_images()`. Axis labels and legends land in the text stream unstructured.
- **OCR is effectively inert.** It only runs for chunks with no text at all, and
  it needs the `tesseract` binary (`brew install tesseract`), which is not
  installed. `OCR_AVAILABLE` reports `True` regardless because the Python
  wrapper imports fine. Chart data needs a vision model or a chart-to-data tool.

Storage
- The vector database persists in `VECTOR_STORE_DIR` (default `./data/chroma`).
- BM25 sparse retrieval uses in-memory text chunks, rebuilt on startup and after each ingest.

See `ARCHITECTURE.md` for the full pipeline and known limitations.