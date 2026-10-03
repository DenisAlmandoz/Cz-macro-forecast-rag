FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY app /app/app

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-fetch the embedding model so the container starts without network access.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV HF_HUB_OFFLINE=1 \
    DOCUMENTS_DIR=/app/documents \
    VECTOR_STORE_DIR=/app/data/chroma
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]