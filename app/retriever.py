import hashlib
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import embed_texts, normalize_vectors
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

IMAGE_CHUNK_PLACEHOLDER = "[IMAGE_CHUNK]"
_WHITESPACE = re.compile(r"\s+")


def fingerprint(text: str) -> str:
    """Stable content hash, insensitive to incidental whitespace."""
    return hashlib.sha256(_WHITESPACE.sub(" ", text).strip().encode("utf-8")).hexdigest()


class HybridRetriever:
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        lambda_rank: Optional[float] = None,
        dense_candidate_multiplier: Optional[int] = None,
    ):
        self.persist_dir = persist_dir
        self.lambda_rank = (
            float(os.getenv("RAG_LAMBDA", "0.5")) if lambda_rank is None else float(lambda_rank)
        )
        self.dense_candidate_multiplier = (
            int(os.getenv("RAG_DENSE_MULTIPLIER", "4"))
            if dense_candidate_multiplier is None
            else int(dense_candidate_multiplier)
        )
        self.docs: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.ids: List[str] = []
        self.bm25 = None
        self._lock = threading.RLock()
        self._index_by_id: Dict[str, int] = {}
        self._seen: set = set()
        self.vector_store = VectorStore(persist_dir)
        self._load_existing_collection()

    # ---------------------------------------------------------------- loading

    def _rebuild_bm25(self) -> None:
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self._index_by_id = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self._seen = {fingerprint(t) for t in self.texts}

    def _load_existing_collection(self) -> None:
        stored = self.vector_store.get_all()
        if not stored:
            return
        self.ids = [item["id"] for item in stored]
        self.texts = [item["text"] for item in stored]
        self.docs = [{"text": item["text"], "meta": item["meta"]} for item in stored]
        self._rebuild_bm25()
        logger.info("Loaded %d existing chunks from %s", len(self.texts), self.persist_dir)

    # ------------------------------------------------------------- ingestion

    def add_documents(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Index chunks, skipping any whose text already exists. Returns ingestion stats."""
        with self._lock:
            pending_texts: List[str] = []
            pending_metas: List[Dict[str, Any]] = []
            batch_fingerprints: set = set()
            skipped_duplicates = 0
            skipped_empty = 0

            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                if not text and chunk.get("images"):
                    text = IMAGE_CHUNK_PLACEHOLDER
                if not text:
                    skipped_empty += 1
                    continue
                digest = fingerprint(text)
                if digest in self._seen or digest in batch_fingerprints:
                    skipped_duplicates += 1
                    continue
                batch_fingerprints.add(digest)

                meta = dict(chunk.get("meta") or {})
                meta.setdefault("type", chunk.get("type", "text"))
                pending_texts.append(text)
                pending_metas.append(meta)

            if not pending_texts:
                return {
                    "indexed": 0,
                    "skipped_duplicates": skipped_duplicates,
                    "skipped_empty": skipped_empty,
                    "total_chunks": len(self.texts),
                }

            # Embed before mutating in-memory state so a failure leaves us consistent.
            vectors = embed_texts(pending_texts)
            new_ids = [str(uuid4()) for _ in pending_texts]

            self.ids.extend(new_ids)
            self.texts.extend(pending_texts)
            self.docs.extend({"text": t, "meta": m} for t, m in zip(pending_texts, pending_metas))
            self._rebuild_bm25()
            self.vector_store.add_documents(new_ids, pending_texts, pending_metas, vectors)

            logger.info(
                "Indexed %d chunks (%d duplicates, %d empty skipped); collection now %d",
                len(new_ids), skipped_duplicates, skipped_empty, len(self.texts),
            )
            return {
                "indexed": len(new_ids),
                "skipped_duplicates": skipped_duplicates,
                "skipped_empty": skipped_empty,
                "total_chunks": len(self.texts),
            }

    # -------------------------------------------------------------- clearing

    def clear(self) -> None:
        """Drop every persisted chunk and reset in-memory state."""
        with self._lock:
            self.vector_store.reset()
            self.docs = []
            self.texts = []
            self.ids = []
            self._rebuild_bm25()
        logger.info("Cleared all chunks from %s", self.persist_dir)

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        lambda_rank: Optional[float] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        with self._lock:
            if self.bm25 is None or not self.texts:
                return []
            bm25 = self.bm25
            ids = list(self.ids)
            docs = list(self.docs)
            dense_multiplier = self.dense_candidate_multiplier

        weight = self.lambda_rank if lambda_rank is None else float(lambda_rank)

        bm25_scores = np.asarray(bm25.get_scores(query.split()), dtype=float)
        low, high = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (
            (bm25_scores - low) / (high - low) if high > low else np.zeros_like(bm25_scores)
        )

        query_vec = normalize_vectors([embed_texts([query])[0]])[0]
        n_dense = max(top_k * dense_multiplier, top_k)
        dense = self.vector_store.query(query_vec, n_results=n_dense)
        dense_ids = dense.get("ids", [[]])[0] if dense.get("ids") else []
        dense_dists = dense.get("distances", [[]])[0] if dense.get("distances") else []

        # Normalise distances within the retrieved set so the best hit scores 1.0
        # regardless of the absolute distance scale.
        dense_norm: Dict[str, float] = {}
        if dense_dists:
            dists = np.asarray(dense_dists, dtype=float)
            d_low, d_high = dists.min(), dists.max()
            span = d_high - d_low
            for doc_id, dist in zip(dense_ids, dists):
                dense_norm[doc_id] = 1.0 - ((dist - d_low) / span) if span > 0 else 1.0

        index_by_id = {doc_id: i for i, doc_id in enumerate(ids)}
        candidates = set(dense_ids)
        for idx in np.argsort(bm25_scores)[-top_k:][::-1]:
            if idx < len(ids):
                candidates.add(ids[idx])

        results: List[Tuple[float, Dict[str, Any]]] = []
        for candidate_id in candidates:
            idx = index_by_id.get(candidate_id)
            if idx is None:
                continue
            combined = weight * dense_norm.get(candidate_id, 0.0) + (1 - weight) * float(bm25_norm[idx])
            results.append((combined, docs[idx]))

        results.sort(key=lambda pair: pair[0], reverse=True)
        return results[:top_k]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            by_type: Dict[str, int] = {}
            by_source: Dict[str, int] = {}
            for doc in self.docs:
                meta = doc.get("meta") or {}
                kind = meta.get("type", "text")
                by_type[kind] = by_type.get(kind, 0) + 1
                source = meta.get("source", "unknown")
                by_source[source] = by_source.get(source, 0) + 1
            return {
                "total_chunks": len(self.texts),
                "by_type": by_type,
                "by_source": by_source,
            }