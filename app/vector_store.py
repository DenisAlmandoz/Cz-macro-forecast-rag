import logging
import os
from typing import Any, Dict, List

import chromadb

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pdf_chunks")


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def get_all(self) -> List[Dict[str, Any]]:
        data = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": _id, "text": txt or "", "meta": meta or {}}
            for _id, txt, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection, discarding all persisted chunks."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection %s in %s", COLLECTION_NAME, self.persist_dir)