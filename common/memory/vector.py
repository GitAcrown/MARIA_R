"""Couche vectorielle Chroma (embeddings OpenAI)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("MARIA.Memory.Vector")

CHROMA_DIR = Path("data") / "chroma"
COLLECTION_NAME = "maria_memories"
EMBEDDING_MODEL = "text-embedding-3-small"


class VectorStore:
    """Wrapper Chroma local. Soft-fail si chromadb absent ou indisponible."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._collection = None
        self._ok = False
        try:
            import chromadb
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            ef = OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=EMBEDDING_MODEL,
            )
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._ok = True
        except Exception as e:
            logger.warning("Chroma indisponible — RAG mémoire désactivé: %s", e)

    @property
    def available(self) -> bool:
        return self._ok and self._collection is not None

    def upsert(
        self,
        memory_id: str,
        content: str,
        *,
        category: str,
        guild_id: int,
        user_id: Optional[int],
        confidence: float,
    ) -> None:
        if not self.available:
            return
        meta = {
            "id": memory_id,
            "category": category,
            "guild_id": guild_id,
            "confidence": float(confidence),
            "user_id": int(user_id) if user_id is not None else -1,
        }
        try:
            self._collection.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[meta],
            )
        except Exception as e:
            logger.warning("Chroma upsert échoué (%s): %s", memory_id, e)

    def delete(self, memory_id: str) -> None:
        if not self.available:
            return
        try:
            self._collection.delete(ids=[memory_id])
        except Exception as e:
            logger.warning("Chroma delete échoué (%s): %s", memory_id, e)

    def query(
        self,
        text: str,
        *,
        guild_id: int,
        n: int = 10,
    ) -> list[dict]:
        """Recherche sémantique filtrée par guild. Renvoie [{id, distance, metadata}]."""
        if not self.available or not text.strip():
            return []
        try:
            result = self._collection.query(
                query_texts=[text.strip()[:2000]],
                n_results=n,
                where={"guild_id": guild_id},
            )
        except Exception as e:
            logger.warning("Chroma query échoué: %s", e)
            return []

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        out: list[dict] = []
        for i, mid in enumerate(ids):
            out.append({
                "id": mid,
                "distance": distances[i] if i < len(distances) else 1.0,
                "metadata": metadatas[i] if i < len(metadatas) else {},
            })
        return out
