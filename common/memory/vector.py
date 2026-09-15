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
        guild_id: Optional[int] = None,
        user_id: Optional[int] = None,
        people_ids: Optional[set[int]] = None,
        n: int = 10,
    ) -> list[dict]:
        """Recherche sémantique.

        - Si `people_ids` / `user_id` : souvenirs du guild **ou** perso globaux de ces users.
        - Sinon si `guild_id` : filtre guild uniquement.
        """
        if not self.available or not text.strip():
            return []
        people: list[int] = []
        for uid in people_ids or ():
            if uid is not None:
                people.append(int(uid))
        if user_id is not None and int(user_id) not in people:
            people.append(int(user_id))
        where = None
        user_clauses = [
            {"$and": [{"category": "user"}, {"user_id": uid}]} for uid in people
        ]
        if people and guild_id is not None:
            where = {
                "$or": [
                    {"guild_id": int(guild_id)},
                    {"category": "self"},
                    *user_clauses,
                ]
            }
        elif user_id is not None and guild_id is not None:
            where = {
                "$or": [
                    {"guild_id": int(guild_id)},
                    {"$and": [{"category": "user"}, {"user_id": int(user_id)}]},
                    {"category": "self"},
                ]
            }
        elif guild_id is not None:
            where = {
                "$or": [
                    {"guild_id": int(guild_id)},
                    {"category": "self"},
                ]
            }
        elif people:
            where = {
                "$or": user_clauses + [{"category": "self"}],
            }
        elif user_id is not None:
            where = {
                "$or": [
                    {"$and": [{"category": "user"}, {"user_id": int(user_id)}]},
                    {"category": "self"},
                ]
            }

        try:
            kwargs: dict = {
                "query_texts": [text.strip()[:2000]],
                "n_results": n,
            }
            if where is not None:
                kwargs["where"] = where
            result = self._collection.query(**kwargs)
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

    def all_ids(self) -> list[str]:
        if not self.available:
            return []
        try:
            data = self._collection.get(include=[])
        except Exception as e:
            logger.warning("Chroma list ids échoué: %s", e)
            return []
        return list(data.get("ids") or [])

    def reconcile(self, valid_ids: set[str]) -> int:
        """Retire de Chroma les ids absents de SQLite active."""
        existing = self.all_ids()
        stale = [i for i in existing if i not in valid_ids]
        deleted = 0
        chunk = 100
        for i in range(0, len(stale), chunk):
            batch = stale[i:i + chunk]
            try:
                self._collection.delete(ids=batch)
                deleted += len(batch)
            except Exception as e:
                logger.warning("Chroma reconcile delete échoué: %s", e)
        if deleted:
            logger.info("Chroma reconcile : %d id(s) orphelin(s) retirés", deleted)
        return deleted
