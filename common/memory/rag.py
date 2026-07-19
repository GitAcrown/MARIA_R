"""RAG — récupération ciblée de souvenirs pour le prompt conversation."""

from __future__ import annotations

import logging
from typing import Optional

from common.memory.store import STATUS_ACTIVE, Memory, MemoryStore
from common.memory.vector import VectorStore

logger = logging.getLogger("MARIA.Memory.RAG")

MIN_CONFIDENCE = 0.3


def retrieve_memories(
    store: MemoryStore,
    vectors: VectorStore,
    *,
    query: str,
    guild_id: int,
    author_id: int,
    top_k: int = 5,
) -> list[Memory]:
    """Top-k souvenirs actifs via similarité Chroma + boost auteur."""
    if not query.strip():
        return []

    hits = vectors.query(query, guild_id=guild_id, n=max(top_k * 2, 10))
    if not hits:
        return []

    ids = [h["id"] for h in hits]
    by_id = {m.id: m for m in store.get_many(ids)}
    distance_by_id = {h["id"]: float(h.get("distance") or 1.0) for h in hits}

    candidates: list[Memory] = []
    for mid in ids:
        m = by_id.get(mid)
        if m is None or m.status != STATUS_ACTIVE:
            continue
        if m.confidence < MIN_CONFIDENCE:
            continue
        candidates.append(m)

    def sort_key(m: Memory) -> tuple:
        author_boost = 0 if m.user_id == author_id else 1
        return (author_boost, distance_by_id.get(m.id, 1.0), -m.confidence)

    candidates.sort(key=sort_key)
    return candidates[:top_k]


def format_memory_ctx(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = ["MEMOIRE PERTINENTE (utilise pour personnaliser, ne récite pas, n'invente rien au-delà) :"]
    for m in memories:
        uid = f" uid={m.user_id}" if m.user_id else ""
        lines.append(f"- [{m.category}]{uid} {m.content}")
    return "\n".join(lines)
