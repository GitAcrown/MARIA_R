"""RAG — récupération ciblée de souvenirs pour le prompt conversation."""

from __future__ import annotations

import logging
import re
from typing import Optional

from common.memory.store import STATUS_ACTIVE, Memory, MemoryStore
from common.memory.vector import VectorStore

logger = logging.getLogger("MARIA.Memory.RAG")

MIN_CONFIDENCE = 0.3

# Normalise le contenu pour dédup profil ↔ RAG (casse / ponctuation légère).
_DEDUP_RE = re.compile(r"\s+")


def _norm_content(text: str) -> str:
    return _DEDUP_RE.sub(" ", (text or "").strip().lower())


def build_profile_ctx(
    store: MemoryStore,
    *,
    guild_id: int,
    people: list[tuple[int, str]],
    facts_per_user: int = 5,
) -> tuple[str, set[str]]:
    """Mini-profils stables (actifs) pour personnaliser sans dépendre du wording.

    people = [(user_id, display_name), ...] — auteur en premier.
    Retourne (bloc prompt, contenus normalisés déjà injectés pour dédup RAG).
    """
    if not people:
        return "", set()

    lines: list[str] = []
    seen_contents: set[str] = set()
    for uid, name in people:
        memories = store.list_for_user(guild_id, uid, limit=facts_per_user)
        if not memories:
            continue
        label = (name or "?").strip() or "?"
        facts: list[str] = []
        for m in memories:
            content = (m.content or "").strip()
            if not content:
                continue
            seen_contents.add(_norm_content(content))
            # Évite de répéter « Alice : … » / « Alice (id) : … » si on a déjà le label.
            stripped = content
            for prefix in (f"{label} :", f"{label}:"):
                if stripped.lower().startswith(prefix.lower()):
                    stripped = stripped[len(prefix) :].strip()
                    break
            else:
                prefix_match = re.match(
                    rf"^{re.escape(label)}\s*\(\d{{17,20}}\)\s*:\s*(.+)$",
                    stripped,
                    re.IGNORECASE,
                )
                if prefix_match:
                    stripped = prefix_match.group(1).strip()
            if "↔" not in stripped:
                stripped = re.sub(r"\s*\(\d{17,20}\)", "", stripped).strip()
            facts.append(stripped or content)
        if facts:
            lines.append(f"- {label} ({uid}): " + " · ".join(facts))

    if not lines:
        return "", seen_contents

    header = (
        "PROFILS (détails retenus sur ces membres — personnalise naturellement, "
        "ne récite pas la liste, ne confonds pas les ids ; croise s'il y a un lien) :"
    )
    return header + "\n" + "\n".join(lines), seen_contents


def retrieve_memories(
    store: MemoryStore,
    vectors: VectorStore,
    *,
    query: str,
    guild_id: int,
    author_id: int,
    top_k: int = 3,
    prefer_collective: bool = False,
    exclude_contents: Optional[set[str]] = None,
) -> list[Memory]:
    """Top-k : perso auteur + souvenirs serveur du guild (complément aux profils)."""
    if not query.strip():
        return []

    hits = vectors.query(
        query,
        guild_id=guild_id,
        user_id=author_id,
        n=max(top_k * 3, 12),
    )
    if not hits:
        return []

    ids = [h["id"] for h in hits]
    by_id = {m.id: m for m in store.get_many(ids)}
    distance_by_id = {h["id"]: float(h.get("distance") or 1.0) for h in hits}
    exclude = exclude_contents or set()

    candidates: list[Memory] = []
    for mid in ids:
        m = by_id.get(mid)
        if m is None or m.status != STATUS_ACTIVE:
            continue
        if m.confidence < MIN_CONFIDENCE:
            continue
        if _norm_content(m.content) in exclude:
            continue
        # Perso : global. Collectif / event : guild courant uniquement.
        if m.category == "user":
            if m.user_id != author_id:
                continue
            # Déjà couvert par les profils → évite le doublon perso dans le RAG.
            if prefer_collective:
                continue
        elif m.guild_id != guild_id:
            continue
        candidates.append(m)

    def sort_key(m: Memory) -> tuple:
        if prefer_collective:
            # server/event d'abord, puis distance, puis confiance.
            collective = 0 if m.category in ("server", "event") else 1
            return (collective, distance_by_id.get(m.id, 1.0), -m.confidence)
        author_boost = 0 if m.user_id == author_id else 1
        return (author_boost, distance_by_id.get(m.id, 1.0), -m.confidence)

    candidates.sort(key=sort_key)
    return candidates[:top_k]


def format_memory_ctx(
    memories: list[Memory],
    *,
    name_by_user_id: Optional[dict[int, str]] = None,
) -> str:
    if not memories:
        return ""
    names = name_by_user_id or {}
    lines = [
        "MEMOIRE PERTINENTE (complément précis — personnalise avec, "
        "ne récite pas, n'invente aucun détail manquant) :"
    ]
    for m in memories:
        if m.user_id:
            label = names.get(m.user_id) or "?"
            who = f" {label} ({m.user_id})"
        else:
            who = ""
        lines.append(f"- [{m.category}]{who} {m.content}")
    return "\n".join(lines)
