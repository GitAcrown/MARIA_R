"""RAG — récupération ciblée de souvenirs pour le prompt conversation."""

from __future__ import annotations

import logging
import re
from typing import Optional

from common.memory.store import (
    CATEGORY_SELF,
    STATUS_ACTIVE,
    Memory,
    MemoryStore,
)
from common.memory.vector import VectorStore

logger = logging.getLogger("MARIA.Memory.RAG")

MIN_CONFIDENCE = 0.3

# Faits d'identité à garder en tête de profil (sinon noyés par les goûts).
_IDENTITY_RE = re.compile(
    r"\b(habite|habitant|ville|adresse|vit à|demeure|anniv|naissance|né[e]?|"
    r"âge|ans\b|prénom|s'appelle)\b",
    re.I,
)

# Normalise le contenu pour dédup profil ↔ RAG (casse / ponctuation légère).
_DEDUP_RE = re.compile(r"\s+")


def _norm_content(text: str) -> str:
    return _DEDUP_RE.sub(" ", (text or "").strip().lower())


def build_self_ctx(
    store: MemoryStore,
    *,
    bot_name: str = "MARIA",
    limit: int = 8,
) -> tuple[str, set[str]]:
    """Goûts / faits retenus sur MARIA — injectés pour rester constante."""
    memories = store.list_self(limit=limit)
    if not memories:
        return "", set()

    name = (bot_name or "MARIA").strip() or "MARIA"
    seen: set[str] = set()
    facts: list[str] = []
    for m in memories:
        content = (m.content or "").strip()
        if not content:
            continue
        seen.add(_norm_content(content))
        stripped = content
        for prefix in (f"{name} :", f"{name}:", "MARIA :", "MARIA:"):
            if stripped.lower().startswith(prefix.lower()):
                stripped = stripped[len(prefix) :].strip()
                break
        facts.append(stripped or content)
    if not facts:
        return "", seen

    header = (
        f"TES GOÛTS / TOI ({name}) — trait de fond, PAS un sujet à amener toi-même : "
        "ne les mentionne QUE si on te demande explicitement ton avis sur CE sujet précis, "
        "et reste cohérente avec quand tu le fais. Sinon ignore complètement cette liste — "
        "ne la récite pas, ne la glisse pas dans une réponse sans rapport, jamais deux fois "
        "de suite sur le même point même si le sujet revient :"
    )
    return header + "\n- " + " · ".join(facts), seen


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
        # On tire plus large que le quota pour pouvoir prioriser ville/âge/anniv.
        pool = store.list_for_user(guild_id, uid, limit=max(facts_per_user * 3, 15))
        if not pool:
            continue
        identity = [m for m in pool if _IDENTITY_RE.search(m.content or "")]
        seen_ids = {m.id for m in identity}
        others = [m for m in pool if m.id not in seen_ids]
        memories = (identity + others)[:facts_per_user]
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
        "PROFILS (détails retenus — personnalise / allusion naturelle si ça colle au fil, "
        "ne récite pas, ne force aucun callback, ne confonds pas les ids) :"
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
    fts_hits = store.search_fts(
        query, guild_id=guild_id, user_id=author_id, limit=max(top_k * 2, 8),
    )

    ids: list[str] = []
    seen_ids: set[str] = set()
    for h in hits:
        hid = h.get("id")
        if hid and hid not in seen_ids:
            ids.append(hid)
            seen_ids.add(hid)
    for m in fts_hits:
        if m.id not in seen_ids:
            ids.append(m.id)
            seen_ids.add(m.id)
    if not ids:
        return []

    by_id = {m.id: m for m in store.get_many(ids)}
    distance_by_id = {h["id"]: float(h.get("distance") or 1.0) for h in hits}
    fts_ids = {m.id for m in fts_hits}
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
        # Perso / self : globaux. Collectif / event : guild courant uniquement.
        if m.category == "user":
            if m.user_id != author_id:
                continue
            # Faits déjà dans le profil : exclus via exclude_contents, pas un skip total.
        elif m.category == CATEGORY_SELF:
            # Déjà injecté via TES GOÛTS — pas de doublon RAG.
            continue
        elif m.guild_id != guild_id:
            continue
        candidates.append(m)

    def sort_key(m: Memory) -> tuple:
        # Match FTS : petit bonus de distance (le mot du message est dans le fait).
        dist = distance_by_id.get(m.id, 1.0)
        if m.id in fts_ids:
            dist = max(0.0, dist - 0.15)
        if prefer_collective:
            collective = 0 if m.category in ("server", "event") else 1
            return (collective, dist, -m.confidence)
        author_boost = 0 if m.user_id == author_id else 1
        return (author_boost, dist, -m.confidence)

    candidates.sort(key=sort_key)
    return candidates[:top_k]


def format_memory_ctx(
    memories: list[Memory],
    *,
    name_by_user_id: Optional[dict[int, str]] = None,
    bot_name: str = "MARIA",
) -> str:
    if not memories:
        return ""
    names = name_by_user_id or {}
    name = (bot_name or "MARIA").strip() or "MARIA"
    lines = [
        "MEMOIRE PERTINENTE (complément — allusion OK si pertinent au fil, "
        "sinon ignore ; ne récite pas, n'invente aucun détail) :"
    ]
    for m in memories:
        content = m.content
        if m.category == CATEGORY_SELF:
            # Stocké « MARIA : fait » — jamais répéter ce préfixe dans le contexte,
            # ça lui fait prendre l'habitude de commencer ses réponses par son nom.
            for prefix in (f"{name} :", f"{name}:"):
                if content.lower().startswith(prefix.lower()):
                    content = content[len(prefix):].strip()
                    break
        if m.user_id:
            label = names.get(m.user_id) or "?"
            who = f" {label} ({m.user_id})"
        else:
            who = ""
        lines.append(f"- [{m.category}]{who} {content}")
    return "\n".join(lines)
