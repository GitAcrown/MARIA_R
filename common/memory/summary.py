"""Résumés mémoire pour /moi et /global."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Literal

from common.memory.store import Memory

logger = logging.getLogger("MARIA.Memory.Summary")

# Fingerprint du lot → texte. Invalidé dès qu'un souvenir change (id/texte/confiance).
_CACHE_MAX = 256
_summary_cache: OrderedDict[tuple, str] = OrderedDict()


def _fingerprint(
    memories: list[Memory],
    *,
    scope: str,
    model: str,
    display_name: str,
) -> tuple:
    items = tuple(
        sorted(
            (
                m.id,
                m.content,
                m.status,
                round(float(m.confidence or 0), 4),
            )
            for m in memories
        )
    )
    return (scope, model, display_name, items)


def _cache_get(key: tuple) -> str | None:
    text = _summary_cache.get(key)
    if text is None:
        return None
    _summary_cache.move_to_end(key)
    return text


def _cache_put(key: tuple, text: str) -> None:
    _summary_cache[key] = text
    _summary_cache.move_to_end(key)
    while len(_summary_cache) > _CACHE_MAX:
        _summary_cache.popitem(last=False)

_USER_SUMMARY = """Résume ce que MARIA sait d'un membre, à partir de souvenirs PERSONNELS uniquement (globaux).
2–5 phrases courtes, naturelles, à la 2e personne (« tu… »). Pas de listes, d'emojis, ni d'intro.
Ignore toute info collective. Vide/flou → dis que tu ne sais presque rien. N'invente rien."""

_SERVER_SUMMARY = """Résume la mémoire collective d'un serveur (gags, habitudes, événements).
2–5 phrases courtes, naturelles, à la 3e (« le serveur… », « on… »). Pas de listes, d'emojis, ni d'intro.
Pas de goûts/projets d'un membre précis. Vide/flou → dis que peu est retenu. N'invente rien."""


async def summarize_memories(
    llm_client: Any,
    *,
    model: str,
    memories: list[Memory],
    scope: Literal["user", "server"],
    display_name: str = "",
) -> str:
    empty_user = "Je n'ai encore rien de solide en mémoire sur toi."
    empty_server = "Peu de choses retenues sur le serveur pour l'instant."
    if not memories:
        return empty_user if scope == "user" else empty_server

    cache_key = _fingerprint(
        memories, scope=scope, model=model, display_name=display_name,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    lines = []
    for m in memories:
        conf = f"{m.confidence:.0%}"
        uid = f" user={m.user_id}" if m.user_id else ""
        lines.append(f"- [{m.category}{uid} conf={conf}] {m.content}")

    system = _USER_SUMMARY if scope == "user" else _SERVER_SUMMARY
    header = f"Membre : {display_name}\n" if scope == "user" and display_name else ""
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{header}Souvenirs :\n" + "\n".join(lines),
        },
    ]
    try:
        completion = await llm_client.chat(
            messages, model=model, max_tokens=300,
        )
        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content if choice else None) or ""
        text = text.strip()
        if text:
            _cache_put(cache_key, text)
            return text
        logger.warning(
            "Résumé mémoire vide (scope=%s finish=%s usage=%s)",
            scope,
            getattr(choice, "finish_reason", None),
            getattr(completion, "usage", None),
        )
    except Exception as e:
        logger.warning("Résumé mémoire échoué: %s", e)

    bullets = "\n".join(f"› {m.content}" for m in memories[:8])
    return bullets or (empty_user if scope == "user" else empty_server)


# Rétrocompat
async def summarize_memories_for_user(
    llm_client: Any,
    *,
    model: str,
    display_name: str,
    memories: list[Memory],
) -> str:
    return await summarize_memories(
        llm_client,
        model=model,
        memories=memories,
        scope="user",
        display_name=display_name,
    )
