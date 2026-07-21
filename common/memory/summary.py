"""Résumés mémoire pour /moi et /global."""

from __future__ import annotations

import logging
from typing import Any, Literal

from common.memory.store import Memory

logger = logging.getLogger("MARIA.Memory.Summary")

_USER_SUMMARY = """Tu résumes ce que MARIA sait d'un membre Discord, à partir de souvenirs PERSONNELS uniquement (globaux, tous serveurs confondus).
Rédige 2 à 5 phrases courtes, naturelles, à la 2e personne (« tu… »).
Pas de listes, pas d'emojis, pas d'intro du type « Voici ce que je sais ».
Ignore toute info collective/serveur si elle apparaît par erreur.
Si les souvenirs sont vides ou trop flous, dis clairement que tu ne sais encore presque rien sur cette personne.
N'invente rien au-delà des souvenirs fournis."""

_SERVER_SUMMARY = """Tu résumes la mémoire collective d'un serveur Discord (gags, habitudes, règles implicites, événements).
Rédige 2 à 5 phrases courtes, naturelles, à la 3e personne (« le serveur… », « on… »).
Pas de listes, pas d'emojis, pas d'intro.
Ne parle PAS des goûts ou projets d'un membre en particulier — uniquement le collectif.
Si vide ou flou, dis clairement que peu de choses sont encore retenues sur le serveur.
N'invente rien au-delà des souvenirs fournis."""


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
        text = (completion.choices[0].message.content or "").strip()
        if text:
            return text
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
