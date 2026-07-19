"""Résumé mémoire pour la commande /memory."""

from __future__ import annotations

import logging
from typing import Any

from common.memory.store import Memory

logger = logging.getLogger("MARIA.Memory.Summary")

_SUMMARY_SYSTEM = """Tu résumes ce que MARIA sait d'un membre Discord, à partir de souvenirs structurés.
Rédige 2 à 5 phrases courtes, naturelles, à la 2e personne (« tu… »).
Pas de listes, pas d'emojis, pas d'intro du type « Voici ce que je sais ».
Si les souvenirs sont vides ou trop flous, dis clairement que tu ne sais encore presque rien sur cette personne.
N'invente rien au-delà des souvenirs fournis."""


async def summarize_memories_for_user(
    llm_client: Any,
    *,
    model: str,
    display_name: str,
    memories: list[Memory],
) -> str:
    if not memories:
        return "Je n'ai encore rien de solide en mémoire sur toi."

    lines = []
    for m in memories:
        conf = f"{m.confidence:.0%}"
        scope = f"user={m.user_id}" if m.user_id else "serveur"
        lines.append(f"- [{m.category}/{scope} conf={conf}] {m.content}")

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Membre : {display_name}\n"
                f"Souvenirs :\n" + "\n".join(lines)
            ),
        },
    ]
    try:
        completion = await llm_client.chat(
            messages, model=model, max_tokens=300,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text or "Je n'ai encore rien de solide en mémoire sur toi."
    except Exception as e:
        logger.warning("Résumé mémoire échoué: %s", e)
        # Fallback : liste brute courte
        bullets = "\n".join(f"› {m.content}" for m in memories[:8])
        return bullets or "Je n'ai encore rien de solide en mémoire sur toi."
