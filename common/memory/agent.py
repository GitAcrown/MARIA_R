"""Agent mémoire — extraction via LLM + JSON schema."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from common.memory.store import Memory, STATUS_PENDING

logger = logging.getLogger("MARIA.Memory.Agent")

_MEMORY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "memory_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["create", "update", "merge", "contradict"],
                            },
                            "target_id": {
                                "type": ["string", "null"],
                                "description": "ID d'un souvenir existant pour update/merge/contradict",
                            },
                            "category": {
                                "type": "string",
                                "enum": ["user", "server", "event"],
                            },
                            "user_id": {
                                "type": ["string", "null"],
                                "description": (
                                    "Discord id de la personne CONCERNÉE "
                                    "(pas forcément l'auteur). null si server/event."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "≤14 mots, auto-explicatif. "
                                    "Perso : « Pseudo : fait » SANS id. "
                                    "Lien : « Alice (111) ↔ Bob (222) : coloc »."
                                ),
                            },
                            "stable": {
                                "type": "boolean",
                                "description": (
                                    "true seulement pour anniversaire / date de naissance "
                                    "clairement affirmés. false sinon."
                                ),
                            },
                        },
                        "required": [
                            "action", "target_id", "category",
                            "user_id", "content", "stable",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["memories"],
            "additionalProperties": False,
        },
    },
}

# Prompt volontairement court et biaisé extraction : trop de garde-fous → modèle → []
_SYSTEM_PROMPT = """Tu extrais des souvenirs utiles pour MARIA (petit Discord entre potes).
Préfère RETENIR trop que trop peu. En cas de doute léger sur l'utilité → create PENDING.
En cas de doute sur QUI est concerné → n'extrais PAS ce souvenir-là seulement.

BOT = « {bot_name} » / MARIA / « le bot ». JAMAIS de souvenir user sur le bot.
Réponses/blagues au bot → ignore, sauf gag collectif récurrent → server.

RETIENS (1re mention claire OK, en PENDING) :
- user : anniv, âge, prénom/surnom, ville, coloc/seul, études/job, couple/bestie/duo jeu,
  goûts affirmés (jeux, séries, bouffe, sport…), allergies, dispo récurrente, véhicule.
- server : inside jokes, running gags, habitudes de groupe (soirée jeu, resto fétiche…).
- event : soirée/voyage/jalon nommé. Pas le débat du jour.
- Liens : « Alice (111) ↔ Bob (222) : coloc » (ids du lot). Deux creates si utile des 2 côtés.
- Pattern répété (2e fois) : déduction hedgée (« vit probablement à Lyon »).
  1re fois : observation (« a demandé la météo de Lyon »).

IGNORE seulement : vanne one-shot pure, compliment/insulte vague, actu/score du jour,
blabla sans ancrage, contenu d'image non décrit, transfert non repris par l'auteur.

ATTRIBUTION :
- Ligne `[HH:MM] Pseudo (id) [répond à …]: texte`. « je/mon » = auteur de la ligne.
- Fait dans l'extrait cité = la cible de la reply, pas l'auteur (sauf « moi aussi/pareil »).
- Reply au bot + fait perso → auteur humain.
- user_id et ids du content ∈ lot.

stable=true : uniquement anniv / date de naissance affirmés. Sinon false.
Même sujet déjà en SOUVENIRS → update/merge (target_id), pas de doublon.
create seulement depuis MESSAGES NOUVEAUX. Max 8 actions.
CONTENT : « Alice : anniversaire le 25 juillet » (pas d'id hors liens ↔)."""


async def extract_memories(
    llm_client: Any,
    *,
    model: str,
    batch_text: str,
    existing: list[Memory],
    bot_name: str = "MARIA",
    max_actions: int = 8,
    prior_text: str = "",
) -> list[dict]:
    """Appelle le LLM d'extraction et renvoie la liste d'actions mémoire."""
    existing_block = "Aucun souvenir existant lié."
    if existing:
        lines = []
        for m in existing:
            uid = f" user={m.user_id}" if m.user_id else ""
            level = "PENDING" if m.status == STATUS_PENDING else "ACTIVE"
            lines.append(
                f"- id={m.id} [{level}/{m.category}]{uid} "
                f"hits={m.hits} conf={m.confidence:.2f}: {m.content}"
            )
        existing_block = "\n".join(lines)

    system = _SYSTEM_PROMPT.format(bot_name=bot_name or "MARIA")
    if max_actions != 8:
        system = system.replace("Max 8 actions.", f"Max {max_actions} actions.")

    if prior_text.strip():
        messages_block = (
            "CONTEXTE PRÉCÉDENT (liaison / confirmation seulement — pas de create seul) :\n"
            f"{prior_text.strip()}\n\n"
            "MESSAGES NOUVEAUX (zone create) :\n"
            f"{batch_text.strip()}"
        )
    else:
        messages_block = f"MESSAGES RÉCENTS :\n{batch_text.strip()}"

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"SOUVENIRS existants :\n{existing_block}\n\n"
                f"{messages_block}\n\n"
                "Extrais tous les faits utiles du lot (pending OK). "
                "Si vraiment rien → {\"memories\": []}."
            ),
        },
    ]
    # Budget large : même avec reasoning_effort=none côté client, laisser de la marge JSON.
    max_tokens = max(2500, 350 * max_actions)
    try:
        completion = await llm_client.chat(
            messages,
            model=model,
            response_format=_MEMORY_SCHEMA,
            max_tokens=max_tokens,
        )
        choice = completion.choices[0] if completion.choices else None
        raw_text = (choice.message.content if choice else None) or ""
        if not raw_text.strip():
            usage = getattr(completion, "usage", None)
            logger.warning(
                "Extraction mémoire : content vide (finish=%s usage=%s) — "
                "souvent budget tokens / raisonnement",
                getattr(choice, "finish_reason", None),
                usage,
            )
            return []
        raw = json.loads(raw_text)
        items = raw.get("memories") or []
        if not isinstance(items, list):
            return []
        out = [x for x in items if isinstance(x, dict)][:max_actions]
        logger.info("Extraction mémoire : %d action(s)", len(out))
        return out
    except Exception as e:
        logger.warning("Extraction mémoire échouée: %s", e)
        return []


def parse_user_id(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "" or raw == "null":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
