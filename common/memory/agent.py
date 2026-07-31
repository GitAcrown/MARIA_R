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

# Deux barres : collectif souple · perso prudent (évite faux profils).
_SYSTEM_PROMPT = """Tu extrais des souvenirs pour MARIA (petit Discord entre potes).

DEUX BARRES — ne les mélange pas :
1) COLLECTIF (server / event) — SOUPLE : en cas de doute léger → RETIENS.
   Gags, surnoms, habitudes de salon, « chez nous… », soirées, voyages, projets de groupe,
   blagues récurrentes, restos/bars, calls réguliers. 1re occurrence identifiable OK.
   Plusieurs gens en parlent ou réagissent → server, pas user.
2) PERSO (user) — PRUDENT : seulement si clairement affirmé / répété / non sarcastique.
   En cas de doute sur le FAIT ou sur QUI → n'extrais PAS.
   OK : anniv, âge, prénom, ville, coloc, études/job, couple/bestie/duo, goûts nets,
   allergies, dispo récurrente. Liens : « Alice (111) ↔ Bob (222) : coloc » (ids ∈ lot).
   Pattern répété (2e fois) → déduction hedgée ; 1re → observation concrète seulement.

BOT = « {bot_name} » / MARIA. JAMAIS de souvenir user sur le bot.
Blagues sur le bot → ignore, sauf gag collectif → server (user_id=null).

IGNORE (les deux) : actu/score du jour, blabla vague, image non décrite,
transfert non repris. Banter one-shot → ignore en user ; en server OK si ça devient un gag.

ATTRIBUTION (surtout user) :
- `[HH:MM] Pseudo (id) [répond à …]: texte`. « je/mon » = auteur de la ligne.
- Fait dans l'extrait cité = la cible (sauf « moi aussi/pareil »).
- Reply au bot + fait perso → auteur humain. user_id / ids content ∈ lot.

stable=true : uniquement anniv / date de naissance affirmés (user). Sinon false.
server/event : user_id=null, stable=false, content sans ids Discord.
Même sujet en SOUVENIRS → update/merge (target_id). create = MESSAGES NOUVEAUX seulement.
Max 8 actions. Priorise le collectif s'il y a de la matière ; perso seulement si solide.
CONTENT user : « Alice : anniversaire le 25 juillet ». server : « Running gag du kebab 4h »."""


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
                "Collectif : sois ouvert. Perso : seulement les faits solides. "
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
