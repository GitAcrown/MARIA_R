"""Agent mémoire — extraction via modèle nano + JSON schema."""

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
                                "description": "Discord user id concerné, ou null (server/event)",
                            },
                            "content": {
                                "type": "string",
                                "description": "Souvenir concis (1 phrase), durable ≥ 1 mois",
                            },
                            "confidence_delta": {
                                "type": "number",
                                "description": "Suggestion optionnelle ; ignorée côté serveur",
                            },
                        },
                        "required": [
                            "action", "target_id", "category",
                            "user_id", "content", "confidence_delta",
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

_SYSTEM_PROMPT = """Tu es l'agent mémoire de MARIA, un bot Discord sur un petit serveur de potes.

Tu as DEUX niveaux de mémoire :
1) PENDING (tampon) — observation fragile, pas encore « vraie » mémoire.
2) ACTIVE — souvenir confirmé (vu au moins 2 fois, ou déclaré manuellement).

PRIORITÉ ABSOLUE : regarde d'abord les PENDING. Si un sujet / gag / préférence REVIENT
dans les messages → action update ou merge sur ce pending (target_id = son id).
C'est comme ça qu'un one-shot devient un running gag. NE crée PAS un nouveau souvenir
si un pending proche existe déjà.

SOIS TRÈS SÉLECTIF. Préfère 0 action plutôt que du bruit. Max 3 actions par lot.
Si rien de clair → {"memories": []}.

RÈGLE ANTI ONE-SHOT :
- Une blague, une anecdote, un événement raconté UNE fois ≠ running gag.
- Un « j'aime X » dit une fois ≠ préférence solide.
- Pour un possible gag / habitude / préférence : create (→ ira en pending).
- Seulement si ça REVIENT (pending existant ou répétition claire dans le lot) : update/merge.
- event : réservé aux vrais jalons (arrivée, départ, soirée organisée, lancement de projet) — pas une anecdote.

IGNORE : débats du jour, scores/actus, demandes au bot, avis passagers.

CATÉGORIES :
- user : perso d'UN membre (user_id obligatoire). Global tous serveurs. Pas de gag collectif ici.
- server : collectif de CE serveur (user_id=null). Gags/habitudes de groupe.
- event : jalon de CE serveur.

Actions : create (target_id=null) | update | merge | contradict (target_id = id).
Content : français, 1 phrase neutre, 3e personne."""


async def extract_memories(
    llm_client: Any,
    *,
    model: str,
    batch_text: str,
    existing: list[Memory],
) -> list[dict]:
    """Appelle le modèle nano et renvoie la liste d'actions mémoire."""
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

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"SOUVENIRS (pending = à confirmer si ça revient ; active = déjà retenus) :\n"
                f"{existing_block}\n\n"
                f"MESSAGES RÉCENTS :\n{batch_text}"
            ),
        },
    ]
    try:
        completion = await llm_client.chat(
            messages,
            model=model,
            response_format=_MEMORY_SCHEMA,
            max_tokens=800,
        )
        raw = json.loads(completion.choices[0].message.content or "{}")
        items = raw.get("memories") or []
        if not isinstance(items, list):
            return []
        return [x for x in items if isinstance(x, dict)][:3]
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
