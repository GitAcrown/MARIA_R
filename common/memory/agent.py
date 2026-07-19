"""Agent mémoire — extraction via modèle nano + JSON schema."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from common.memory.store import Memory

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
                                "description": "Suggestion optionnelle ; ignorée côté serveur si hors règles",
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
Tu analyses un lot de messages récents et tu décides quoi RETENIR pour plusieurs semaines/mois.

NE RETIENS QUE si la réponse est clairement OUI à :
- Cette info sera-t-elle encore utile dans un mois ?
- Est-ce une préférence durable, un projet, une relation, un rôle, un événement important, une habitude ou un running gag récurrent ?

IGNORE : blagues ponctuelles, débats éphémères, scores/actus du jour, demandes au bot, petites phrases sans substance.
Si rien à retenir → {"memories": []}. Préfère 0 souvenir plutôt que du bruit. Max 5 souvenirs par lot.

Actions :
- create : nouveau souvenir (target_id=null)
- update : même info re-observée ou précisée (target_id = id existant)
- merge : fusionner dans un souvenir existant (target_id)
- contradict : l'info existante est contredite (target_id)

Catégories : user (goûts, projets, relations, habitudes d'une personne), server (règles implicites, gags, habitudes communauté), event (événements, arrivées/départs, projets collectifs).
Rédige content en français, 1 phrase neutre et concise."""


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
            lines.append(
                f"- id={m.id} [{m.category}]{uid} conf={m.confidence:.2f}: {m.content}"
            )
        existing_block = "\n".join(lines)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"SOUVENIRS EXISTANTS (à update/merge/contradict si pertinent) :\n"
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
        return [x for x in items if isinstance(x, dict)]
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
