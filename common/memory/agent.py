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
Tu analyses un lot de messages et tu décides quoi RETENIR pour plusieurs semaines/mois.

SOIS TRÈS SÉLECTIF. Préfère 0 souvenir plutôt que du bruit. Max 3 souvenirs par lot.
Si rien de clairement durable → {"memories": []}.

NE RETIENS QUE si c'est clairement utile dans ≥ 1 mois :
préférence durable, projet, relation, rôle, habitude, running gag récurrent, événement important.

IGNORE absolument : blagues ponctuelles, débats du jour, scores/actus, demandes au bot,
avis passagers, petits faits sans lendemain, « j'aime bien X » dit une seule fois sans suite.

CATÉGORIES — ne les confonds JAMAIS :
- user : info PERSONNELLE sur UN membre précis (goût, projet, habitude, relation).
  Obligatoire : user_id = l'id Discord de CE membre (fourni dans les messages).
  N'attribue jamais à user une info collective (« on fait souvent… », gag du serveur).
- server : info COLLECTIVE du serveur (règle implicite, gag partagé, habitude de groupe, rituels).
  user_id = null. Ne range JAMAIS ici le goût ou le projet d'une seule personne.
- event : événement ponctuel mais mémorable (arrivée/départ, lancement de projet, soirée organisée).
  user_id = le membre concerné si pertinent, sinon null.

Actions : create (target_id=null) | update | merge | contradict (target_id = id existant).
Rédige content en français, 1 phrase neutre et concise, à la 3e personne."""


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
