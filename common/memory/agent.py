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

Tu as DEUX niveaux :
1) PENDING — observation fragile (surtout perso).
2) ACTIVE — souvenir confirmé / collectif retenu.

PRIORITÉ : si un souvenir existant (surtout PENDING) couvre déjà le sujet → update/merge
(target_id = son id). Ne duplique pas.

Max 4 actions par lot. Si vraiment rien → {"memories": []}.

CATÉGORIES — NE LES MÉLANGE PAS :
- user : perso d'UN membre (user_id obligatoire). Préférences, ville, anniversaire, goûts.
  Reste sélectif : un avis passager ≠ mémoire. Un « j'aime X » une fois → create pending OK
  si c'est clair et durable ; sinon ignore.
- server : collectif de CE serveur (user_id=null). Gags de groupe, surnoms collectifs,
  habitudes du salon, running gags, blagues récurrentes, « chez nous on… ».
  SOIS PLUS OUVERT ICI : dès qu'un gag / habitude de groupe est clairement identifiable
  dans le lot (même première fois), create en server. Si plusieurs personnes en parlent
  ou réagissent → server, pas user.
- event : jalon du serveur (soirée, arrivée/départ, projet lancé). Anecdote du jour ≠ event.

IGNORE : débats du jour, scores/actus, demandes au bot, blabla sans ancrage.

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
        return [x for x in items if isinstance(x, dict)][:4]
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
