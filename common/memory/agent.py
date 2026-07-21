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

_SYSTEM_PROMPT = """Tu es l'agent mémoire de MARIA, un bot Discord sur un petit serveur entre potes.

TON DU SERVEUR (important) :
- Beaucoup d'ironie, de sarcasme, de second degré et d'inside jokes.
- Ne prends PAS tout au premier degré : une phrase « sérieuse » peut être une blague ;
  un insultant / un « je déteste X » entre potes est souvent du banter, pas un fait perso.
- Les inside jokes et running gags du groupe SONT précieux → catégorie server.
- Ne transforme pas une vanne isolée en préférence user (« il déteste le café »)
  sauf si c'est clairement factuel / répété / assumé hors blague.

RÈGLE ANTI-GÉNÉRALISATION (critique) :
- Une histoire / anecdote racontée ≠ une vérité générale sur la personne ou le serveur.
  Ex. « l'autre jour j'ai mangé un kebab à 4h » ≠ « il mange toujours des kebabs à 4h ».
  Ex. « on a foiré le BBQ samedi » ≠ « le serveur rate toujours les BBQ ».
- Ne reformule JAMAIS en absolu (« toujours », « jamais », « déteste », « adore »,
  « habite », « est ») à partir d'un récit ponctuel.
- Si tu retiens quelque chose d'une anecdote : reste factuel et daté/ponctuel
  (« a raconté avoir… », « une fois… ») — ou mieux : IGNORE, sauf si ça revient
  clairement (running gag / préférence répétée / fait stable affirmé hors récit).
- Les faits stables OK : anniversaire, ville, job, préférence affirmée hors histoire,
  gag de groupe qui revient. Le reste → {"memories": []}.

Tu as DEUX niveaux :
1) PENDING — observation fragile (surtout perso).
2) ACTIVE — souvenir confirmé / collectif retenu.

PRIORITÉ : si un souvenir existant (surtout PENDING) couvre déjà le sujet → update/merge
(target_id = son id). Ne duplique pas.

Max 4 actions par lot. Si vraiment rien → {"memories": []}.

CATÉGORIES — NE LES MÉLANGE PAS :
- user : perso d'UN membre (user_id obligatoire). Préférences, genre, ville, anniversaire, goûts
  réellement affirmés (pas du sarcasme, pas une anecdote). Reste sélectif.
- server : collectif de CE serveur (user_id=null). Inside jokes, surnoms, habitudes du salon,
  running gags, blagues récurrentes, « chez nous on… ». SOIS PLUS OUVERT ICI pour les
  gags de groupe identifiables — pas pour transformer une soirée en « règle du serveur ».
  Si plusieurs personnes en parlent ou réagissent → server, pas user.
- event : jalon du serveur (soirée, arrivée/départ, projet lancé). Anecdote du jour ≠ event
  sauf vrai jalon nommé / organisé.

IGNORE : débats du jour, scores/actus, demandes au bot, blabla sans ancrage,
sarcasme one-shot, histoires one-shot sans potentiel de gag récurrent.

Actions : create (target_id=null) | update | merge | contradict (target_id = id).
Content : français, 1 phrase neutre, 3e personne — sans généraliser au-delà de la preuve."""


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
