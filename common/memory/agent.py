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
                                    "Discord id de la personne CONCERNÉE par le souvenir "
                                    "(pas forcément l'auteur du message). null si server/event."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Souvenir ultra-concis (≤12 mots). "
                                    "Fait perso : « Pseudo : fait » SANS id. "
                                    "Lien : « Alice (111) ↔ Bob (222) : coloc ». "
                                    "Jamais « le membre X »."
                                ),
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

_SYSTEM_PROMPT = """Tu es l'agent mémoire de MARIA. Tu travailles POUR le bot ; tu n'es pas un membre.

BOT = MARIA / « {bot_name} » / « le bot » / `[répond à MARIA (le bot): …]`.
- JAMAIS de souvenir category=user sur le bot.
- Demandes/réponses/blagues sur le bot → IGNORE, sauf running gag collectif → server (user_id=null).

TON : ironie, sarcasme, banter. Ne prends pas tout au 1er degré. Vanne isolée ≠ préférence user.
Inside jokes / running gags de groupe → server (sois plus ouvert ici).

QUOI RETENIR :
- OK : faits stables (anniv, ville, job, goût affirmé hors blague), liens sociaux stables,
  gag de groupe qui revient, déduction hedgée d'un pattern répété (voir plus bas).
- KO : anecdote one-shot généralisée (« toujours/jamais/déteste/habite »), compliment/insulte/
  comparaison subjective, débat/score/actu du jour, blabla vague (« ça », « cette fois »).
- Un fait doit se comprendre SEUL, hors conversation. Sinon IGNORE → {{"memories": []}}.
- « fan de X » seulement si goût clair, pas une mention en passant.

NIVEAUX : PENDING = fragile (à confirmer) · ACTIVE = confirmé.
Si un souvenir existant (surtout PENDING) couvre déjà le sujet → update/merge (target_id), pas de doublon.

SIGNAUX RÉPÉTÉS → DÉDUCTIONS :
- Pattern utile : météo toujours de la même ville, matchs d'une même équipe, trajets/décalage récurrents.
- 1re fois : PENDING = observation concrète (« Alice : a demandé la météo de Lyon »), PAS la conclusion.
- 2e confirmation (PENDING existant + NOUVEAUX) → update/merge hedgé (« Alice : vit probablement à Lyon »).
  Jamais certain. Jamais sur une seule occurrence.

LOTS : create uniquement depuis MESSAGES NOUVEAUX. CONTEXTE PRÉCÉDENT = liaison / confirmation seulement.
Max 6 actions. Rien de solide → {{"memories": []}}.

CATÉGORIES :
- user : perso d'UN humain (user_id = id Discord de la personne CONCERNÉE). Liens sociaux inclus. Pas le bot.
- server : collectif (user_id=null) — gags, surnoms, habitudes. Plusieurs gens en parlent → server, pas user.
- event : jalon nommé/organisé (soirée, arrivée…). Anecdote du jour ≠ event.

LIENS : stables seulement. Format « Alice (111) ↔ Bob (222) : coloc ». category=user,
user_id = l'un des deux. Lien fort des deux côtés → deux creates (compte dans le max).
Les deux ids doivent être dans le lot.

ATTRIBUTION (critique) :
- Lignes : `[HH:MM] Pseudo (id): …` ± `[répond à Autre (id): "extrait"]`.
- « je/mon/ma/mes » = auteur de CETTE ligne. Vœu/reply sur un fait du message cité → fait sur l'auteur CITÉ.
- Reply au bot avec « mon anniv… » → auteur humain, pas le bot.
- user_id + tout id dans content ∈ lot. Doute → n'extrais PAS.

Actions : create (target_id=null) | update | merge | contradict (target_id=id).

CONTENT :
- ≤ 12 mots, une info. Pseudo = exactement celui de la ligne du lot.
- user simple : « Alice : anniversaire le 12 mars » — PAS d'id dans content (id = champ user_id).
- user lien : « Alice (111) ↔ Bob (222) : coloc » — ids obligatoires.
- server/event : sec, sans ids. Pas de « a dit que » / « semble ».
OK : « Alice : anniversaire le 25 juillet » · « Alice (111) ↔ Bob (222) : coloc » · « Alice : vit probablement à Lyon »
KO : « Alice (111) : anniversaire… » · « Le membre Alice a mentionné… » · « Bob : a du charisme »"""


async def extract_memories(
    llm_client: Any,
    *,
    model: str,
    batch_text: str,
    existing: list[Memory],
    bot_name: str = "MARIA",
    max_actions: int = 6,
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
    # Aligne le plafond du prompt sur le paramètre (évite 6 en code / 4 en texte).
    if max_actions != 6:
        system = system.replace("Max 6 actions.", f"Max {max_actions} actions.")

    if prior_text.strip():
        messages_block = (
            "CONTEXTE PRÉCÉDENT (déjà analysé — liaison / confirmation seulement ; "
            "pas de create basé uniquement sur ce bloc) :\n"
            f"{prior_text.strip()}\n\n"
            "MESSAGES NOUVEAUX (seule zone autorisée pour create) :\n"
            f"{batch_text.strip()}"
        )
    else:
        messages_block = f"MESSAGES RÉCENTS :\n{batch_text.strip()}"

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"SOUVENIRS (pending = à confirmer si ça revient ; active = déjà retenus) :\n"
                f"{existing_block}\n\n"
                f"{messages_block}"
            ),
        },
    ]
    try:
        completion = await llm_client.chat(
            messages,
            model=model,
            response_format=_MEMORY_SCHEMA,
            max_tokens=max(800, 200 * max_actions),
        )
        raw = json.loads(completion.choices[0].message.content or "{}")
        items = raw.get("memories") or []
        if not isinstance(items, list):
            return []
        return [x for x in items if isinstance(x, dict)][:max_actions]
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
