"""Agent mémoire — extraction via LLM + JSON schema."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from common.memory.store import Memory, STATUS_PENDING
from common.timezones import PARIS_TZ

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
                                    "Grand fait durable (≤18 mots) : ville, boulot, anniv, "
                                    "lien nommé, goût ancré avec le titre. "
                                    "Perso : « Pseudo : fait » SANS id. "
                                    "Lien : « Alice (111) ↔ Bob (222) : coloc depuis 2023 »."
                                ),
                            },
                            "stable": {
                                "type": "boolean",
                                "description": (
                                    "true seulement pour anniversaire / date de naissance "
                                    "clairement affirmés avec assez de détail. false sinon."
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

_SYSTEM_PROMPT = """Tu extrais des souvenirs DURABLES pour MARIA (petit Discord entre potes).
Le fil du jour est déjà dans le résumé de session : n'en refais PAS des souvenirs.

RÈGLE D'OR — CARTE D'IDENTITÉ, PAS LE JOURNAL :
- Un souvenir = un grand fait sur UNE personne, encore vrai dans des mois.
  OK : « Alice : habite à Saint-Ouen (95) », « Bob : anniversaire le 22 juillet 1999 »,
  « Chloé : main Jett en ranked Valorant », « Alice (111) ↔ Bob (222) : coloc depuis 2023 ».
- KO : ce soir / cette semaine, un score, la météo, un gag, un avis, « a regardé X »,
  « aime les jeux », « habite quelque part », formulations coupées.
- Pas d'identité durable → n'extrais PAS. Mieux vaut [] que trois petits faits.
- N'invente aucun détail absent du lot.

DATES RELATIVES → toujours résoudre en date absolue (AUJOURD'HUI = {current_date}).
« demain », « ce week-end », « lundi prochain »… lus tels quels deviennent faux le lendemain.
Calcule la date réelle. Date relative non résolue → skip.
Un plan daté (voyage, soirée) n'est PAS de l'identité : skip, le résumé de session s'en charge.

DIRECT vs PASSIF :
- `[→ MARIA]` = parle À MARIA. À retenir seulement si c'est identitaire, pas un caprice du tour.
- Sans `[→ MARIA]` = passif. Encore plus strict : net, non sarcastique. Doute → skip.
- Lien durable seulement : « Alice (111) ↔ Bob (222) : coloc ».

COLLECTIF : presque jamais. Pas d'event (soirée, voyage = résumé). server seulement
si habitude de groupe durable et nommable. user_id=null, stable=false, pas d'ids dans content.

BOT = « {bot_name} » / MARIA.
JAMAIS category=self (goûts MARIA) — gérés ailleurs, réservés au créateur.
JAMAIS category=user avec l'id du bot. Blagues sur le bot → ignore.

IGNORE : actu, blabla, image, banter, humeur, ce que le résumé de session couvre déjà.

ATTRIBUTION :
- `[HH:MM] Pseudo (id) [répond à …]: texte`. « je/mon » = auteur de la ligne.
- Fait dans l'extrait cité = la cible (sauf « moi aussi/pareil »).
- Reply au bot + fait perso → auteur humain. user_id / ids ∈ lot. Doute sur qui → skip.

STYLE — une info, prêt à être relu :
- user : « Alice : anniversaire le 22 juillet 1999 »
- Pas de « le membre », « a dit que », « semble », pas de troncature « … ».

stable=true : anniv / date de naissance avec jour+mois (année si dite). Sinon false.
Même sujet en SOUVENIRS → update/merge (target_id) en enrichissant, pas de doublon.
create = MESSAGES NOUVEAUX seulement. Max {max_actions} actions. Souvent 0, rarement plus d'1."""


async def extract_memories(
    llm_client: Any,
    *,
    model: str,
    batch_text: str,
    existing: list[Memory],
    bot_name: str = "MARIA",
    max_actions: int = 3,
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

    now = datetime.now(PARIS_TZ)
    current_date = f"{now.strftime('%A %d/%m/%Y')} {now.strftime('%H:%M')}"
    system = _SYSTEM_PROMPT.format(
        bot_name=bot_name or "MARIA",
        current_date=current_date,
        max_actions=max_actions,
    )

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
                "Priorise les grands faits d'identité (ville, boulot, anniv, lien, goût ancré). "
                "Le fil du jour n'est pas un souvenir. Flou / one-shot → ignore. "
                "Si rien de durable → {\"memories\": []}."
            ),
        },
    ]
    max_tokens = max(800, 220 * max_actions)
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
