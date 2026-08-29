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
                                    "Fait précis et naturel (≤22 mots), avec le détail utile "
                                    "(date, lieu, titre, nom…). "
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

_SYSTEM_PROMPT = """Tu extrais des souvenirs pour MARIA (petit Discord entre potes).

RÈGLE D'OR — PRÉCISION OU RIEN :
- Un souvenir doit contenir le détail concret utile (quoi / qui / où / quand assez pour
  le réutiliser plus tard sans le fil). Pas de détail → n'extrais PAS.
- KO : « aime les jeux », « habite quelque part », « anniversaire le… », « gag du serveur »,
  « joue souvent », formulations coupées ou vagues.
- OK : « anniversaire le 22 juillet 1999 », « habite à Saint-Ouen (95) »,
  « main Jett en ranked Valorant », « Running gag : kebab commandé à 4h du mat ».
- N'invente aucun détail absent du lot. Mieux vaut [] qu'un fait flou.

DATES RELATIVES → toujours résoudre en date absolue avant d'écrire le fait (AUJOURD'HUI = {current_date}).
« demain », « après-demain », « ce week-end », « lundi prochain », « dans 3 jours »… lus tels quels
deviendraient faux dès le lendemain. Calcule la date réelle et écris-la (ex. « demain » un jeudi 12/03
→ « voyage prévu le 13/03 »). Fait avec date relative non résolue → n'extrais PAS plutôt que de la garder.

DIRECT vs PASSIF (critique pour le perso) :
- Lignes marquées `[→ MARIA]` = l'humain parle À MARIA (mention / reply au bot).
  Fait perso précis ici → PRIORITAIRE, à retenir volontiers (toujours avec détail).
- Lignes sans `[→ MARIA]` = lecture passive du salon. PERSO beaucoup plus prudent :
  seulement faits très clairs, non ambigus, non sarcastiques. En cas de doute → skip.
- Collectif : OK depuis le passif si le gag/habitude est identifiable et précis.

TROIS BARRES :
1) COLLECTIF (server/event) — plus ouvert, mais toujours précis.
   Gags nommables, habitudes concrètes, soirées/voyages clairement identifiés.
   Plusieurs gens → server. user_id=null, stable=false, pas d'ids Discord dans content.
2) PERSO (user) — précis + prudent surtout en passif.
   DIRECT `[→ MARIA]` : affirmations à MARIA = bonne source.
   PASSIF : seulement si affirmé net / répété. Liens : « Alice (111) ↔ Bob (222) : coloc ».
   Pattern (2e fois) → déduction hedgée précise ; 1re → observation détaillée.
BOT = « {bot_name} » / MARIA.
JAMAIS category=self (goûts MARIA) — gérés ailleurs, réservés au créateur.
JAMAIS category=user avec l'id du bot.
Blagues sur le bot → ignore, sauf gag collectif précis → server.

IGNORE : actu/score du jour, blabla, image non décrite, transfert non repris,
banter one-shot sans ancrage, tout fait sans détail réutilisable.

ATTRIBUTION :
- `[HH:MM] Pseudo (id) [répond à …]: texte`. « je/mon » = auteur de la ligne.
- Fait dans l'extrait cité = la cible (sauf « moi aussi/pareil »).
- Reply au bot + fait perso → auteur humain. user_id / ids ∈ lot. Doute sur qui → skip.

STYLE — naturel, une info, prêt à être relu :
- user : « Alice : anniversaire le 22 juillet 1999 »
- server : « Running gag : kebab à 4h du mat »
- Pas de « le membre », « a dit que », « semble », pas de troncature « … ».

stable=true : anniv / date de naissance avec jour+mois (année si dite). Sinon false.
Même sujet en SOUVENIRS → update/merge (target_id) en enrichissant le détail, pas de doublon.
create = MESSAGES NOUVEAUX seulement. Max 8 actions."""


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

    now = datetime.now(PARIS_TZ)
    current_date = f"{now.strftime('%A %d/%m/%Y')} {now.strftime('%H:%M')}"
    system = _SYSTEM_PROMPT.format(bot_name=bot_name or "MARIA", current_date=current_date)
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
                "Priorise les faits PRÉCIS des lignes [→ MARIA]. "
                "Passif perso : très sélectif. Flou → ignore. "
                "Si rien de solide → {\"memories\": []}."
            ),
        },
    ]
    max_tokens = max(2000, 300 * max_actions)
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
