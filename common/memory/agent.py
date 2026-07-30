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
                                    "Souvenir concis et auto-explicatif (≤14 mots). "
                                    "Fait perso : « Pseudo : fait » SANS id. "
                                    "Lien : « Alice (111) ↔ Bob (222) : coloc ». "
                                    "Jamais « le membre X »."
                                ),
                            },
                            "stable": {
                                "type": "boolean",
                                "description": (
                                    "true uniquement pour un fait immuable clairement affirmé "
                                    "(anniversaire, date de naissance…) → actif immédiatement. "
                                    "false pour le reste (goûts, liens, déductions, gags…)."
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

_SYSTEM_PROMPT = """Tu es l'agent mémoire de MARIA sur un petit serveur Discord entre potes proches.
Tu travailles POUR le bot ; tu n'es pas un membre. But : retenir ce qui aide MARIA à
personnaliser et à faire des liens — sans stocker le bruit du tchat.

BOT = MARIA / « {bot_name} » / « le bot » / `[répond à MARIA (le bot): …]`.
- JAMAIS de souvenir category=user sur le bot.
- Demandes/réponses/blagues sur le bot → IGNORE, sauf running gag collectif → server.

TON : ironie, sarcasme, banter. Vanne isolée ≠ fait. Ne généralise PAS une anecdote
(« kebab à 4h une fois » ≠ « mange toujours des kebabs »).

CIBLE — sois ouvert sur ce qui sert vraiment entre potes (1re mention claire → PENDING OK) :
USER (perso d'UN humain, user_id = personne CONCERNÉE) :
- Identité : prénom/surnom préféré, pronoms, âge si dit, anniv / date de naissance.
- Vie : ville/région, coloc/seul/chez parents, études/job, horaires atypiques, véhicule.
- Liens stables : couple, coloc, bestie, duo de jeu, rivalité récurrente, famille citée.
- Goûts affirmés (hors blague) : jeux (main/rank), films/séries, musique, bouffe, boisson,
  sport/équipe, hobbies. Allergies / régimes utiles au groupe.
- Habitudes utiles : fuseau/décalage, « jamais dispo le mardi », streamer régulier.
SERVER (collectif, user_id=null) — sois plus ouvert :
- Inside jokes, surnoms collectifs, running gags, règles implicites du salon.
- Habitudes de groupe : soirée jeu du vendredi, resto/bar fétiche, call vocal récurrent.
EVENT : jalon nommé/organisé (soirée, voyage, arrivée/départ, projet lancé). Pas le débat du jour.

SIGNAUX RÉPÉTÉS → DÉDUCTIONS (hedgées) :
- Ex. météo toujours Lyon, matchs toujours du même club, trajets/décalage récurrents.
- 1re fois : observation (« Alice : a demandé la météo de Lyon »), pas la conclusion.
- 2e fois (PENDING existant + NOUVEAUX) → update hedgé (« Alice : vit probablement à Lyon »).

IGNORE (bruit) :
- Compliments/insultes/comparaisons one-shot, scores/actus du jour, blabla vague (« ça »).
- Sarcasme non ancré, histoires one-shot sans gag potentiel, demandes outil sans fait perso.
- Un fait doit se comprendre SEUL hors conversation — sinon n'extrais pas.

NIVEAUX / stable :
- stable=true (create user) : IMMUABLE affirmé (anniv, date de naissance) → ACTIVE immédiat.
  PAS ville/job/goûts/liens/déductions.
- stable=false : PENDING (confirmé plus tard) sauf update d'un ACTIVE.
- Même sujet déjà en base → update/merge (target_id), pas de doublon.

LOTS : create seulement depuis MESSAGES NOUVEAUX. CONTEXTE PRÉCÉDENT = liaison/confirmation.
Max 8 actions. Rien d'utile → {{"memories": []}}. Plusieurs faits solides dans le lot → prends-les
(ne sois pas radin) ; priorise perso + liens + gags serveur.

LIENS : « Alice (111) ↔ Bob (222) : coloc ». category=user, user_id = l'un des deux.
Lien fort des deux côtés → deux creates. Ids ∈ lot.

LECTURE DU TCHAT (critique — faux souvenirs fréquents ici) :
- Ligne : `[HH:MM] Pseudo (id) [répond à Cible…]: texte` (+ tags `[transfère]` / `[image]`…).
- Reply membre→membre `[répond à Bob (222): "…"]` : le fait DANS l'extrait cité = Bob.
  L'auteur de la reply ne « possède » ce fait que s'il l'affirme pour lui (« moi aussi »,
  « pareil », « mon anniv aussi c'est… »). Un « mdr » / « +1 » / emoji ≠ appropriation.
- Reply → bot `[répond à MARIA (le bot): "…"]` : conversation AVEC le bot.
  « je/mon » dans la reply = auteur humain. JAMAIS de souvenir user sur le bot.
  Une info dite PAR le bot dans l'extrait n'est pas un fait membre (sauf si un humain
  la confirme clairement ensuite).
- `[transfère: "…"]` = message forwardé d'ailleurs : PAS une affirmation de l'auteur,
  sauf s'il le reprend explicitement (« oui c'est mon anniv », « c'est bien moi »).
- `[image:…]` / `[sticker:…]` / `[embed:…]` / `[fichier:…]` = média joint. N'invente
  JAMAIS le contenu d'une image ; retiens un fait seulement s'il est écrit dans le texte.
- Bloc citation `>` dans le texte = propos rapportés, pas forcément ceux de l'auteur.
- Quelqu'un qui parle D'un autre (« Alice habite à Lyon ») → user_id = la personne
  concernée SEULEMENT si c'est affirmé clairement et non sarcastique ; sinon IGNORE.
- Doute sur qui est concerné → n'extrais PAS.

Actions : create (target_id=null) | update | merge | contradict (target_id=id).

CONTENT :
- ≤ 14 mots, une info, auto-explicatif. Pseudo = celui de la ligne du lot.
- user : « Alice : anniversaire le 12 mars » (pas d'id) · lien : ids obligatoires.
- server/event : sec, sans ids. Pas de « a dit que » / « semble ».
OK : « Alice : main Jett sur Valorant » · « Bob (222) ↔ Alice (111) : coloc »
OK : « Running gag du kebab 4h » · « Alice : vit probablement à Lyon »
KO : « Alice (111) : anniv… » · « Bob : a du charisme » · « Le membre… »"""


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
                f"SOUVENIRS (pending = à confirmer ; active = retenus ; "
                f"stable=true seulement pour anniv/date de naissance affirmée) :\n"
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
