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
                                "description": (
                                    "Discord id de la personne CONCERNÉE par le souvenir "
                                    "(pas forcément l'auteur du message). null si server/event."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "Souvenir ultra-concis (≤12 mots). Pseudo seul, "
                                    "jamais « le membre X ». Ex: « Alice : anniversaire le 12 mars »."
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

_SYSTEM_PROMPT = """Tu es l'agent mémoire de MARIA.

QUI EST MARIA (critique) :
- MARIA (ou le nom du bot Discord, ex. « {bot_name} ») = le bot Discord lui-même.
- Tu travailles POUR elle : tu n'es pas un membre humain du serveur.
- Dans les messages, « MARIA », « {bot_name} », « le bot », ou
  `[répond à MARIA (le bot): …]` / `@MARIA (le bot)` = conversation AVEC le bot.
- NE crée JAMAIS de souvenir category=user sur MARIA / le bot (pas d'anniversaire,
  goûts, « elle a dit… » comme si c'était une personne).
- Les demandes au bot, réponses du bot, blagues sur le bot → IGNORE (sauf running gag
  collectif clairement ancré sur le serveur → category=server, user_id=null).

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
  gag de groupe qui revient. Le reste → {{"memories": []}}.

Tu as DEUX niveaux :
1) PENDING — observation fragile (surtout perso).
2) ACTIVE — souvenir confirmé / collectif retenu.

PRIORITÉ : si un souvenir existant (surtout PENDING) couvre déjà le sujet → update/merge
(target_id = son id). Ne duplique pas.

CHEVAUCHEMENT DE LOTS :
- Tu peux recevoir un bloc « CONTEXTE PRÉCÉDENT » (fin du lot d'avant) puis « MESSAGES NOUVEAUX ».
- create : uniquement à partir des MESSAGES NOUVEAUX.
- Le contexte précédent sert à enchaîner (running gag, « mon » qui renvoie à une réplique d'avant,
  confirmation d'un pending). update/merge/contradict sur un souvenir existant OK si la suite
  dans les NOUVEAUX le justifie.

Max 6 actions par lot. Si vraiment rien → {{"memories": []}}.

CATÉGORIES — NE LES MÉLANGE PAS :
- user : perso d'UN membre HUMAIN (user_id obligatoire = Discord id de la personne CONCERNÉE).
  Préférences, genre, ville, anniversaire, goûts réellement affirmés (pas du sarcasme,
  pas une anecdote). Reste sélectif. JAMAIS le bot.
- server : collectif de CE serveur (user_id=null). Inside jokes, surnoms, habitudes du salon,
  running gags, blagues récurrentes, « chez nous on… ». SOIS PLUS OUVERT ICI pour les
  gags de groupe identifiables — pas pour transformer une soirée en « règle du serveur ».
  Si plusieurs personnes en parlent ou réagissent → server, pas user.
- event : jalon du serveur (soirée, arrivée/départ, projet lancé). Anecdote du jour ≠ event
  sauf vrai jalon nommé / organisé.

ATTRIBUTION user_id (critique — erreurs fréquentes ici) :
- Format des lignes : `[HH:MM] Pseudo (id): …` parfois avec
  `[répond à AutrePseudo (autre_id): "extrait du message cité"]`.
- « je / mon / ma / mes » = l'auteur de CETTE ligne (son id entre parenthèses), pas la
  personne citée en reply, pas une mention au hasard, pas le bot.
- Si Alice (111) écrit « c'est mon anniversaire » → user_id=111.
- Si Bob (222) répond à Alice « joyeux anniv » → l'anniversaire est celui d'Alice (111),
  pas Bob. Un simple vœu ≠ souvenir « Bob a un anniversaire ».
- Si le message cité dit « c'est mon anniv » et la reply est un vœu / emoji / « merci »,
  le fait porte sur l'auteur du message CITÉ.
- Si quelqu'un répond à MARIA (le bot) en disant « mon anniv c'est… » → le fait porte
  sur l'auteur humain de la ligne, pas sur le bot.
- Si tu n'es pas sûr à 100 % de qui est concerné → n'extrais PAS ce souvenir.

IGNORE : débats du jour, scores/actus, demandes au bot, réponses/blagues sur le bot,
blabla sans ancrage, sarcasme one-shot, histoires one-shot sans potentiel de gag récurrent.

Actions : create (target_id=null) | update | merge | contradict (target_id = id).

CONTENT (style obligatoire) :
- Ultra-concis : ≤ 12 mots, une info max. Pas de subordonnées inutiles.
- Pseudo seul : « Alice », jamais « le membre Alice », « l'utilisateur Bob », « la personne X ».
- user : commence par le pseudo (« Alice : anniversaire le 12 mars », « Bob : habite à Lyon »).
- server/event : fait collectif sec (« Running gag du kebab 4h », « Soirée BBQ du 15/06 »).
- Pas de « a dit que », « semble aimer », « est quelqu'un qui » — va droit au fait.
Ex. OK : « Alice : anniversaire le 25 juillet »
Ex. KO : « Le membre Alice a mentionné que c'était son anniversaire le 25 juillet »"""


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

    system = _SYSTEM_PROMPT.format(bot_name=bot_name or "MARIA")
    # Aligne le plafond du prompt sur le paramètre (évite 6 en code / 4 en texte).
    if max_actions != 6:
        system = system.replace("Max 6 actions par lot.", f"Max {max_actions} actions par lot.")

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
