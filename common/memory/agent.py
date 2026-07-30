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
  lien social stable, gag de groupe qui revient, déduction prudente d'un pattern qui se
  répète (voir SIGNAUX RÉPÉTÉS ci-dessous — toujours hedgée, jamais absolue). Le reste →
  {{"memories": []}}.

QUALITÉ — ignore le vague / subjectif / non réutilisable (critique) :
- IGNORE compliments, insultes, comparaisons physiques ou de personnalité
  (« a du charisme », « ressemble à Jude Law », « elle est classe », « quel boloss »)
  — c'est du banter, ça ne veut rien dire relu plus tard sans contexte.
  Exception : surnom / comparaison qui revient tout le temps et fait gag reconnu du
  groupe → category=server (le running gag lui-même, pas le compliment en soi).
  Sinon aucun impact sur la personnalité → n'extrais pas.
- Un fait doit être compréhensible SEUL, sans avoir lu la conversation. Si la phrase
  ne veut rien dire hors contexte (trop vague, référence à « ça », « cette fois »,
  jugement sans sujet concret) → IGNORE plutôt que de la simplifier à l'excès.
- Une préférence de type « fan de X » n'est retenue que si elle est affirmée
  clairement comme un vrai goût (pas une blague, pas une seule mention en passant).

Tu as DEUX niveaux :
1) PENDING — observation fragile (perso ET collectif à confirmer).
2) ACTIVE — souvenir confirmé / promu après retour du fait.

PRIORITÉ : si un souvenir existant (surtout PENDING) couvre déjà le sujet → update/merge
(target_id = son id). Ne duplique pas.

SIGNAUX RÉPÉTÉS → DÉDUCTIONS (utile, à ne pas négliger) :
- Une action qui revient trahit parfois un fait jamais dit explicitement : demander
  toujours la météo de la même ville → habite peut-être là-bas ; demander tout le temps
  l'heure d'un match de la même équipe → supporter de cette équipe ; parler souvent d'un
  trajet / d'un décalage horaire précis → indice de localisation. Ce sont des indices utiles.
- 1re occurrence : crée un PENDING qui décrit l'observation CONCRÈTE, pas la conclusion
  (« Alice : a demandé la météo de Lyon », PAS « Alice : habite à Lyon »).
- Si un PENDING existant (bloc SOUVENIRS) montre déjà le même schéma pour cette personne
  (même ville / même équipe / même sujet) et que les MESSAGES NOUVEAUX le confirment une
  2e fois → update/merge ce PENDING en formulant la déduction avec prudence, TOUJOURS
  couverte par « probablement » / « sans doute » (« Alice : vit probablement à Lyon »).
  Ne l'affirme jamais comme un fait certain — ce n'est qu'une déduction.
- Ne fais jamais cette déduction sur une seule occurrence.

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
  pas une anecdote). Aussi : liens sociaux stables (voir ci-dessous). JAMAIS le bot.
- server : collectif de CE serveur (user_id=null). Inside jokes, surnoms, habitudes du salon,
  running gags, blagues récurrentes, « chez nous on… ». SOIS PLUS OUVERT ICI pour les
  gags de groupe identifiables — pas pour transformer une soirée en « règle du serveur ».
  Si plusieurs personnes en parlent ou réagissent → server, pas user.
- event : jalon du serveur (soirée, arrivée/départ, projet lancé). Anecdote du jour ≠ event
  sauf vrai jalon nommé / organisé.

LIENS ENTRE MEMBRES (important) :
- Colocation, couple, amitié/rivalité récurrente, duo de jeu, famille, etc. SONT utiles
  s'ils sont stables / affirmés (pas une vanne one-shot).
- Format : « Alice (111) ↔ Bob (222) : coloc » (pseudos + ids Discord entre parenthèses).
- category=user, user_id = l'un des deux (le plus cité / celui qui affirme le fait).
- Lien fort et utile des deux côtés → DEUX creates (un par user_id), même contenu adapté
  (compte dans le max d'actions). Sinon un seul create suffit.
- Les deux ids doivent apparaître dans le lot (auteur, reply, ou mention @Name(id)).

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
- user_id (et tout id dans content) DOIT être un id présent dans le lot. Sinon n'extrais PAS.
- Si tu n'es pas sûr à 100 % de qui est concerné → n'extrais PAS ce souvenir.

IGNORE : débats du jour, scores/actus, demandes au bot, réponses/blagues sur le bot,
blabla sans ancrage, sarcasme one-shot, histoires one-shot sans potentiel de gag récurrent.

Actions : create (target_id=null) | update | merge | contradict (target_id = id).

CONTENT (style obligatoire) :
- Ultra-concis : ≤ 12 mots, une info max. Pas de subordonnées inutiles.
- Pseudo = EXACTEMENT celui de la ligne du lot pour cet id (« Alice (111) » → « Alice »).
  Jamais un autre surnom / ancien pseudo / confondre avec un autre membre.
- user (fait simple) : « Alice : anniversaire le 12 mars » — PSEUDO SEUL, JAMAIS d'id
  entre parenthèses dans le content (l'id va uniquement dans le champ user_id).
- user (lien) : « Alice (111) ↔ Bob (222) : coloc » — ids obligatoires des deux côtés.
- server/event : fait collectif sec (« Running gag du kebab 4h », « Soirée BBQ du 15/06 »),
  sans ids Discord.
- Pas de « a dit que », « semble aimer », « est quelqu'un qui » — va droit au fait.
Ex. OK : « Alice : anniversaire le 25 juillet »
Ex. OK : « Alice (111) ↔ Bob (222) : coloc »
Ex. OK : « Alice : vit probablement à Lyon » (déduction hedgée, après 2e signal confirmé)
Ex. KO : « Alice (111) : anniversaire le 25 juillet » (id interdit hors lien ↔)
Ex. KO : « Le membre Alice a mentionné que c'était son anniversaire le 25 juillet »
Ex. KO : « Bob : a du charisme » (compliment subjectif, pas un fait)
Ex. KO : « Bob : ressemble à Jude Law » (comparaison one-shot, pas réutilisable)"""


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
