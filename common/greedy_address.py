"""Adresse au pseudo (mode greedy) sans appel JEV ni Discord.

Premier ou dernier mot → réponse quasi systématique, sauf une liste de prénoms.
Au milieu → réponse directe seulement s'il y a une question ou un tutoiement.
"""

from __future__ import annotations

import re

from common.emoji_usage import strip_emojis

_CUSTOM_EMOJI_MARKUP_RE = re.compile(r"<a?:\w+:\d+>")
_WORD_RE = re.compile(r"[a-z0-9àâäéèêëïîôùûüçœæ_]+", re.IGNORECASE)
_GREETINGS = frozenset({
    "hey", "hé", "he", "ey", "yo", "salut", "coucou", "bonjour", "bonsoir",
    "hello", "hi", "dis", "oh", "ohé", "eh", "ok", "okay",
})
_LIST_TAIL_RE = re.compile(
    r"(?:\s*(?:,|et|ou|&|avec)\s+[a-zàâäéèêëïîôùûüçœæ][\w'-]*)+\s*",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"[?？]|\b(?:quoi|comment|pourquoi|quel|quelle|quels|quelles)\b|\best-ce\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(?:tu|toi|te|vous)\b|\bt['’]", re.IGNORECASE)

# ignore JEV en greedy seulement au-dessus de ce seuil.
GREEDY_IGNORE_CONFIDENCE = 0.75


def _normalize(content: str) -> str:
    text = _CUSTOM_EMOJI_MARKUP_RE.sub(" ", content or "")
    return strip_emojis(text).lower().strip()


def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _strip_leading_greetings(text: str) -> str:
    parts = (text or "").split()
    i = 0
    while i < len(parts):
        word = _words(parts[i])
        if len(word) != 1 or word[0] not in _GREETINGS:
            break
        i += 1
    return " ".join(parts[i:])


def _name_at(text: str, name: str) -> re.Match[str] | None:
    return re.search(
        rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])",
        text,
        flags=re.IGNORECASE,
    )


def _tail_is_name_list(body: str, name: str) -> bool:
    """« Maria et Bob », « Maria, Léa » — pas « Maria, tu viens »."""
    m = re.match(
        rf"{re.escape(name)}(?![a-z0-9_])",
        body,
        flags=re.IGNORECASE,
    )
    if m is None:
        return False
    tail = body[m.end():]
    if not tail.strip():
        return False
    return _LIST_TAIL_RE.fullmatch(tail) is not None


def name_hit_kind(content: str, bot_name: str) -> str:
    """absent | list | edge | middle.

    `edge` : le pseudo est le premier ou le dernier mot (salut et ponctuation ignorés).
    `list` : le pseudo ouvre une liste de prénoms, pas une adresse.
    """
    name = (bot_name or "").strip().lower()
    if not name:
        return "absent"
    text = _normalize(content)
    if not text or _name_at(text, name) is None:
        return "absent"
    body = _strip_leading_greetings(text)
    name_tokens = _words(name)
    tokens = _words(body)
    if not name_tokens or not tokens:
        return "absent"
    at_start = tokens[:len(name_tokens)] == name_tokens
    at_end = tokens[-len(name_tokens):] == name_tokens
    if at_start and _tail_is_name_list(body, name):
        return "list"
    if at_start or at_end:
        return "edge"
    return "middle"


def is_question(content: str) -> bool:
    return _QUESTION_RE.search(content or "") is not None


def middle_is_address(content: str) -> bool:
    """Question ou tutoiement : le pseudo au milieu s'adresse à elle."""
    text = content or ""
    return is_question(text) or _PRONOUN_RE.search(text) is not None


def finalize_greedy_choice(
    choice: str,
    conf: float,
    *,
    question: bool,
    react_min: float,
) -> str:
    """Décision greedy après JEV. Une question ne devient jamais un simple react."""
    if choice not in ("respond", "react", "ignore"):
        return "respond"
    if choice == "ignore":
        return "ignore" if conf >= GREEDY_IGNORE_CONFIDENCE else "respond"
    if choice == "react":
        if question:
            return "respond"
        return "react" if conf >= react_min else "ignore"
    return "respond"
