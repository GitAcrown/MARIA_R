"""Recipes de tâches — DSL contraint + rendu pseudo-code (COOLDOWN / MAX / EXPIRE)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from common.tasks import (
    EVENT_COOLDOWN_DEFAULT,
    EVENT_MAX_FIRES_DEFAULT,
    KIND_AT,
    KIND_EVENT,
    KIND_RECURRING,
    KIND_WATCH,
    WATCH_INTERVAL_MIN_MINUTES,
    format_schedule,
)

_GENERIC_PATTERNS = frozenset({
    "ok", "oui", "non", "lol", "mdr", "ptdr", "ouais", "ouai", "yes", "no",
    "hi", "hey", "salut", "a", "b", "??", "...",
})


def pattern_ok(pattern: str) -> str | None:
    """None si OK, sinon message d'erreur FR."""
    p = (pattern or "").strip()
    if len(p) < 3:
        return "Mot-clé trop court (3–24 caractères)."
    if len(p) > 24:
        return "Mot-clé trop long (max 24 caractères)."
    if p.casefold() in _GENERIC_PATTERNS:
        return f"Mot-clé trop générique (« {p} »). Choisis quelque chose de plus précis."
    return None


def parse_trigger(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_recipe(raw: Any) -> dict:
    return parse_trigger(raw)


def trigger_terms(trigger: dict) -> list[str]:
    """Mot-clé principal + variantes (max 4), sans doublon."""
    out: list[str] = []
    seen: set[str] = set()
    raw = [trigger.get("pattern") or ""] + list(trigger.get("aliases") or [])
    for item in raw:
        term = str(item).strip()
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            out.append(term)
    return out[:4]


# ---------------------------------------------------------------------------
# Regex précis : tranchent les cas évidents pour économiser JEV (il ne reste
# que les cas ambigus). Tout ce qui est incertain renvoie None → JEV décide.
# ---------------------------------------------------------------------------

def _norm_words(text: str) -> str:
    """Minuscules, apostrophes/traits d'union → espace, sans ponctuation ni emoji."""
    s = (text or "").casefold()
    s = re.sub(r"[’'`\-_]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


_YES_CORE = (
    r"oui|ouais|ouep|yep|yes|ok|okay|okey|d accord|dac|go|parfait|top|nickel|carrément|"
    r"carrement|exact|valide|validé|confirme|confirmé|banco|bien sûr|bien sur|c est bon|"
    r"ça marche|ca marche|vas y|vasy|active|active la|active le|let s go"
)
_YES_TAIL = r"merci|stp|svp|go|vas y|c est bon|active|parfait|top|ok|d accord|pour moi|à fond|a fond|nickel"
_YES_RE = re.compile(rf"^(?:{_YES_CORE})(?: (?:{_YES_TAIL}))*$")

_NO_CORE = (
    r"non|nan|nope|no|annule|annuler|annule la|annule le|laisse tomber|laisse|oublie|oublie ça|"
    r"oublie ca|stop|pas la peine|pas besoin|finalement non|cancel|surtout pas|inutile"
)
_NO_TAIL = r"merci|stp|svp|finalement|ça|ca|tomber|laisse|c est bon|pas la peine|pas besoin"
_NO_RE = re.compile(rf"^(?:{_NO_CORE})(?: (?:{_NO_TAIL}))*$")

# Un ajustement ou une question n'est ni un oui ni un non : GPT doit répondre.
_ADJUST_RE = re.compile(
    r"\b(?:mais|sauf|plutôt|plutot|moins|plus|souvent|rarement|aussi|ajoute|enlève|enleve|"
    r"change|modifie|seulement|uniquement|quand|combien|comment|que|qui|quoi|ou)\b"
)


def quick_reply_verdict(text: str) -> Optional[str]:
    """Réponse à un brouillon : confirm | cancel | other | None (indécis → JEV)."""
    raw = (text or "").strip()
    if not raw:
        return "other"
    norm = _norm_words(raw)
    if not norm:  # emoji seuls
        if re.search(r"[👍✅👌🙌💯]", raw):
            return "confirm"
        if re.search(r"[❌👎🚫✖]", raw):
            return "cancel"
        return "other"
    if _NO_RE.match(norm):
        return "cancel"
    if _YES_RE.match(norm):
        return "confirm"
    if "?" in raw or _ADJUST_RE.search(norm) or len(norm.split()) > 8:
        return "other"
    return None


_NOISE_RE = re.compile(
    r"^(?:m+d+r+|p+t+d+r+|lol|lmao|xd+|ha(?:ha)+h?|hi(?:hi)+|a+h+|o+h+|ok|oui|ouais|non|nan|"
    r"merci|gg|wp|\+1|bien|top|exact|ah ouais|ah ok|pas mal|trop bien)$"
)
_CMD_RE = re.compile(r"^\s*[!/.$?]\w+")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>|:\w+:")


def looks_like_noise(text: str) -> bool:
    """Message sans sujet possible (rires, interjections, commandes, emoji, URL seule)."""
    raw = (text or "").strip()
    if _CMD_RE.match(raw):
        return True
    if re.fullmatch(r"https?://\S+", raw):
        return True
    cleaned = _CUSTOM_EMOJI_RE.sub(" ", raw)
    norm = _norm_words(cleaned)
    if not norm or _NOISE_RE.match(norm):
        return True
    return len(norm.split()) < 3  # un sujet se dit en 3 mots ou plus


_GENERATE_RE = re.compile(
    r"\b(?:météo|meteo|cherche|recherche|résum\w*|resum\w*|explique\w*|compare\w*|traduis\w*|"
    r"calcul\w*|actu\w*|scores?|horaires?|trajets?|recettes?|conseils?|idées?|idee\w*|propose\w*|"
    r"trouve\w*|analyse\w*|rédige\w*|redige\w*|écris|ecris|blague|raconte\w*|combien|quel(?:le)?s?|"
    r"prix de|donne\w*|liste\w*)\b",
    re.IGNORECASE,
)
_NOTIFY_RE = re.compile(
    r"^\s*(?:préviens|previens|prévenez|ping|pingue|alerte|averti[s]?|notifie|rappelle|dis|tiens)"
    r"[\s-]*(?:moi|nous|me)\b[^.\n]{0,40}[.!]?\s*$",
    re.IGNORECASE,
)
_SHORT_LINE_RE = re.compile(r"^[^.\n]{2,60}[?!]$")


def quick_delivery_mode(instruction: str) -> Optional[str]:
    """verbatim | ping_only | generate | None (indécis → JEV)."""
    text = (instruction or "").strip()
    if not text:
        return "ping_only"
    if _GENERATE_RE.search(text):
        return "generate"
    if _NOTIFY_RE.match(text):
        return "ping_only"
    quoted = re.fullmatch(r"[«\"“](.{2,120})[»\"”]", text)
    if quoted:
        return "verbatim"
    if _SHORT_LINE_RE.match(text) and len(text.split()) <= 8:
        return "verbatim"
    return None


def keyword_is_specific(pattern: str) -> bool:
    """Mot assez distinctif pour se passer du contrôle JEV (chiffre, tiret, espace, long)."""
    p = (pattern or "").strip()
    return bool(
        len(p) >= 8
        or re.search(r"[\d\s_\-]", p)
        or re.search(r"[A-Z].*[A-Z]", p)  # sigle (CS2, GTA)
    )


def event_label(trigger: dict) -> str:
    """Libellé court d'une écoute : mots-clés, sinon le sujet en clair."""
    terms = trigger_terms(trigger)
    if terms:
        return " / ".join(terms)
    return str(trigger.get("topic") or "?").strip()[:60]


def clean_aliases(pattern: str, aliases: Any) -> list[str]:
    """Variantes valides (même règles que le mot-clé), max 3."""
    if isinstance(aliases, str):
        aliases = [a for a in re.split(r"[,;|]", aliases)]
    out: list[str] = []
    main = (pattern or "").strip().casefold()
    for item in aliases or []:
        term = str(item).strip()
        if not term or term.casefold() == main or pattern_ok(term):
            continue
        if term.casefold() not in {o.casefold() for o in out}:
            out.append(term)
        if len(out) >= 3:
            break
    return out


def match_message_pattern(content: str, pattern: str, *, whole_word: bool = True) -> bool:
    text = (content or "")
    pat = (pattern or "").strip()
    if not pat:
        return False
    if not whole_word:
        return pat.casefold() in text.casefold()
    # Mot entier approximatif (bordures non alphanumériques).
    return re.search(
        rf"(?<![a-z0-9_àâäéèêëïîôùûüç]){re.escape(pat)}(?![a-z0-9_àâäéèêëïîôùûüç])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def _fmt_cooldown(seconds: int) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{max(1, seconds // 60)} min"
    return f"{seconds}s"


def _fmt_expire(expires_at: Optional[datetime], *, now: Optional[datetime] = None) -> str:
    if expires_at is None:
        return "—"
    now = now or datetime.now(timezone.utc)
    days = max(0, int((expires_at - now).total_seconds() // 86400))
    if days <= 0:
        hours = max(1, int((expires_at - now).total_seconds() // 3600))
        return f"dans {hours}h"
    return f"dans {days}j"


def render_pseudocode(task) -> str:
    """Pseudo-code lisible ; COOLDOWN / MAX / EXPIRE toujours présents pour event/watch."""
    kind = getattr(task, "kind", None) or KIND_AT
    trigger = parse_trigger(getattr(task, "trigger_json", None) or {})
    recipe = parse_recipe(getattr(task, "recipe_json", None) or {})
    lines: list[str] = []

    if kind == KIND_EVENT:
        terms = trigger_terms(trigger)
        topic = str(trigger.get("topic") or "").strip()
        scope = getattr(task, "scope", "channel") or "channel"
        where = "tout le serveur" if scope == "guild" else "salon(s) choisi(s)"
        author = trigger.get("author") or "any"
        if terms:
            pat = '" ou "'.join(terms)
            lines.append(f'QUAND message contient "{pat}" ({where})')
        else:
            lines.append(f'QUAND on parle de "{topic or "?"}" ({where})')
        if author == "not_self":
            lines.append("  SI auteur ≠ moi")
            indent = "    "
        else:
            indent = "  "
        say = (recipe.get("say") or getattr(task, "instruction", "") or "").strip()
        if recipe.get("ping", True):
            lines.append(f"{indent}PING moi")
        if say:
            short = say.replace("\n", " ")[:80]
            lines.append(f'{indent}DIRE "{short}"')
    elif kind == KIND_WATCH:
        url = (trigger.get("url") or "")[:60]
        op = trigger.get("op") or "lt"
        thr = trigger.get("threshold")
        op_fr = {"lt": "<", "lte": "≤", "gt": ">", "change": "change"}.get(op, op)
        lines.append(f"QUAND prix sur {url or 'URL'} {op_fr} {thr}")
        lines.append("  PING moi")
        say = (recipe.get("say") or getattr(task, "instruction", "") or "").strip()
        if say:
            lines.append(f'  DIRE "{say[:80]}"')
        interval = int(trigger.get("interval_minutes") or WATCH_INTERVAL_MIN_MINUTES)
        if interval >= 60 and interval % 60 == 0:
            lines.append(f"  VERIF toutes les {interval // 60}h")
        else:
            lines.append(f"  VERIF toutes les {interval} min")
    elif kind == KIND_RECURRING:
        lines.append(f"QUAND {format_schedule(task)}")
        instr = (getattr(task, "instruction", "") or "").strip()
        if instr:
            lines.append(f'  FAIRE "{instr[:100]}"')
    else:
        lines.append("QUAND l'heure arrive")
        instr = (getattr(task, "instruction", "") or "").strip()
        if instr:
            lines.append(f'  FAIRE "{instr[:100]}"')

    if kind in (KIND_EVENT, KIND_WATCH):
        cd = int(getattr(task, "cooldown_seconds", 0) or EVENT_COOLDOWN_DEFAULT)
        mx = int(getattr(task, "max_fires", 0) or EVENT_MAX_FIRES_DEFAULT)
        exp = getattr(task, "expires_at", None)
        lines.append(f"COOLDOWN {_fmt_cooldown(cd)}")
        lines.append(f"MAX {mx} fois")
        lines.append(f"EXPIRE {_fmt_expire(exp)}")
    return "\n".join(lines)


def build_event_trigger(
    *,
    pattern: str,
    channel_ids: list[int],
    author: str = "not_self",
    scope: str = "channel",
    aliases: Optional[list[str]] = None,
    intent: str = "",
    topic: str = "",
) -> dict:
    topic = (topic or "").strip()[:160]
    return {
        "type": "message",
        "pattern": (pattern or "").strip(),
        "aliases": clean_aliases(pattern, aliases),
        "topic": topic,
        # Détection par le sens (JEV) : sujet décrit en clair, mots-clés facultatifs.
        "semantic": bool(topic),
        "intent": (intent or "").strip()[:200],
        "match": "word",
        "author": author if author in ("any", "not_self") else "not_self",
        "channel_ids": [int(c) for c in channel_ids],
        "scope": scope,
    }


def build_watch_trigger(
    *,
    url: str,
    threshold: float,
    op: str = "lt",
    interval_minutes: int = WATCH_INTERVAL_MIN_MINUTES,
    currency: str = "EUR",
    var_key: str = "",
    anchor: str = "",
) -> dict:
    return {
        "type": "url_price",
        "anchor": (anchor or "").strip()[:40],
        "url": (url or "").strip(),
        "op": op if op in ("lt", "lte", "gt", "change") else "lt",
        "threshold": float(threshold),
        "currency": currency or "EUR",
        "interval_minutes": max(WATCH_INTERVAL_MIN_MINUTES, int(interval_minutes)),
        "var_key": var_key,
    }


def build_recipe(*, say: str = "", ping: bool = True, mode: str = "generate") -> dict:
    """mode : generate (GPT rédige au déclenchement) | verbatim | ping_only (aucun appel GPT)."""
    if mode not in ("generate", "verbatim", "ping_only"):
        mode = "generate"
    return {"say": (say or "").strip(), "ping": bool(ping), "mode": mode}


def _fmt_duration_days(expires_at: Optional[datetime]) -> str:
    if expires_at is None:
        return "quelques jours"
    secs = (expires_at - datetime.now(timezone.utc)).total_seconds()
    if secs < 36 * 3600:
        return "jusqu'à demain" if secs > 12 * 3600 else "ce soir"
    return f"{max(1, round(secs / 86400))} jours"


def kind_label(kind: str) -> str:
    return {"event": "Écoute", "watch": "Veille", "recurring": "Rappel", "at": "Rappel"}.get(
        kind or "", "Tâche",
    )


def focus_label(task, *, max_len: int = 60) -> str:
    """Sujet court d'une tâche (liste, select, titres) — topic > mots-clés > seuil > titre."""
    kind = getattr(task, "kind", None) or KIND_AT
    trigger = parse_trigger(getattr(task, "trigger_json", None) or {})
    if kind == KIND_EVENT:
        topic = str(trigger.get("topic") or "").strip()
        terms = trigger_terms(trigger)
        raw = topic or (" / ".join(terms) if terms else "") or (getattr(task, "title", "") or "")
        raw = re.sub(r"^Écoute\s*[·•\-:]?\s*", "", raw, flags=re.IGNORECASE).strip(" «»\"'")
    elif kind == KIND_WATCH:
        thr = trigger.get("threshold")
        raw = f"≤ {thr:g} €" if thr is not None else (getattr(task, "title", "") or "prix")
        raw = re.sub(r"^Veille\s*[·•\-:]?\s*", "", str(raw), flags=re.IGNORECASE).strip()
    else:
        raw = (getattr(task, "title", None) or getattr(task, "instruction", None) or "Sans consigne")
    text = " ".join(str(raw).split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def scope_label(task) -> str:
    scope = getattr(task, "scope", "channel") or "channel"
    if scope == "guild":
        return "tout le serveur"
    n = len(getattr(task, "channel_ids", None) or [])
    if n <= 1:
        return "ce salon"
    return f"{n} salons"


def delivery_hint(task) -> str:
    """Ce que le membre recevra — vide si mode générique (GPT)."""
    recipe = parse_recipe(getattr(task, "recipe_json", None) or {})
    mode = str(recipe.get("mode") or "generate")
    say = (recipe.get("say") or getattr(task, "instruction", "") or "").strip()
    say = " ".join(say.split())
    if mode == "verbatim" and say:
        short = say if len(say) <= 70 else say[:69] + "…"
        return f'Enverra « {short} »'
    if mode == "ping_only":
        return "Ping simple (sans message rédigé)"
    return ""


def compact_limits(task, *, price: Optional[float] = None) -> str:
    """Une ligne de limites, sans jargon (confirm + détail)."""
    kind = getattr(task, "kind", None) or KIND_AT
    trigger = parse_trigger(getattr(task, "trigger_json", None) or {})
    bits: list[str] = []
    if kind == KIND_EVENT:
        bits.append(scope_label(task))
    elif kind == KIND_WATCH:
        if price is not None:
            bits.append(f"actuellement {price:.2f} €")
        interval = int(trigger.get("interval_minutes") or WATCH_INTERVAL_MIN_MINUTES)
        bits.append(f"vérif ~{max(1, interval // 60)} h")
    mx = int(getattr(task, "max_fires", 0) or EVENT_MAX_FIRES_DEFAULT)
    fires = int(getattr(task, "fires_count", 0) or 0)
    if mx == 1:
        bits.append("1 seule alerte")
    elif fires:
        bits.append(f"{fires}/{mx} alertes")
    else:
        bits.append(f"{mx} alertes max")
    cd = int(getattr(task, "cooldown_seconds", 0) or EVENT_COOLDOWN_DEFAULT)
    if mx > 1 and kind == KIND_EVENT:
        bits.append(f"≥ {_fmt_cooldown(cd)} entre deux")
    exp = getattr(task, "expires_at", None)
    if exp is not None:
        bits.append(f"expire <t:{int(exp.timestamp())}:R>")
    if getattr(task, "deliver_dm", False):
        bits.append("MP")
    return " · ".join(bits)


def natural_summary(task, *, price: Optional[float] = None) -> str:
    """Phrase en clair (ce qui déclenche). Les chiffres vont dans `compact_limits`."""
    kind = getattr(task, "kind", None) or KIND_AT
    trigger = parse_trigger(getattr(task, "trigger_json", None) or {})
    if kind == KIND_EVENT:
        topic = str(trigger.get("topic") or "").strip()
        terms = trigger_terms(trigger)
        if topic and terms:
            about = f"« {topic} »"
            # Mot-clé secondaire seulement s'il n'est pas déjà dans le sujet.
            extras = [t for t in terms if t.casefold() not in topic.casefold()]
            if extras:
                about += f" (aussi « {' / '.join(extras)} »)"
        elif topic:
            about = f"« {topic} »"
        elif terms:
            about = " / ".join(f"« {t} »" for t in terms)
        else:
            about = "« ? »"
        where = "sur tout le serveur" if (getattr(task, "scope", "") or "") == "guild" else "dans ce salon"
        return f"Je te ping quand quelqu'un parle de {about} {where}."
    if kind == KIND_WATCH:
        thr = trigger.get("threshold")
        op = trigger.get("op") or "lt"
        op_fr = {"lt": "passe sous", "lte": "passe à", "gt": "dépasse", "change": "bouge de"}.get(op, "passe sous")
        now_bit = f" (actuellement {price:.2f} €)" if price is not None else ""
        return f"Je te ping si le prix {op_fr} {thr} €{now_bit}."
    return ""


def human_status_line(task) -> str:
    """Ligne courte pour liste / widget (sans jargon cd/Mot)."""
    kind = getattr(task, "kind", KIND_AT) or KIND_AT
    trigger = parse_trigger(getattr(task, "trigger_json", None) or {})
    if kind == KIND_EVENT:
        fires = int(getattr(task, "fires_count", 0) or 0)
        mx = int(getattr(task, "max_fires", 0) or EVENT_MAX_FIRES_DEFAULT)
        focus = focus_label(task, max_len=36)
        return f"« {focus} » · {fires}/{mx}"
    if kind == KIND_WATCH:
        thr = trigger.get("threshold")
        fires = int(getattr(task, "fires_count", 0) or 0)
        mx = int(getattr(task, "max_fires", 0) or EVENT_MAX_FIRES_DEFAULT)
        return f"≤ {thr:g} € · {fires}/{mx}"
    from common.tasks import SCHEDULE_ONCE
    if getattr(task, "schedule_kind", SCHEDULE_ONCE) != SCHEDULE_ONCE:
        return format_schedule(task)
    ts = int(task.execute_at.timestamp())
    return f"<t:{ts}:R>"
