"""Outil LLM — résumé d'un salon / thread Discord.

Stratégie : on jette le bruit (ok, mdr, ping seul) avant le modèle.
Transcript court → 1 passe. Sinon lots larges mis en cache, puis une fusion :
seul le lot modifié et la fusion rappellent le modèle.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import discord

from common.emojis import RESUME
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.timezones import PARIS_TZ
from common.widget_catalog import render_free_widget

logger = logging.getLogger("MARIA.Chat.Summary")

# Sans fenêtre horaire : un extrait récent, pas tout l'historique du salon.
_DEFAULT_LIMIT = 60
_MAX_LIMIT = 500
# Filet anti-emballement. L'heure demandée borne la lecture : ce plafond
# ne doit pas couper un pic. 2 000 messages / 24 h arrivent, souvent
# concentrés sur la soirée (« depuis 16h »).
_WINDOW_CAP_SHORT = 2500
_WINDOW_CAP_DAY = 4000
_MAX_CHARS = 200_000
# Lots larges : moins d'appels. Le cache de lot évite de les refaire si seul
# le bout du fil a changé.
_CHUNK_CHARS = 24_000
_PARTIAL_MAX_TOKENS = 350
_SUMMARY_MAX_TOKENS = 700
# Cache process-local : même salon / fenêtre / focus, tant que le dernier msg
# n'a pas bougé et que l'entrée n'est pas trop vieille.
_CACHE_TTL = timedelta(hours=6)
_CACHE_MAX = 64
_PARTIAL_CACHE_MAX = 256
# Réactions sans contenu. « oui » / « non » restent : souvent une décision.
_NOISE_BODY_RE = re.compile(
    r"^(?:ok+|okay|k{2,}|mdr+|lol+|lmao|ptdr+|ah+|oh+|"
    r"ha(?:ha)+|he(?:he)+|jsp|osef|wtf|omg|\+1|-1)$",
    re.IGNORECASE,
)
_MENTION_RE = re.compile(r"<@!?\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
_NOISE_PUNCT_RE = re.compile(r"[\s!.?…,~'\"`]+")


@dataclass
class _SummaryCacheEntry:
    tip_id: int
    created_at: datetime
    summary: str
    name: str
    useful: int
    raw_count: int
    oldest: Optional[datetime]
    newest: Optional[datetime]
    focus: str


_summary_cache: dict[tuple, _SummaryCacheEntry] = {}
# Résumés de lots, clé = hash du texte. Indépendant du dernier message :
# un « ok » filtré ou un lot inchangé ne relance pas le modèle.
_partial_cache: dict[str, tuple[datetime, str]] = {}


def _cache_key(
    channel_id: int, after: Optional[datetime], limit: int, focus: str,
) -> tuple:
    after_key = (
        after.astimezone(timezone.utc).strftime("%Y%m%d%H%M") if after is not None else None
    )
    return (channel_id, after_key, limit, focus.lower())


def _cache_get(key: tuple, tip_id: int) -> Optional[_SummaryCacheEntry]:
    entry = _summary_cache.get(key)
    if entry is None:
        return None
    if entry.tip_id != tip_id:
        return None
    if datetime.now(timezone.utc) - entry.created_at > _CACHE_TTL:
        _summary_cache.pop(key, None)
        return None
    return entry


def _cache_put(key: tuple, entry: _SummaryCacheEntry) -> None:
    if len(_summary_cache) >= _CACHE_MAX and key not in _summary_cache:
        # Éviction FIFO approximative (ordre d'insertion CPython 3.7+).
        oldest_key = next(iter(_summary_cache))
        _summary_cache.pop(oldest_key, None)
    _summary_cache[key] = entry


def _text_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _partial_get(key: str) -> Optional[str]:
    hit = _partial_cache.get(key)
    if hit is None:
        return None
    created_at, text = hit
    if datetime.now(timezone.utc) - created_at > _CACHE_TTL:
        _partial_cache.pop(key, None)
        return None
    return text


def _partial_put(key: str, text: str) -> None:
    if len(_partial_cache) >= _PARTIAL_CACHE_MAX and key not in _partial_cache:
        oldest_key = next(iter(_partial_cache))
        _partial_cache.pop(oldest_key, None)
    _partial_cache[key] = (datetime.now(timezone.utc), text)


def _is_noise_body(body: str) -> bool:
    """Ping seul, emoji seul, ou réaction (« ok », « mdr ») — pas un fait à résumer."""
    text = _MENTION_RE.sub(" ", body)
    text = _CUSTOM_EMOJI_RE.sub(" ", text).strip()
    if not text or text == "[embed]":
        return True
    core = _NOISE_PUNCT_RE.sub("", text)
    return bool(core) and _NOISE_BODY_RE.fullmatch(core) is not None


async def _channel_tip_id(channel: discord.abc.Messageable) -> Optional[int]:
    """Id du message le plus récent — check cheap pour invalider le cache."""
    try:
        async for msg in channel.history(limit=1):
            return msg.id
    except (discord.Forbidden, discord.HTTPException):
        return None
    return None

_SYSTEM_FINAL = """Tu résumes une conversation Discord pour le salon.
Règles :
- Sans focus : 1 court paragraphe d'intro (sujet global), puis 3–6 puces max des points clés.
- Avec focus : réponds UNIQUEMENT à cette demande (sujet, personne, décisions…). Pas de récap général autour. 1 phrase de cadre puis 3–6 puces. Si le fil n'en parle presque pas, dis-le en une phrase.
- Cite les pseudos quand c'est utile ; ne invente rien.
- Ignore le bruit (pings seuls, « ok », réactions textuelles sans contenu).
- Pas d'emojis, pas d'intro du type « Voici le résumé ».
- Si trop peu de contenu utile : dis-le clairement en une phrase."""

_SYSTEM_PARTIAL = """Tu résumes un EXTRAIT chronologique d'une conversation Discord.
Règles :
- 2–4 puces max, densés, avec les pseudos utiles.
- Ne invente rien. Ignore le bruit.
- Pas d'intro, pas d'emojis, pas de conclusion globale (d'autres extraits suivront).
- Si focus fourni : ne retiens que ce qui sert cette demande."""

_SYSTEM_MERGE = """Tu fusionnes des résumés partiels chronologiques d'une même conversation Discord
en UN résumé final cohérent.
Règles :
- Sans focus : 1 court paragraphe d'intro, puis 3–6 puces max des points clés.
- Avec focus : réponds UNIQUEMENT à cette demande, pas de récap général. Si peu présent, dis-le.
- Déduplique, garde l'ordre du fil, cite les pseudos utiles.
- Ne invente rien. Pas d'emojis, pas d'« Voici le résumé »."""


def build_channel_summary_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget résumé de salon."""
    if not isinstance(data, dict) or "error" in data:
        return None
    return render_free_widget(data.get("spec"), commentary=commentary)


def _history_limit(raw: Any) -> int:
    """Plafond sans fenêtre de temps. Défaut 60."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return max(5, min(n, _MAX_LIMIT))


def _limit_for_span(span_hours: float) -> int:
    if span_hours <= 8:
        return _WINDOW_CAP_SHORT
    return _WINDOW_CAP_DAY


@dataclass(frozen=True)
class SummaryWindow:
    """Borne de lecture. `after` est en UTC ; None = les derniers `limit` messages."""

    after: Optional[datetime]
    limit: int
    source: str
    label: str


_CLOCK_RE = re.compile(
    r"(?:depuis|dès|des|à partir de|a partir de)\s+"
    r"(\d{1,2})\s*(?:h(?!eures?\b)|:)\s*(\d{2})?",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"(?:depuis|pendant|sur|ces|les)\s+(\d+(?:[.,]\d+)?)\s*heures?\b"
    r"|(\d+(?:[.,]\d+)?)\s*derni[eè]res?\s+heures?\b",
    re.IGNORECASE,
)
_TODAY_RE = re.compile(
    r"\b(?:aujourd['’]hui|la journ[ée]e|cette journ[ée]e|la journ[ée]e enti[eè]re)\b",
    re.IGNORECASE,
)
_YESTERDAY_RE = re.compile(r"\bdepuis hier\b", re.IGNORECASE)
_PERIOD_RES: tuple[tuple[re.Pattern[str], int, int], ...] = (
    (re.compile(r"\bce matin\b", re.IGNORECASE), 8, 0),
    (re.compile(r"\bce midi\b", re.IGNORECASE), 12, 0),
    (re.compile(r"\b(?:cet apr[eè]s[- ]midi|cette apr[eè]m)\b", re.IGNORECASE), 14, 0),
    (re.compile(r"\bce soir\b", re.IGNORECASE), 18, 0),
)


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=PARIS_TZ)
    return moment.astimezone(timezone.utc)


def _paris_now(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(PARIS_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=PARIS_TZ)
    return now.astimezone(PARIS_TZ)


def _clock_on(day: datetime, hour: int, minute: int) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _last_clock(now_paris: datetime, hour: int, minute: int) -> Optional[datetime]:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    candidate = _clock_on(now_paris, hour, minute)
    if candidate > now_paris:
        candidate -= timedelta(days=1)
    return candidate


def _parse_clock_token(raw: str) -> Optional[tuple[int, int]]:
    text = (raw or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?:h(?!eures?\b)|:)(\d{2})?", text)
    if m is None:
        m = re.fullmatch(r"(\d{1,2})h(\d{2})?", text)
    if m is None:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _duration_hours(text: str) -> Optional[float]:
    m = _DURATION_RE.search(text or "")
    if m is None:
        return None
    raw = (m.group(1) or m.group(2) or "").replace(",", ".")
    try:
        hours = float(raw)
    except ValueError:
        return None
    if hours <= 0:
        return None
    return min(hours, 72.0)


def _clock_from_text(text: str, now_paris: datetime) -> Optional[datetime]:
    m = _CLOCK_RE.search(text or "")
    if m is None:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    # « depuis 2h » = 2 heures. « depuis 16h » / « 16h30 » = l'heure.
    if m.group(2) is None and hour <= 6:
        return None
    return _last_clock(now_paris, hour, minute)


def _period_from_text(text: str, now_paris: datetime) -> Optional[datetime]:
    if _YESTERDAY_RE.search(text or ""):
        start = _clock_on(now_paris - timedelta(days=1), 0, 0)
        return start
    for pattern, hour, minute in _PERIOD_RES:
        if pattern.search(text or ""):
            return _last_clock(now_paris, hour, minute)
    if _TODAY_RE.search(text or ""):
        return _clock_on(now_paris, 0, 0)
    return None


def resolve_summary_window(
    text: str,
    *,
    since: Optional[str] = None,
    hours: Optional[float] = None,
    limit: Any = None,
    now: Optional[datetime] = None,
) -> SummaryWindow:
    """Choisit la borne. Le texte (« depuis 16h ») prime sur un hours=24 du modèle."""
    now_paris = _paris_now(now)
    blob = text or ""

    clock = _clock_from_text(blob, now_paris)
    if clock is not None:
        span = max(0.25, (now_paris - clock).total_seconds() / 3600)
        label = clock.strftime("%H:%M")
        return SummaryWindow(
            after=_as_utc(clock),
            limit=_limit_for_span(span),
            source="clock",
            label=f"depuis {label}",
        )

    period = _period_from_text(blob, now_paris)
    if period is not None:
        span = max(0.25, (now_paris - period).total_seconds() / 3600)
        return SummaryWindow(
            after=_as_utc(period),
            limit=_limit_for_span(span),
            source="period",
            label=f"depuis {period.strftime('%d/%m %H:%M')}",
        )

    duration = _duration_hours(blob)
    if duration is None:
        short = _CLOCK_RE.search(blob)
        if short is not None and short.group(2) is None:
            hour = int(short.group(1))
            if 1 <= hour <= 6:
                duration = float(hour)
    if duration is None and since:
        parsed = _parse_clock_token(since)
        if parsed is not None:
            moment = _last_clock(now_paris, parsed[0], parsed[1])
            if moment is not None:
                span = max(0.25, (now_paris - moment).total_seconds() / 3600)
                return SummaryWindow(
                    after=_as_utc(moment),
                    limit=_limit_for_span(span),
                    source="since",
                    label=f"depuis {moment.strftime('%H:%M')}",
                )
        duration = _duration_hours(since)

    if duration is None and hours is not None and 0 < hours <= 12:
        duration = hours
    if duration is not None:
        after = now_paris - timedelta(hours=duration)
        shown = int(duration) if duration == int(duration) else round(duration, 1)
        return SummaryWindow(
            after=_as_utc(after),
            limit=_limit_for_span(duration),
            source="duration",
            label=f"{shown} h",
        )

    return SummaryWindow(
        after=None,
        limit=_history_limit(limit),
        source="recent",
        label=f"{_history_limit(limit)} derniers",
    )


def _resolve_channel(ctx, arguments: dict) -> tuple[Optional[discord.abc.Messageable], Optional[str]]:
    trigger = getattr(ctx, "trigger_message", None) if ctx else None
    if not trigger:
        return None, "Contexte manquant"

    cid_str = (arguments.get("channel_id") or "").strip()
    if not cid_str:
        ch = trigger.channel
        if isinstance(ch, (discord.TextChannel, discord.Thread)):
            return ch, None
        return None, "Salon non textuel"

    try:
        cid = int(cid_str)
    except ValueError:
        return None, "channel_id invalide"

    channel = None
    if trigger.guild:
        channel = trigger.guild.get_channel(cid) or trigger.guild.get_thread(cid)
    if channel is None:
        return None, "Salon introuvable"
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        return None, "Salon non textuel"
    return channel, None


def _format_line(msg: discord.Message) -> Optional[str]:
    content = (msg.content or "").strip()
    if not content and not msg.attachments and not msg.embeds:
        return None
    author = getattr(msg.author, "display_name", None) or msg.author.name
    when = msg.created_at.astimezone(PARIS_TZ).strftime("%H:%M")
    bits: list[str] = []
    if content:
        bits.append(content.replace("\n", " "))
    if msg.attachments:
        bits.append(f"[{len(msg.attachments)} pièce(s) jointe(s)]")
    if msg.embeds and not content:
        bits.append("[embed]")
    body = " ".join(bits).strip()
    if not body or _is_noise_body(body):
        return None
    return f"[{when}] {author}: {body[:400]}"


async def _fetch_transcript(
    channel: discord.abc.Messageable,
    *,
    limit: int,
    after: Optional[datetime],
) -> tuple[list[str], int, Optional[datetime], Optional[datetime]]:
    lines: list[str] = []
    oldest: Optional[datetime] = None
    newest: Optional[datetime] = None
    raw_count = 0
    chars = 0

    async for msg in channel.history(limit=limit, after=after, oldest_first=False):
        raw_count += 1
        if oldest is None or msg.created_at < oldest:
            oldest = msg.created_at
        if newest is None or msg.created_at > newest:
            newest = msg.created_at
        line = _format_line(msg)
        if not line:
            continue
        if chars + len(line) > _MAX_CHARS:
            break
        lines.append(line)
        chars += len(line) + 1

    lines.reverse()  # chronologique
    return lines, raw_count, oldest, newest


def _chunk_lines(lines: list[str], max_chars: int = _CHUNK_CHARS) -> list[list[str]]:
    """Découpe le transcript en lots chronologiques par budget caractères."""
    chunks: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for line in lines:
        cost = len(line) + 1
        if current and chars + cost > max_chars:
            chunks.append(current)
            current = []
            chars = 0
        current.append(line)
        chars += cost
    if current:
        chunks.append(current)
    return chunks


async def _llm_text(
    llm_client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> str:
    completion = await llm_client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


async def _cached_llm_text(
    llm_client: Any,
    *,
    kind: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    key_parts: list[str],
) -> tuple[str, bool]:
    """Retourne (texte, True si servit par le cache)."""
    key = _text_key(kind, model, *key_parts)
    cached = _partial_get(key)
    if cached is not None:
        return cached, True
    text = await _llm_text(
        llm_client,
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
    )
    if text:
        _partial_put(key, text)
    return text, False


async def _summarize(
    llm_client: Any,
    *,
    model: str,
    channel_name: str,
    lines: list[str],
    focus: str,
) -> str:
    focus_block = (
        f"\nDemande (pas un résumé général) : {focus}\n"
        if focus else ""
    )
    chunks = _chunk_lines(lines)
    calls = 0
    hits = 0

    async def _call(
        kind: str,
        system: str,
        user: str,
        max_tokens: int,
        key_parts: list[str],
    ) -> str:
        nonlocal calls, hits
        text, hit = await _cached_llm_text(
            llm_client,
            kind=kind,
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            key_parts=key_parts,
        )
        if hit:
            hits += 1
        else:
            calls += 1
        return text

    # Petit volume → une seule passe (comportement d'origine).
    if len(chunks) <= 1:
        user = (
            f"Salon : #{channel_name}\n"
            f"{focus_block}"
            f"Messages ({len(lines)}) :\n"
            + "\n".join(lines)
        )
        summary = await _call(
            "final", _SYSTEM_FINAL, user, _SUMMARY_MAX_TOKENS,
            [channel_name, focus, "\n".join(lines)],
        )
        logger.info(
            "Résumé salon #%s : %d msgs, 1 lot, %d appel(s), %d cache",
            channel_name, len(lines), calls, hits,
        )
        return summary

    # Gros volume → résumés partiels puis fusion. Chaque lot est caché sur son
    # texte : seul le lot qui a changé, puis la fusion, rappellent le modèle.
    partials: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        body = "\n".join(chunk)
        user = (
            f"Salon : #{channel_name}\n"
            f"Extrait {i}/{len(chunks)} ({len(chunk)} msgs)\n"
            f"{focus_block}"
            f"Messages :\n"
            f"{body}"
        )
        part = await _call(
            "partial", _SYSTEM_PARTIAL, user, _PARTIAL_MAX_TOKENS,
            [channel_name, focus, body],
        )
        if part:
            partials.append(part)

    if not partials:
        logger.info(
            "Résumé salon #%s : %d lots vides, %d appel(s), %d cache",
            channel_name, len(chunks), calls, hits,
        )
        return ""

    numbered = [f"[Extrait {i}/{len(partials)}]\n{part}" for i, part in enumerate(partials, start=1)]
    if len(numbered) == 1:
        user = (
            f"Salon : #{channel_name}\n"
            f"{focus_block}"
            f"Notes :\n{numbered[0]}"
        )
        kind = "merge-one"
    else:
        user = (
            f"Salon : #{channel_name}\n"
            f"{focus_block}"
            f"Résumés partiels chronologiques ({len(numbered)}) :\n\n"
            + "\n\n".join(numbered)
        )
        kind = "merge"
    summary = await _call(
        kind, _SYSTEM_MERGE, user, _SUMMARY_MAX_TOKENS,
        [channel_name, focus, "\n\n".join(partials)],
    )
    logger.info(
        "Résumé salon #%s : %d msgs → %d lots, %d appel(s), %d cache",
        channel_name, len(lines), len(chunks), calls, hits,
    )
    return summary


def _build_summary_payload(
    *,
    name: str,
    summary: str,
    useful: int,
    raw_count: int,
    oldest: Optional[datetime],
    newest: Optional[datetime],
    focus: str,
    from_cache: bool,
) -> dict:
    if oldest and newest:
        o = oldest.astimezone(PARIS_TZ).strftime("%d/%m %H:%M")
        n = newest.astimezone(PARIS_TZ).strftime("%H:%M")
        window = f"{o} → {n}"
    else:
        window = "fenêtre récente"

    footer = f"{useful}/{raw_count} · {window}"
    if from_cache:
        footer += " · cache"

    # Pas de coupure brute ici : render_free_widget (_text_block) tronque déjà
    # proprement (fin de phrase/mot) si le résumé dépasse la place disponible.
    # Le focus va en sous-titre, pas dans le ## (sinon Discord/nos [:40] coupent).
    content = summary
    title = f"Résumé — #{name}"
    blocks: list[dict] = []
    if focus:
        blocks.append({"type": "text", "content": f"-# {focus.strip()}"})
    blocks.append({"type": "text", "content": content})
    blocks.append({"type": "footer", "text": footer})
    spec = {
        "title": title,
        "emoji": RESUME,
        "blocks": blocks,
    }
    note = (
        f"Résumé de #{name} ({useful} msgs"
        + (f", demande : {focus[:80]}" if focus else "")
        + (", cache" if from_cache else "")
        + ") : "
        + (summary.strip()[:1500] if summary else "(vide)")
    )
    return {
        "_tool": "summarize_channel",
        "_llm_summary": note,
        "spec": spec,
        "channel": name,
        "messages_used": useful,
        "cached": from_cache,
    }


def build_channel_summary_tools(
    llm_client: Any,
    *,
    model: str,
) -> list[Tool]:
    """Construit l'outil summarize_channel."""

    async def _tool_summarize_channel(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        args = tc.arguments or {}
        channel, err = _resolve_channel(ctx, args)
        if err or channel is None:
            return ToolResponseRecord(tc.id, {"error": err or "Salon introuvable"}, datetime.now(timezone.utc))

        hours_raw = args.get("hours")
        hours: Optional[float] = None
        if hours_raw is not None:
            try:
                hours = float(hours_raw)
            except (TypeError, ValueError):
                hours = None
            if hours is not None and hours <= 0:
                hours = None
            elif hours is not None:
                hours = min(hours, 72.0)

        focus = (args.get("focus") or "").strip() if isinstance(args.get("focus"), str) else ""
        since_raw = args.get("since")
        since = since_raw.strip() if isinstance(since_raw, str) and since_raw.strip() else None
        trigger = getattr(ctx, "trigger_message", None)
        user_text = ""
        if trigger is not None:
            user_text = (
                getattr(trigger, "clean_content", None)
                or getattr(trigger, "content", "")
                or ""
            )
        window = resolve_summary_window(
            f"{user_text}\n{focus}",
            since=since,
            hours=hours,
            limit=args.get("limit"),
        )
        limit = window.limit
        name = getattr(channel, "name", None) or str(channel.id)
        logger.info(
            "Résumé salon #%s fenêtre=%s source=%s limit=%d (modèle hours=%s since=%s)",
            name, window.label, window.source, limit, hours, since,
        )
        cache_key = _cache_key(channel.id, window.after, limit, focus)

        tip_id = await _channel_tip_id(channel)
        if tip_id is not None:
            cached = _cache_get(cache_key, tip_id)
            if cached is not None:
                logger.info("Résumé salon #%s : cache hit", name)
                return ToolResponseRecord(
                    tc.id,
                    _build_summary_payload(
                        name=cached.name,
                        summary=cached.summary,
                        useful=cached.useful,
                        raw_count=cached.raw_count,
                        oldest=cached.oldest,
                        newest=cached.newest,
                        focus=cached.focus,
                        from_cache=True,
                    ),
                    datetime.now(timezone.utc),
                )

        try:
            lines, raw_count, oldest, newest = await _fetch_transcript(
                channel, limit=limit, after=window.after,
            )
        except discord.Forbidden:
            return ToolResponseRecord(
                tc.id,
                {"error": "Pas la permission de lire l'historique de ce salon."},
                datetime.now(timezone.utc),
            )
        except discord.HTTPException as e:
            return ToolResponseRecord(
                tc.id, {"error": f"Lecture historique échouée : {e}"}, datetime.now(timezone.utc),
            )

        if not lines:
            return ToolResponseRecord(
                tc.id,
                {"error": "Pas assez de messages utiles à résumer dans cette fenêtre."},
                datetime.now(timezone.utc),
            )

        try:
            summary = await _summarize(
                llm_client, model=model, channel_name=name, lines=lines, focus=focus,
            )
        except Exception as e:
            logger.warning("Résumé salon échoué: %s", e)
            return ToolResponseRecord(
                tc.id, {"error": "Échec de la génération du résumé."}, datetime.now(timezone.utc),
            )

        if not summary:
            return ToolResponseRecord(
                tc.id, {"error": "Résumé vide."}, datetime.now(timezone.utc),
            )

        if tip_id is not None:
            _cache_put(cache_key, _SummaryCacheEntry(
                tip_id=tip_id,
                created_at=datetime.now(timezone.utc),
                summary=summary,
                name=name,
                useful=len(lines),
                raw_count=raw_count,
                oldest=oldest,
                newest=newest,
                focus=focus,
            ))

        return ToolResponseRecord(
            tc.id,
            _build_summary_payload(
                name=name,
                summary=summary,
                useful=len(lines),
                raw_count=raw_count,
                oldest=oldest,
                newest=newest,
                focus=focus,
                from_cache=False,
            ),
            datetime.now(timezone.utc),
        )

    return [
        Tool(
            name="summarize_channel",
            description=(
                "Résume un salon/thread Discord et l'affiche en widget. "
                "« résume le salon » sans borne → ne pas passer since ni hours "
                "(les derniers messages suffisent). "
                "« depuis 16h » ou « depuis 16h30 » est une heure, pas une durée : "
                "since=\"16:00\" ou \"16:30\". Jamais hours=24 pour ça. "
                "« les 3 dernières heures » ou « depuis 3 heures » → hours=3. "
                "« aujourd'hui » ou « la journée » → since=\"00:00\". "
                "Demande précise (« ce qu'a dit X », « les décisions ») → focus, "
                "pas un récap global. Défaut : salon actuel. "
                "Après l'appel : aucun texte autour — le widget contient déjà le résumé."
            ),
            properties={
                "channel_id": {
                    "type": "string",
                    "description": "ID du salon/thread (optionnel, défaut = salon actuel)",
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Heure de départ à Paris, pas une durée. "
                        "« depuis 16h » → \"16:00\", « depuis 16h30 » → \"16:30\", "
                        "« aujourd'hui » → \"00:00\". Ne pas combiner avec hours."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Sans since ni hours : nombre max de messages "
                        f"(défaut {_DEFAULT_LIMIT}, max {_MAX_LIMIT})."
                    ),
                },
                "hours": {
                    "type": "number",
                    "description": (
                        "Durée glissante, seulement pour « les N dernières heures » "
                        "ou « depuis N heures ». Max 12. "
                        "Interdit pour une heure (« depuis 16h ») et pour une journée."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Demande précise à traiter à la place d'un résumé général "
                        "(sujet, personne, décisions, blagues, le plan…). "
                        "Vide = récap global."
                    ),
                },
            },
            optional_props=["channel_id", "since", "limit", "hours", "focus"],
            function=_tool_summarize_channel,
        ),
    ]
