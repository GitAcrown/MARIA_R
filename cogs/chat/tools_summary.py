"""Outil LLM — résumé d'un salon / thread Discord.

Stratégie : si le transcript est court → 1 passe. Sinon → résumés partiels
par lots, puis synthèse finale (map-reduce léger).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import discord

from common.emojis import RESUME
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.timezones import PARIS_TZ
from common.widget_catalog import render_free_widget

logger = logging.getLogger("MARIA.Chat.Summary")

_DEFAULT_LIMIT = 60
_MAX_LIMIT = 500
# Fenêtre longue (journée…) : on lit plus sans attendre que le LLM monte le limit.
_LONG_WINDOW_LIMIT = 400
_MAX_CHARS = 80_000
_CHUNK_CHARS = 10_000
_PARTIAL_MAX_TOKENS = 350
_SUMMARY_MAX_TOKENS = 700
# Cache process-local : même salon / fenêtre / focus, tant que le dernier msg
# n'a pas bougé et que l'entrée n'est pas trop vieille.
_CACHE_TTL = timedelta(hours=6)
_CACHE_MAX = 64


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


def _cache_key(channel_id: int, hours: Optional[float], limit: int, focus: str) -> tuple:
    hours_key = round(hours, 2) if hours is not None else None
    return (channel_id, hours_key, limit, focus.lower())


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
- 1 court paragraphe d'intro (sujet global), puis 3–6 puces max des points clés.
- Cite les pseudos quand c'est utile ; ne invente rien.
- Ignore le bruit (pings seuls, « ok », réactions textuelles sans contenu).
- Pas d'emojis, pas d'intro du type « Voici le résumé ».
- Si focus fourni : concentre-toi dessus, signale si peu présent.
- Si trop peu de contenu utile : dis-le clairement en une phrase."""

_SYSTEM_PARTIAL = """Tu résumes un EXTRAIT chronologique d'une conversation Discord.
Règles :
- 2–4 puces max, densés, avec les pseudos utiles.
- Ne invente rien. Ignore le bruit.
- Pas d'intro, pas d'emojis, pas de conclusion globale (d'autres extraits suivront).
- Si focus fourni : privilégie cet angle."""

_SYSTEM_MERGE = """Tu fusionnes des résumés partiels chronologiques d'une même conversation Discord
en UN résumé final cohérent.
Règles :
- 1 court paragraphe d'intro, puis 3–6 puces max des points clés.
- Déduplique, garde l'ordre du fil, cite les pseudos utiles.
- Ne invente rien. Pas d'emojis, pas d'« Voici le résumé ».
- Si focus fourni : concentre-toi dessus."""


def build_channel_summary_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget résumé de salon."""
    if not isinstance(data, dict) or "error" in data:
        return None
    return render_free_widget(data.get("spec"), commentary=commentary)


def _clamp_limit(raw: Any, *, hours: Optional[float]) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        # Journée / fenêtre longue → lire large sans forcer le LLM à le demander.
        if hours is not None and hours >= 12:
            return _LONG_WINDOW_LIMIT
        return _DEFAULT_LIMIT
    return max(5, min(n, _MAX_LIMIT))


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
    if not body:
        return None
    return f"[{when}] {author}: {body[:400]}"


async def _fetch_transcript(
    channel: discord.abc.Messageable,
    *,
    limit: int,
    hours: Optional[float],
) -> tuple[list[str], int, Optional[datetime], Optional[datetime]]:
    after = None
    if hours is not None and hours > 0:
        after = datetime.now(timezone.utc) - timedelta(hours=hours)

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


async def _summarize(
    llm_client: Any,
    *,
    model: str,
    channel_name: str,
    lines: list[str],
    focus: str,
) -> str:
    focus_block = f"\nFocus demandé : {focus}\n" if focus else ""
    chunks = _chunk_lines(lines)

    # Petit volume → une seule passe (comportement d'origine).
    if len(chunks) <= 1:
        user = (
            f"Salon : #{channel_name}\n"
            f"{focus_block}"
            f"Messages ({len(lines)}) :\n"
            + "\n".join(lines)
        )
        return await _llm_text(
            llm_client,
            model=model,
            system=_SYSTEM_FINAL,
            user=user,
            max_tokens=_SUMMARY_MAX_TOKENS,
        )

    # Gros volume → résumés partiels puis fusion.
    logger.info(
        "Résumé salon #%s : %d msgs → %d lots",
        channel_name, len(lines), len(chunks),
    )
    partials: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        user = (
            f"Salon : #{channel_name}\n"
            f"Extrait {i}/{len(chunks)} ({len(chunk)} msgs)\n"
            f"{focus_block}"
            f"Messages :\n"
            + "\n".join(chunk)
        )
        part = await _llm_text(
            llm_client,
            model=model,
            system=_SYSTEM_PARTIAL,
            user=user,
            max_tokens=_PARTIAL_MAX_TOKENS,
        )
        if part:
            partials.append(f"[Extrait {i}/{len(chunks)}]\n{part}")

    if not partials:
        return ""
    if len(partials) == 1:
        # Un seul partial utile → reformule en format final.
        user = (
            f"Salon : #{channel_name}\n"
            f"{focus_block}"
            f"Notes :\n{partials[0]}"
        )
        return await _llm_text(
            llm_client,
            model=model,
            system=_SYSTEM_MERGE,
            user=user,
            max_tokens=_SUMMARY_MAX_TOKENS,
        )

    user = (
        f"Salon : #{channel_name}\n"
        f"{focus_block}"
        f"Résumés partiels chronologiques ({len(partials)}) :\n\n"
        + "\n\n".join(partials)
    )
    return await _llm_text(
        llm_client,
        model=model,
        system=_SYSTEM_MERGE,
        user=user,
        max_tokens=_SUMMARY_MAX_TOKENS,
    )


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

    footer = f"{useful} msgs utiles / {raw_count} lus · {window}"
    if focus:
        footer += f" · focus : {focus[:60]}"
    if from_cache:
        footer += " · cache"

    # Pas de coupure brute ici : render_free_widget (_text_block) tronque déjà
    # proprement (fin de phrase/mot) si le résumé dépasse la place disponible.
    content = summary
    spec = {
        "title": f"Résumé — #{name}",
        "emoji": RESUME,
        "blocks": [
            {"type": "text", "content": content},
            {"type": "footer", "text": footer},
        ],
    }
    note = (
        f"Résumé de #{name} (cache, {useful} msgs)."
        if from_cache
        else f"Résumé de #{name} affiché ({useful} msgs)."
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

        limit = _clamp_limit(args.get("limit"), hours=hours)
        focus = (args.get("focus") or "").strip()
        name = getattr(channel, "name", None) or str(channel.id)
        cache_key = _cache_key(channel.id, hours, limit, focus)

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
                channel, limit=limit, hours=hours,
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
                "Résume la conversation récente d'un salon/thread Discord et l'affiche "
                "en widget. Pour « résume le salon », « c'était quoi ce fil », "
                "« récap des derniers messages », « résume la journée ». Défaut : salon actuel. "
                "Pour une journée : hours=24 (le volume est géré automatiquement, lots + synthèse). "
                "Ne pas utiliser pour un simple avis en tchat."
            ),
            properties={
                "channel_id": {
                    "type": "string",
                    "description": "ID du salon/thread (optionnel, défaut = salon actuel)",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Nombre max de messages à lire (défaut {_DEFAULT_LIMIT}, "
                        f"auto ~{_LONG_WINDOW_LIMIT} si hours≥12, max {_MAX_LIMIT})."
                    ),
                },
                "hours": {
                    "type": "number",
                    "description": "Ne garder que les messages des N dernières heures (optionnel, max 72)",
                },
                "focus": {
                    "type": "string",
                    "description": "Angle du résumé (ex. décisions, blagues, un sujet précis)",
                },
            },
            optional_props=["channel_id", "limit", "hours", "focus"],
            function=_tool_summarize_channel,
        ),
    ]
