"""Outil LLM — résumé d'un salon / thread Discord."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import discord

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.timezones import PARIS_TZ
from common.widget_catalog import render_free_widget

logger = logging.getLogger("MARIA.Chat.Summary")

_DEFAULT_LIMIT = 40
_MAX_LIMIT = 80
_MAX_CHARS = 12_000
_SUMMARY_MAX_TOKENS = 500

_SYSTEM = """Tu résumes une conversation Discord pour le salon.
Règles :
- 1 court paragraphe d'intro (sujet global), puis 3–6 puces max des points clés.
- Cite les pseudos quand c'est utile ; ne invente rien.
- Ignore le bruit (pings seuls, « ok », réactions textuelles sans contenu).
- Pas d'emojis, pas d'intro du type « Voici le résumé ».
- Si focus fourni : concentre-toi dessus, signale si peu présent.
- Si trop peu de contenu utile : dis-le clairement en une phrase."""


def build_channel_summary_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget résumé de salon."""
    if not isinstance(data, dict) or "error" in data:
        return None
    return render_free_widget(data.get("spec"), commentary=commentary)


def _clamp_limit(raw: Any) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
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


async def _summarize(
    llm_client: Any,
    *,
    model: str,
    channel_name: str,
    lines: list[str],
    focus: str,
) -> str:
    focus_block = f"\nFocus demandé : {focus}\n" if focus else ""
    user = (
        f"Salon : #{channel_name}\n"
        f"{focus_block}"
        f"Messages ({len(lines)}) :\n"
        + "\n".join(lines)
    )
    completion = await llm_client.chat(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=_SUMMARY_MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()


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

        limit = _clamp_limit(args.get("limit"))
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

        focus = (args.get("focus") or "").strip()

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

        name = getattr(channel, "name", None) or str(channel.id)
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

        # Fenêtre temporelle pour le footer
        if oldest and newest:
            o = oldest.astimezone(PARIS_TZ).strftime("%d/%m %H:%M")
            n = newest.astimezone(PARIS_TZ).strftime("%H:%M")
            window = f"{o} → {n}"
        else:
            window = "fenêtre récente"

        footer = f"{len(lines)} msgs utiles / {raw_count} lus · {window}"
        if focus:
            footer += f" · focus : {focus[:60]}"

        # Découpe douce si le résumé dépasse la limite widget
        content = summary[:800]
        spec = {
            "title": f"Résumé — #{name}",
            "emoji": "📋",
            "blocks": [
                {"type": "text", "content": content},
                {"type": "footer", "text": footer},
            ],
        }

        return ToolResponseRecord(tc.id, {
            "_tool": "summarize_channel",
            "_llm_summary": f"Résumé de #{name} affiché ({len(lines)} msgs).",
            "spec": spec,
            "channel": name,
            "messages_used": len(lines),
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="summarize_channel",
            description=(
                "Résume la conversation récente d'un salon/thread Discord et l'affiche "
                "en widget. Pour « résume le salon », « c'était quoi ce fil », "
                "« récap des derniers messages ». Défaut : salon actuel. "
                "Ne pas utiliser pour un simple avis en tchat."
            ),
            properties={
                "channel_id": {
                    "type": "string",
                    "description": "ID du salon/thread (optionnel, défaut = salon actuel)",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Nombre max de messages à lire (défaut {_DEFAULT_LIMIT}, max {_MAX_LIMIT})",
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
