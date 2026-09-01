"""Outils LLM liés aux métadonnées Discord (membres, salons)."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import discord

from common.activity import ActivityTracker, KIND_MESSAGE, KIND_SUMMON
from common.discord_ui import layout_with_commentary, section_with_thumbnail
from common.emojis import SMALL_CHART
from common.llm import Tool, ToolCallRecord, ToolResponseRecord

# Nombre maximum de membres renvoyés par get_server_users.
MAX_SERVER_USERS = 60
# Fenêtre des classements de la vue get_server_stats.
SERVER_STATS_DAYS = 7


def build_server_stats_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if not isinstance(data, dict) or "error" in data:
        return None
    name = data.get("guild_name") or "Serveur"
    days = data.get("days") or SERVER_STATS_DAYS
    # Section + thumbnail d'abord, PUIS Container — une Section ne peut pas
    # envelopper un Container (Discord 400, message jamais posté).
    header = section_with_thumbnail(
        discord.ui.TextDisplay(
            f"## {SMALL_CHART} {name}\n"
            f"-# {data.get('member_count', '?')} membres · "
            f"{data.get('messages_total', 0)} messages sur {days} j"
        ),
        data.get("guild_icon"),
    )
    children: list[discord.ui.Item] = [header, discord.ui.Separator()]

    def _ranked_block(title: str, items: list[dict], unit: str) -> Optional[discord.ui.TextDisplay]:
        if not items:
            return None
        lines = [f"{i + 1}. **{it['name']}** — {it['count']} {unit}" for i, it in enumerate(items)]
        return discord.ui.TextDisplay(f"**{title}**\n" + "\n".join(lines))

    blocks = [
        b for b in (
            _ranked_block("Salons les plus actifs", data.get("top_channels") or [], "messages"),
            _ranked_block("Membres les plus bavards", data.get("top_members") or [], "messages"),
            _ranked_block("Sollicitent le plus MARIA", data.get("top_summoners") or [], "fois"),
        ) if b is not None
    ]
    if not blocks:
        children.append(discord.ui.TextDisplay("-# Pas encore assez de données (revient dans quelques jours)."))
    else:
        for i, b in enumerate(blocks):
            if i:
                children.append(discord.ui.Separator(spacing=discord.SeparatorSpacing.small))
            children.append(b)

    return layout_with_commentary(discord.ui.Container(*children), commentary)


def build_discord_tools(activity: ActivityTracker) -> list[Tool]:
    """Construit les outils d'introspection serveur / membres / salons."""

    async def _tool_server_users(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        guild = ctx.trigger_message.guild
        if not guild:
            return ToolResponseRecord(tc.id, {"error": "Pas dans un serveur"}, datetime.now(timezone.utc))
        search = (tc.arguments.get("search") or "").strip().lower()
        pool = guild.members
        if search:
            pool = [m for m in pool if search in m.name.lower() or search in m.display_name.lower()]
        pool = pool[:MAX_SERVER_USERS]
        return ToolResponseRecord(tc.id, {
            "total_members": guild.member_count,
            "shown": len(pool),
            "members": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "id": str(m.id),
                    "top_roles": [r.name for r in m.roles if r.name != "@everyone"][-4:],
                }
                for m in pool
            ],
        }, datetime.now(timezone.utc))

    async def _tool_member_info(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        guild = ctx.trigger_message.guild
        if not guild:
            return ToolResponseRecord(tc.id, {"error": "Pas dans un serveur"}, datetime.now(timezone.utc))
        uid_str = (tc.arguments.get("user_id") or "").strip()
        name_q = (tc.arguments.get("username") or "").strip().lower()
        member = None
        if uid_str:
            try:
                member = guild.get_member(int(uid_str))
                if not member:
                    member = await guild.fetch_member(int(uid_str))
            except (ValueError, discord.NotFound):
                pass
        if not member and name_q:
            member = discord.utils.find(
                lambda m: m.name.lower() == name_q or m.display_name.lower() == name_q,
                guild.members,
            )
        if not member:
            return ToolResponseRecord(tc.id, {"error": "Membre introuvable"}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "id": str(member.id),
            "username": member.name,
            "display_name": member.display_name,
            "roles": [r.name for r in member.roles if r.name != "@everyone"],
            "account_created": member.created_at.strftime("%Y-%m-%d"),
            "joined_server": member.joined_at.strftime("%Y-%m-%d") if member.joined_at else None,
            "is_bot": member.bot,
            "avatar_url": str(member.display_avatar.url) if member.display_avatar else None,
        }, datetime.now(timezone.utc))

    async def _tool_channel_info(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        cid_str = (tc.arguments.get("channel_id") or "").strip()
        if cid_str:
            channel = (
                ctx.trigger_message.guild.get_channel(int(cid_str))
                if ctx.trigger_message.guild
                else None
            )
        else:
            channel = ctx.trigger_message.channel
        if not channel:
            return ToolResponseRecord(tc.id, {"error": "Salon introuvable"}, datetime.now(timezone.utc))
        info: dict = {"id": str(channel.id), "name": channel.name, "type": str(channel.type)}
        if isinstance(channel, discord.TextChannel):
            info.update({
                "topic": channel.topic or "",
                "category": channel.category.name if channel.category else None,
                "nsfw": channel.nsfw,
                "slowmode_delay": channel.slowmode_delay,
                "member_count": len(channel.members),
            })
        elif isinstance(channel, discord.Thread):
            info.update({
                "parent": channel.parent.name if channel.parent else None,
                "archived": channel.archived,
                "member_count": channel.member_count,
            })
        elif isinstance(channel, discord.VoiceChannel):
            info.update({
                "category": channel.category.name if channel.category else None,
                "user_limit": channel.user_limit,
                "members_connected": [m.name for m in channel.members],
            })
        return ToolResponseRecord(tc.id, info, datetime.now(timezone.utc))

    async def _tool_server_stats(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc))
        guild = ctx.trigger_message.guild
        days = SERVER_STATS_DAYS

        top_channels_raw, top_members_raw, top_summoners_raw, total = await asyncio.gather(
            asyncio.to_thread(activity.top_channels, guild.id, days=days, limit=3),
            asyncio.to_thread(activity.top_users, guild.id, days=days, limit=3, kind=KIND_MESSAGE),
            asyncio.to_thread(activity.top_users, guild.id, days=days, limit=3, kind=KIND_SUMMON),
            asyncio.to_thread(activity.guild_message_count, guild.id, days=days),
        )

        def _channel_name(cid: int) -> str:
            ch = guild.get_channel(cid)
            return f"#{ch.name}" if ch else "salon supprimé"

        def _member_name(uid: int) -> str:
            m = guild.get_member(uid)
            return m.display_name if m else "membre parti"

        return ToolResponseRecord(tc.id, {
            "_tool": "get_server_stats",
            "_llm_summary": f"Stats de {guild.name} sur {days} j affichées (vue). Ne les récite pas.",
            "guild_name": guild.name,
            "guild_icon": str(guild.icon.url) if guild.icon else None,
            "member_count": guild.member_count,
            "days": days,
            "messages_total": total,
            "top_channels": [{"name": _channel_name(cid), "count": n} for cid, n in top_channels_raw],
            "top_members": [{"name": _member_name(uid), "count": n} for uid, n in top_members_raw],
            "top_summoners": [{"name": _member_name(uid), "count": n} for uid, n in top_summoners_raw],
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="get_server_users",
            description="Liste les membres du serveur avec leurs rôles principaux. Paramètre optionnel 'search' pour filtrer par nom.",
            properties={"search": {"type": "string", "description": "Filtre par nom ou pseudo (optionnel)"}},
            function=_tool_server_users,
        ),
        Tool(
            name="get_member_info",
            description="Carte d'identité complète d'un membre : rôles, dates de création et d'arrivée, avatar. Recherche par ID ou pseudo exact.",
            properties={
                "user_id": {"type": "string", "description": "ID Discord (prioritaire)"},
                "username": {"type": "string", "description": "Nom d'utilisateur ou pseudo (recherche exacte)"},
            },
            function=_tool_member_info,
        ),
        Tool(
            name="get_channel_info",
            description="Informations sur un salon Discord : sujet, catégorie, NSFW, slowmode, membres présents. Par défaut le salon actuel.",
            properties={"channel_id": {"type": "string", "description": "ID du salon (optionnel, défaut = salon actuel)"}},
            function=_tool_channel_info,
        ),
        Tool(
            name="get_server_stats",
            description=(
                "Statistiques communautaires du serveur sur les 7 derniers jours : salon le plus "
                "actif, membres les plus bavards, membres qui te sollicitent le plus. Vue dédiée, "
                "aucun paramètre. Uniquement si on demande explicitement les stats/l'activité du serveur."
            ),
            properties={},
            function=_tool_server_stats,
        ),
    ]
