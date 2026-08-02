"""Outil LLM « carte de membre » — vitrine du widget libre (render_widget)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.memory.store import CATEGORY_USER, MemoryStore
from common.widget_catalog import render_free_widget
from cogs.chat.tools_reminders import _resolve_member

_MAX_FACTS = 5


def build_member_card_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget « carte de membre » — réutilise le renderer libre."""
    if not isinstance(data, dict) or "error" in data:
        return None
    return render_free_widget(data.get("spec"), commentary=commentary)


def build_member_card_tools(store: MemoryStore) -> list[Tool]:
    """Construit l'outil show_member_card."""

    async def _tool_show_member_card(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        guild = ctx.trigger_message.guild
        member, err = await _resolve_member(ctx, tc.arguments or {})
        if err or member is None:
            return ToolResponseRecord(tc.id, {"error": err or "Membre introuvable"}, datetime.now(timezone.utc))

        facts = store.list_for_user(guild.id, member.id, limit=_MAX_FACTS)
        fact_lines = []
        for m in facts:
            if m.category != CATEGORY_USER:
                continue
            content = m.content
            if ":" in content:
                content = content.split(":", 1)[1].strip()
            fact_lines.append(content)

        name = getattr(member, "display_name", None) or member.name
        avatar_url = str(member.display_avatar.url) if getattr(member, "display_avatar", None) else None
        joined = getattr(member, "joined_at", None)
        joined_str = joined.strftime("%d/%m/%Y") if joined else None

        text = "\n".join(f"• {f}" for f in fact_lines) or "Pas grand-chose de retenu pour l'instant."
        footer = f"Sur le serveur depuis le {joined_str}" if joined_str else "Membre du serveur"

        blocks: list[dict] = []
        if avatar_url:
            blocks.append({"type": "thumbnail", "url": avatar_url, "text": text})
        else:
            blocks.append({"type": "text", "content": text})
        blocks.append({"type": "footer", "text": footer})

        spec = {"title": f"Carte de membre — {name}", "emoji": "🪪", "blocks": blocks}

        return ToolResponseRecord(tc.id, {
            "_tool":        "show_member_card",
            "_llm_summary": f"Carte de {name} affichée dans le salon.",
            "spec":         spec,
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="show_member_card",
            description=(
                "Affiche une carte visuelle résumant ce que MARIA sait sur un membre "
                "(avatar + faits retenus). Pour « dis-moi qui est X », « fais la carte de X », "
                "pas pour un simple avis en tchat. Défaut : l'auteur du message."
            ),
            properties={
                "user_id": {
                    "type": "string",
                    "description": "Id Discord (optionnel, défaut = auteur)",
                },
                "username": {
                    "type": "string",
                    "description": "Pseudo Discord (optionnel)",
                },
            },
            optional_props=["user_id", "username"],
            function=_tool_show_member_card,
        ),
    ]
