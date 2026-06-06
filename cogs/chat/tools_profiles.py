"""Outils LLM liés aux profils / notes utilisateur."""

from datetime import datetime, timezone
from typing import Optional

import discord

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.profiles import ProfileStore


def build_profile_tools(profiles: ProfileStore) -> list[Tool]:
    """Construit les outils de mémoire utilisateur (notes dynamiques)."""

    async def _tool_update_notes(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        notes = (tc.arguments.get("addition") or "").strip()
        if not notes or not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Données manquantes"}, datetime.now(timezone.utc))

        user_name = (tc.arguments.get("user_name") or "").strip().lower()
        target_id = ctx.trigger_message.author.id
        if user_name and ctx.trigger_message.guild:
            member = discord.utils.find(
                lambda m: m.name.lower() == user_name or m.display_name.lower() == user_name,
                ctx.trigger_message.guild.members,
            )
            if member:
                target_id = member.id

        for line in notes.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("["):
                line = f"[perso] {line}"
            profiles.append_notes(target_id, line)
        return ToolResponseRecord(tc.id, {"success": True}, datetime.now(timezone.utc))

    async def _tool_search_notes(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        keyword = (tc.arguments.get("keyword") or "").strip()
        if not keyword:
            return ToolResponseRecord(tc.id, {"error": "Mot-clé manquant"}, datetime.now(timezone.utc))
        matches = profiles.search_notes(keyword)
        if not matches:
            return ToolResponseRecord(tc.id, {"results": [], "count": 0}, datetime.now(timezone.utc))
        guild = ctx.trigger_message.guild
        results = []
        for uid, lines in matches.items():
            member = guild.get_member(uid)
            name = member.display_name if member else f"user_{uid}"
            results.append({"user": name, "user_id": str(uid), "matching_lines": lines})
        return ToolResponseRecord(tc.id, {"keyword": keyword, "results": results, "count": len(results)}, datetime.now(timezone.utc))

    async def _tool_profile(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        identifier = (tc.arguments.get("user_id_or_name") or "").strip()
        if not identifier:
            return ToolResponseRecord(tc.id, {"error": "user_id_or_name manquant"}, datetime.now(timezone.utc))
        target_id: Optional[int] = None
        if identifier.isdigit():
            target_id = int(identifier)
        elif ctx and ctx.trigger_message and ctx.trigger_message.guild:
            member = discord.utils.find(
                lambda m: m.name.lower() == identifier.lower() or m.display_name.lower() == identifier.lower(),
                ctx.trigger_message.guild.members,
            )
            if member:
                target_id = member.id
        if target_id is None:
            return ToolResponseRecord(tc.id, {"error": f"Membre '{identifier}' introuvable"}, datetime.now(timezone.utc))
        full = profiles.get_full(target_id)
        return ToolResponseRecord(tc.id, {"profile": full or "Aucune note."}, datetime.now(timezone.utc))

    return [
        Tool(
            name="update_user_notes",
            description=(
                "Enregistre une info sur un membre. "
                "Déclenche dès qu'un message révèle : prénom/âge/ville/métier, préférence forte, projet en cours, anecdote notable. "
                "Format addition : '[catégorie] info'. Ex : '[identité] Léa, 28 ans, graphiste' · '[préférences] végétarienne'. "
                "Si l'info concerne quelqu'un d'autre que l'auteur du message, passe son pseudo dans user_name."
            ),
            properties={
                "addition": {"type": "string", "description": "Info à noter (format: '[catégorie] info')"},
                "user_name": {"type": "string", "description": "Pseudo du membre concerné (si différent de l'auteur)"},
            },
            function=_tool_update_notes,
        ),
        Tool(
            name="search_user_notes",
            description=(
                "Cherche un mot-clé dans les notes mémorisées de tous les membres du serveur. "
                "Utile pour retrouver qui a une caractéristique précise (ex: 'végétarien', 'Lyon', 'Godot'). "
                "Retourne la liste des membres dont les notes contiennent le mot-clé, avec les lignes correspondantes."
            ),
            properties={
                "keyword": {"type": "string", "description": "Mot ou expression à chercher dans les notes"},
            },
            function=_tool_search_notes,
        ),
        Tool(
            name="get_user_profile",
            description="Consulte les notes mémorisées sur un membre. Utile pour vérifier ce qu'on sait déjà avant de noter ou de répondre.",
            properties={"user_id_or_name": {"type": "string", "description": "ID Discord ou pseudo du membre"}},
            function=_tool_profile,
        ),
    ]
