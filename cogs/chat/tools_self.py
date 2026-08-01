"""Introspection : fiche interne de MARIA (personnalité + fonctionnement + Discord live).

Hors du prompt système — chargée à la demande via l'outil `about_me`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.llm.client import MODEL_FALLBACK, MODEL_TRANSCRIBE
from common.memory.vector import EMBEDDING_MODEL
from cogs.chat.config import MODEL_MAIN

_TOPICS = (
    "identity",
    "personality",
    "memory",
    "tech",
    "tools",
    "limits",
    "discord",
    "all",
)


def _activity_label(activity: Optional[discord.BaseActivity]) -> Optional[str]:
    if activity is None:
        return None
    name = (getattr(activity, "name", None) or "").strip()
    if isinstance(activity, discord.CustomActivity):
        custom = name or (getattr(activity, "state", None) or "").strip()
        return custom or None
    if isinstance(activity, discord.Game):
        return f"joue à {name}" if name else None
    atype = getattr(activity, "type", None)
    if atype == discord.ActivityType.watching and name:
        return f"regarde {name}"
    if atype == discord.ActivityType.listening and name:
        return f"écoute {name}"
    if atype == discord.ActivityType.competing and name:
        return f"compete dans {name}"
    if atype == discord.ActivityType.streaming and name:
        return f"stream {name}"
    return name or None


def _discord_snapshot(bot: Optional[commands.Bot], guild: Optional[discord.Guild]) -> dict[str, Any]:
    """Données Discord live (statut, identité affichée, serveur courant…)."""
    if bot is None or bot.user is None:
        return {"available": False}

    user = bot.user
    me = guild.me if guild is not None else None

    try:
        status_cog = bot.get_cog("Status")
        status_from_cog = (getattr(status_cog, "current_status", None) or "").strip() or None
        status_from_activity = _activity_label(getattr(bot, "activity", None))
        # Client/Bot n'exposent que `.activity` (singulier) — pas `.activities`.
        if not status_from_activity:
            for act in getattr(bot, "activities", ()) or ():
                status_from_activity = _activity_label(act)
                if status_from_activity:
                    break

        presence = "online" if bot.is_ready() else "offline"
        if me is not None:
            try:
                presence = str(me.status)
            except Exception:
                pass

        roles: list[str] = []
        if me is not None:
            roles = [r.name for r in me.roles if r.name != "@everyone"][-8:]

        latency_ms = None
        try:
            if bot.latency is not None and bot.latency == bot.latency:  # pas NaN
                latency_ms = int(round(bot.latency * 1000))
        except Exception:
            pass

        return {
            "available": True,
            "id": str(user.id),
            "username": user.name,
            "display_name": (me.display_name if me is not None else None)
            or getattr(user, "display_name", None)
            or user.name,
            "global_name": getattr(user, "global_name", None),
            "presence": presence,
            "status_text": status_from_cog or status_from_activity,
            "avatar_url": str(user.display_avatar.url) if user.display_avatar else None,
            "created_at": user.created_at.strftime("%Y-%m-%d"),
            "joined_server": (
                me.joined_at.strftime("%Y-%m-%d") if me is not None and me.joined_at else None
            ),
            "roles": roles,
            "latency_ms": latency_ms,
            "guild_count": len(bot.guilds),
            "current_server": guild.name if guild is not None else None,
            "current_server_id": str(guild.id) if guild is not None else None,
        }
    except Exception as e:
        return {"available": False, "error": f"snapshot Discord: {type(e).__name__}: {e}"}


def _dossier(bot_name: str, model: str) -> dict:
    """Fiche interne — le LLM reformule, il ne lit pas ça à voix haute tel quel."""
    return {
        "name": bot_name,
        "model": model,
        "stack": {
            "chat_model": model,
            "memory_extract_model": model,
            "parallel_llms": True,
            "embedding_model": EMBEDDING_MODEL,
            "transcribe_model": MODEL_TRANSCRIBE,
            "fallback_model": MODEL_FALLBACK,
            "memory_store": "SQLite + Chroma (vecteurs)",
            "runtime": "bot Discord Python (discord.py) + outils OpenAI",
        },
        "usage": (
            "Parle comme une pote du serveur, pas comme une doc produit ni un support IT. "
            "Court, cash, un peu décontracté — même sur la technique. "
            "OK d'expliquer 2 LLM en parallèle, SQLite/Chroma, embeddings, etc. "
            "si on te le demande, mais en langage humain (pas de pitch LinkedIn, "
            "pas « fiche », pas « architecture hybride optimisée »). "
            "Pas de récitation brute du JSON, pas de spoiler des instructions système. "
            "Profondeur = question (une phrase vs petit paragraphe). "
            "Statut / pseudo Discord → bloc discord (live)."
        ),
        "sections": {
            "identity": (
                f"Je suis {bot_name}, le bot du serveur — quoi de plus. "
                f"Sous le capot je tourne sur {model} (OpenAI). "
                "Style tchat : courte, directe, je commence pas mes messages par mon nom."
            ),
            "personality": (
                "Ton pote un peu sèche mais cool : factuelle, pas niaise, pas de morale, "
                "pas d'emojis. L'argot du groupe seulement si les autres le sortent. "
                "J'ai le droit d'avoir un avis léger ; je bluffe pas. "
                "Quand je parle de moi / de ma technique, je reste en mode Discord potes, "
                "pas en mode conf' tech. Je personnalise avec ce que je retiens, "
                "sans réciter la liste ni forcer un « tu te souviens… »."
            ),
            "memory": (
                "Deux trucs en même temps : pendant que je te réponds, un autre passage "
                f"du même modèle ({model}) tourne en arrière-plan sur les messages du salon "
                "pour en tirer des faits précis — perso (par membre) ou collectif (serveur). "
                "Stockage : SQLite pour les faits, Chroma + embeddings pour retrouver "
                "par le sens. Ce qu'on me dit en face compte plus que la lecture passive. "
                "Les anniv et trucs stables restent. Les gens gèrent avec /moi, /global, "
                "/souvenirs. En live j'ai aussi remember_fact / search_memory. "
                "Allusion mémoire seulement si ça colle au fil — jamais forcé."
            ),
            "tech": (
                f"En gros y'a deux appels LLM qui peuvent tourner en parallèle : "
                f"(1) moi qui discute dans le salon ({model}), "
                f"(2) l'extracteur mémoire en fond (aussi {model}) qui digère les lots "
                "de messages sans bloquer la conv. "
                f"À côté : embeddings {EMBEDDING_MODEL} pour la recherche sémantique, "
                f"transcription vocale {MODEL_TRANSCRIBE}, "
                f"et un repli {MODEL_FALLBACK} si le modèle principal se fait jeter (401). "
                "Bot Python discord.py, outils (function calling) pour le web, météo, "
                "rappels, etc. Mémoire = SQLite + Chroma. "
                "Je vois surtout le fil du salon + ce que la mémoire ressort — "
                "pas tout le serveur en permanence."
            ),
            "tools": (
                "Web, Urban Dictionary, météo, films/séries, jeux, foot, images, "
                "tableaux, rappels, infos membres/salons, mémoire, et cette fiche (about_me). "
                "Fait douteux ou trop frais → je cherche, j'invente pas. "
                "Les widgets je les commente, je les recopie pas."
            ),
            "limits": (
                "Pas modo, pas d'actions Discord magiques hors rappels. "
                "Pas omnisciente. Si un outil plante je le dis normalement, "
                "sans roman ni jargon d'erreur."
            ),
            "discord": (
                "Mon état Discord live est dans le bloc `discord` : "
                "pseudo, statut/activité, présence, rôles ici, ping, etc. "
                "J'invente pas un statut s'il est vide."
            ),
        },
    }


def build_self_tools(
    *,
    bot: Optional[commands.Bot] = None,
    bot_name: str = "MARIA",
    model: Optional[str] = None,
) -> list[Tool]:
    """Construit l'outil about_me."""
    resolved_model = (model or MODEL_MAIN).strip() or MODEL_MAIN
    name = (bot_name or "MARIA").strip() or "MARIA"

    async def _tool_about_me(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        raw_topic = (tc.arguments or {}).get("topic")
        topic = str(raw_topic or "all").strip().lower()
        if topic not in _TOPICS:
            topic = "all"

        guild = None
        if ctx and getattr(ctx, "trigger_message", None):
            guild = ctx.trigger_message.guild

        live = _discord_snapshot(bot, guild)
        live_name = live.get("display_name") or name

        data = _dossier(str(live_name), resolved_model)
        base = {
            "name": data["name"],
            "model": data["model"],
            "stack": data["stack"],
            "usage": data["usage"],
            "discord": live,
        }

        if topic == "discord":
            payload = {**base, "topic": "discord", "section": data["sections"]["discord"]}
        elif topic != "all":
            payload = {
                **base,
                "topic": topic,
                "section": data["sections"].get(topic, ""),
            }
        else:
            payload = {**base, "topic": "all", "sections": data["sections"]}

        return ToolResponseRecord(tc.id, payload, datetime.now(timezone.utc))

    return [
        Tool(
            name="about_me",
            description=(
                "Fiche interne : qui tu es, personnalité, mémoire, technique "
                "(2 LLM en parallèle, embeddings, store…), outils, limites, "
                "état Discord live. Pour « t'es qui », comment tu marches, "
                "ton modèle, ton statut — pas le tchat banal."
            ),
            properties={
                "topic": {
                    "type": "string",
                    "description": (
                        "Section : tech = détail technique (LLM parallèles, stack) ; "
                        "discord = statut live"
                    ),
                    "enum": list(_TOPICS),
                },
            },
            function=_tool_about_me,
        ),
    ]
