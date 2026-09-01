"""Outil LLM : création de sondages natifs Discord (discord.Poll)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import discord

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.polls import PollStore

POLL_MIN_OPTIONS = 2
POLL_MAX_OPTIONS = 10
POLL_MAX_HOURS = 168  # limite Discord : 7 jours
POLL_DEFAULT_HOURS = 24
POLL_QUESTION_MAX = 300
POLL_OPTION_MAX = 55


def build_poll_tools(store: PollStore) -> list[Tool]:
    async def _tool_create_poll(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        message = ctx.trigger_message
        args = tc.arguments or {}
        question = (args.get("question") or "").strip()[:POLL_QUESTION_MAX]
        options = [str(o).strip()[:POLL_OPTION_MAX] for o in (args.get("options") or []) if str(o).strip()]
        multiple = bool(args.get("multiple", False))

        if not question:
            return ToolResponseRecord(tc.id, {"error": "question vide"}, datetime.now(timezone.utc))
        if len(options) < POLL_MIN_OPTIONS:
            return ToolResponseRecord(
                tc.id, {"error": "Il faut au moins 2 options"}, datetime.now(timezone.utc),
            )
        options = options[:POLL_MAX_OPTIONS]

        try:
            hours = max(1, min(POLL_MAX_HOURS, int(args.get("duration_hours") or POLL_DEFAULT_HOURS)))
        except (TypeError, ValueError):
            hours = POLL_DEFAULT_HOURS

        poll = discord.Poll(question=question, duration=timedelta(hours=hours), multiple=multiple)
        for opt in options:
            poll.add_answer(text=opt)

        try:
            posted = await message.channel.send(poll=poll)
        except discord.HTTPException as e:
            return ToolResponseRecord(
                tc.id, {"error": f"Envoi du sondage échoué : {e}"}, datetime.now(timezone.utc),
            )

        expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        store.create(
            message_id=posted.id,
            channel_id=message.channel.id,
            guild_id=message.guild.id,
            author_id=message.author.id,
            question=question,
            options=options,
            expires_at=expires_at,
        )

        return ToolResponseRecord(tc.id, {
            "ok": True,
            "question": question,
            "options": options,
            "duration_hours": hours,
            "_llm_summary": (
                f"Sondage posté : « {question} » ({', '.join(options)}), clôture dans {hours}h. "
                "Rappel : tu ne votes jamais toi-même, et tu ne peux pas voir les votes en cours."
            ),
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="create_poll",
            description=(
                "Poste un sondage natif Discord (boutons de vote intégrés) pour trancher une "
                "question de groupe (« sonde qui vient samedi », « fais un sondage pizza ou "
                "sushi »). 2 à 10 options courtes. INTERDIT : voter toi-même, répondre au sondage "
                "à la place d'un membre, donner ton avis comme si ça comptait comme un vote — si "
                "on te le demande, dis que tu ne peux pas voter. Discord ne communique pas les "
                "votes en cours : si on demande où ça en est avant la clôture, dis de regarder le "
                "message (tu ne le sais pas non plus)."
            ),
            properties={
                "question": {"type": "string", "description": "La question posée, courte"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2 à 10 réponses possibles, courtes",
                },
                "multiple": {"type": "boolean", "description": "Autoriser plusieurs choix par votant (défaut: non)"},
                "duration_hours": {"type": "integer", "description": "Durée en heures (défaut 24, max 168)"},
            },
            optional_props=["multiple", "duration_hours"],
            function=_tool_create_poll,
        ),
    ]
