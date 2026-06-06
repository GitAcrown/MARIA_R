"""Outils LLM liés aux rappels."""

from datetime import datetime, timedelta, timezone

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.rappels import RappelStore
from common.timezones import PARIS_TZ

# Bornes métier des rappels.
REMINDER_MIN_MINUTES = 2
REMINDER_MAX_MINUTES = 43200  # 30 jours
REMINDER_MAX_PENDING = 10


def build_reminder_tools(rappels: RappelStore) -> list[Tool]:
    """Construit les outils de planification / gestion des rappels."""

    async def _tool_schedule(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        desc = (args.get("task_description") or "").strip()
        if not desc:
            return ToolResponseRecord(tc.id, {"error": "Description manquante"}, datetime.now(timezone.utc))

        execute_at_str = (args.get("execute_at") or "").strip()
        if execute_at_str:
            try:
                execute_at = datetime.fromisoformat(execute_at_str)
                if execute_at.tzinfo is None:
                    execute_at = execute_at.replace(tzinfo=PARIS_TZ)
                execute_at = execute_at.astimezone(timezone.utc)
            except ValueError:
                return ToolResponseRecord(tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"}, datetime.now(timezone.utc))
        else:
            total = (args.get("delay_minutes") or 0) + (args.get("delay_hours") or 0) * 60
            execute_at = datetime.now(timezone.utc) + timedelta(minutes=total)

        total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
        if total < REMINDER_MIN_MINUTES:
            return ToolResponseRecord(tc.id, {"error": "Date trop proche (minimum 2 min)"}, datetime.now(timezone.utc))
        if total > REMINDER_MAX_MINUTES:
            return ToolResponseRecord(tc.id, {"error": "Date trop lointaine (max 30 jours)"}, datetime.now(timezone.utc))
        if rappels.count_pending(ctx.trigger_message.author.id) >= REMINDER_MAX_PENDING:
            return ToolResponseRecord(tc.id, {"error": "Max 10 rappels en attente"}, datetime.now(timezone.utc))

        rid = rappels.add(
            ctx.trigger_message.channel.id,
            ctx.trigger_message.author.id,
            desc,
            execute_at,
            ctx.trigger_message.id,
        )
        return ToolResponseRecord(tc.id, {
            "success": True, "task_id": rid,
            "execute_at": execute_at.isoformat(), "delay_minutes": total,
        }, datetime.now(timezone.utc))

    async def _tool_list_reminders(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        reminders = rappels.get_user_rappels(ctx.trigger_message.author.id)
        if not reminders:
            return ToolResponseRecord(tc.id, {"reminders": []}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "reminders": [
                {"id": r.id, "description": r.description, "execute_at": r.execute_at.isoformat()}
                for r in reminders
            ]
        }, datetime.now(timezone.utc))

    async def _tool_cancel(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        tid = tc.arguments.get("task_id")
        if not tid or not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))
        ok = rappels.cancel(int(tid), ctx.trigger_message.author.id)
        return ToolResponseRecord(tc.id, {"success": ok}, datetime.now(timezone.utc))

    return [
        Tool(
            name="schedule_reminder",
            description=(
                "Programme un rappel. Utilise execute_at (ISO 8601) pour une date absolue "
                "(ex. '2026-03-24T17:00:00' pour demain 17h — le fuseau par défaut est Europe/Paris), "
                "ou delay_minutes/delay_hours pour un délai relatif. execute_at est prioritaire."
            ),
            properties={
                "task_description": {"type": "string", "description": "Description de la tâche"},
                "execute_at": {"type": "string", "description": "Date/heure absolue ISO 8601 (prioritaire sur les délais)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes (si pas de execute_at)"},
                "delay_hours": {"type": "integer", "description": "Délai en heures (si pas de execute_at)"},
            },
            function=_tool_schedule,
        ),
        Tool(
            name="list_reminders",
            description="Liste les rappels en attente de l'utilisateur. À appeler avant cancel_reminder pour obtenir les IDs.",
            properties={},
            function=_tool_list_reminders,
        ),
        Tool(
            name="cancel_reminder",
            description="Annule un rappel par son ID. Appelle list_reminders d'abord si tu n'as pas l'ID.",
            properties={"task_id": {"type": "integer", "description": "ID du rappel"}},
            function=_tool_cancel,
        ),
    ]
