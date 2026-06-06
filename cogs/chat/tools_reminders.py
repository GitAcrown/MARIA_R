"""Outils LLM liés aux rappels."""

from datetime import datetime, timedelta, timezone

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.rappels import (
    RECURRENCE_DAILY,
    RECURRENCE_NONE,
    RECURRENCE_WEEKLY,
    VALID_RECURRENCES,
    RappelStore,
)
from common.timezones import PARIS_TZ

# Bornes métier des rappels.
REMINDER_MIN_MINUTES = 2
REMINDER_MAX_MINUTES = 43200  # 30 jours
REMINDER_MAX_PENDING = 10


def _parse_execute_at(execute_at_str: str) -> datetime:
    """Parse une date ISO 8601 (fuseau Paris si naïf) en datetime UTC. Lève ValueError si invalide."""
    execute_at = datetime.fromisoformat(execute_at_str)
    if execute_at.tzinfo is None:
        execute_at = execute_at.replace(tzinfo=PARIS_TZ)
    return execute_at.astimezone(timezone.utc)


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
                execute_at = _parse_execute_at(execute_at_str)
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

        recurrence = (args.get("recurrence") or RECURRENCE_NONE).strip().lower()
        if recurrence not in VALID_RECURRENCES:
            recurrence = RECURRENCE_NONE

        rid = rappels.add(
            ctx.trigger_message.channel.id,
            ctx.trigger_message.author.id,
            desc,
            execute_at,
            ctx.trigger_message.id,
            recurrence=recurrence,
        )
        return ToolResponseRecord(tc.id, {
            "success": True, "task_id": rid,
            "execute_at": execute_at.isoformat(), "delay_minutes": total,
            "recurrence": recurrence,
        }, datetime.now(timezone.utc))

    async def _tool_edit(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        tid = args.get("task_id")
        if not tid:
            return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))

        new_desc = args.get("task_description")
        new_desc = new_desc.strip() if isinstance(new_desc, str) and new_desc.strip() else None

        execute_at = None
        execute_at_str = (args.get("execute_at") or "").strip()
        if execute_at_str:
            try:
                execute_at = _parse_execute_at(execute_at_str)
            except ValueError:
                return ToolResponseRecord(tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"}, datetime.now(timezone.utc))
            total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
            if total < REMINDER_MIN_MINUTES:
                return ToolResponseRecord(tc.id, {"error": "Date trop proche (minimum 2 min)"}, datetime.now(timezone.utc))
            if total > REMINDER_MAX_MINUTES:
                return ToolResponseRecord(tc.id, {"error": "Date trop lointaine (max 30 jours)"}, datetime.now(timezone.utc))

        recurrence = args.get("recurrence")
        if isinstance(recurrence, str):
            recurrence = recurrence.strip().lower()
            if recurrence not in VALID_RECURRENCES:
                recurrence = None
        else:
            recurrence = None

        if new_desc is None and execute_at is None and recurrence is None:
            return ToolResponseRecord(tc.id, {"error": "Rien à modifier (fournis description, execute_at ou recurrence)"}, datetime.now(timezone.utc))

        ok = rappels.edit(
            int(tid),
            ctx.trigger_message.author.id,
            description=new_desc,
            execute_at=execute_at,
            recurrence=recurrence,
        )
        if not ok:
            return ToolResponseRecord(tc.id, {"error": "Rappel introuvable, déjà passé, ou pas le tien"}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {"success": True, "task_id": int(tid)}, datetime.now(timezone.utc))

    async def _tool_snooze(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        tid = args.get("task_id")
        if not tid:
            return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))
        minutes = (args.get("minutes") or 0) + (args.get("hours") or 0) * 60
        if minutes < 1:
            return ToolResponseRecord(tc.id, {"error": "Durée de report invalide"}, datetime.now(timezone.utc))
        if minutes > REMINDER_MAX_MINUTES:
            return ToolResponseRecord(tc.id, {"error": "Report trop lointain (max 30 jours)"}, datetime.now(timezone.utc))
        new_at = rappels.snooze(int(tid), ctx.trigger_message.author.id, int(minutes))
        if new_at is None:
            return ToolResponseRecord(tc.id, {"error": "Rappel introuvable ou pas le tien"}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "success": True, "task_id": int(tid), "execute_at": new_at.isoformat(),
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
                "ou delay_minutes/delay_hours pour un délai relatif. execute_at est prioritaire. "
                "recurrence='daily' ou 'weekly' pour un rappel répété (par défaut 'none')."
            ),
            properties={
                "task_description": {"type": "string", "description": "Description de la tâche"},
                "execute_at": {"type": "string", "description": "Date/heure absolue ISO 8601 (prioritaire sur les délais)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes (si pas de execute_at)"},
                "delay_hours": {"type": "integer", "description": "Délai en heures (si pas de execute_at)"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_RECURRENCES),
                    "description": "Répétition du rappel : none (défaut), daily ou weekly",
                },
            },
            function=_tool_schedule,
        ),
        Tool(
            name="list_reminders",
            description="Liste les rappels en attente de l'utilisateur. À appeler avant edit/snooze/cancel pour obtenir les IDs.",
            properties={},
            function=_tool_list_reminders,
        ),
        Tool(
            name="edit_reminder",
            description=(
                "Modifie un rappel existant de l'utilisateur (sa description, sa date execute_at ISO 8601, "
                "et/ou sa récurrence). Appelle list_reminders d'abord si tu n'as pas l'ID. "
                "Ne renseigne que les champs à changer."
            ),
            properties={
                "task_id": {"type": "integer", "description": "ID du rappel à modifier"},
                "task_description": {"type": "string", "description": "Nouvelle description (optionnel)"},
                "execute_at": {"type": "string", "description": "Nouvelle date/heure ISO 8601, fuseau Paris par défaut (optionnel)"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_RECURRENCES),
                    "description": "Nouvelle récurrence : none, daily ou weekly (optionnel)",
                },
            },
            function=_tool_edit,
        ),
        Tool(
            name="snooze_reminder",
            description=(
                "Reporte un rappel de l'utilisateur d'une durée à partir de maintenant "
                "(ex. « repousse de 10 min », « plus tard »). Appelle list_reminders d'abord si besoin de l'ID."
            ),
            properties={
                "task_id": {"type": "integer", "description": "ID du rappel à reporter"},
                "minutes": {"type": "integer", "description": "Minutes de report"},
                "hours": {"type": "integer", "description": "Heures de report (cumulées avec minutes)"},
            },
            function=_tool_snooze,
        ),
        Tool(
            name="cancel_reminder",
            description="Annule un rappel par son ID. Appelle list_reminders d'abord si tu n'as pas l'ID.",
            properties={"task_id": {"type": "integer", "description": "ID du rappel"}},
            function=_tool_cancel,
        ),
    ]
