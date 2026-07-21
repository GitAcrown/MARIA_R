"""Outils LLM liés aux rappels."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import re

import discord

from common.discord_ui import layout_with_commentary
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.rappels import (
    RECURRENCE_MAX_DAYS,
    RECURRENCE_NONE,
    REPEAT_EMOJI,
    VALID_RECURRENCES,
    Rappel,
    RappelStore,
)
from common.timezones import PARIS_TZ

# Bornes métier des rappels.
REMINDER_MIN_MINUTES = 2
REMINDER_MAX_DAYS = 365  # horizon de planification
REMINDER_MAX_MINUTES = REMINDER_MAX_DAYS * 24 * 60
REMINDER_MAX_PENDING = 10

_RECURRENCE_LABEL = {
    "daily": "quotidien",
    "weekly": "hebdo",
}

# Préfixes méta que le modèle a tendance à coller (« Rappeler que… »).
_META_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"rappeler?\s+(?:que|qu'|de|d')|"
    r"rappelle[-\s]?moi\s+(?:que|qu'|de|d')|"
    r"n['']?oublie\s+pas\s+(?:de|d'|que|qu')|"
    r"ne\s+pas\s+oublier\s+(?:de|d'|que|qu')|"
    r"pense\s+[àa]\s+|"
    r"rappel\s*[:\-–—]\s*"
    r")\s*",
    re.IGNORECASE,
)


def sanitize_reminder_description(text: str) -> str:
    """Enlève les formulations méta (« Rappeler que… ») → contenu seul."""
    desc = (text or "").strip()
    if not desc:
        return desc
    # Plusieurs passes si le modèle empile les préfixes.
    for _ in range(3):
        cleaned = _META_PREFIX_RE.sub("", desc).strip(" \t-–—:.,")
        if cleaned == desc:
            break
        desc = cleaned
    # « c'est l'anniversaire de X » → « l'anniversaire de X »
    desc = re.sub(r"^c['']est\s+", "", desc, flags=re.IGNORECASE).strip()
    if desc:
        desc = desc[0].upper() + desc[1:]
    return desc[:200]


def _parse_execute_at(execute_at_str: str) -> datetime:
    """Parse une date ISO 8601 (fuseau Paris si naïf) en datetime UTC. Lève ValueError si invalide."""
    execute_at = datetime.fromisoformat(execute_at_str)
    if execute_at.tzinfo is None:
        execute_at = execute_at.replace(tzinfo=PARIS_TZ)
    return execute_at.astimezone(timezone.utc)


def _validate_horizon(execute_at: datetime) -> str | None:
    total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
    if total < REMINDER_MIN_MINUTES:
        return "Date trop proche (minimum 2 min)"
    if total > REMINDER_MAX_MINUTES:
        return f"Date trop lointaine (max {REMINDER_MAX_DAYS} jours)"
    return None


def _serialize_rappel(r: Rappel) -> dict:
    item = {
        "id": r.id,
        "description": r.description,
        "execute_at": r.execute_at.isoformat(),
        "execute_at_ts": int(r.execute_at.timestamp()),
        "recurrence": r.recurrence,
    }
    if r.recurrence != RECURRENCE_NONE and r.recurrence_until:
        item["recurrence_until"] = r.recurrence_until.isoformat()
        item["recurrence_until_ts"] = int(r.recurrence_until.timestamp())
    return item


def _format_widget_line(item: dict) -> str:
    ts = item["execute_at_ts"]
    desc = item["description"]
    rid = item["id"]
    if item.get("recurrence") and item["recurrence"] != RECURRENCE_NONE:
        label = _RECURRENCE_LABEL.get(item["recurrence"], item["recurrence"])
        until = ""
        if item.get("recurrence_until_ts"):
            until = f" · jusqu'au <t:{item['recurrence_until_ts']}:d>"
        return f"-# **#{rid}** {REPEAT_EMOJI} · <t:{ts}:f> · {label}{until}\n› {desc}"
    return f"-# **#{rid}** · <t:{ts}:f> (<t:{ts}:R>)\n› {desc}"


def build_reminders_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Widget lecture seule des rappels d'une personne."""
    if "error" in data or "display_name" not in data:
        return None
    name = data["display_name"]
    items = data.get("reminders") or []
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## Rappels · {name}"),
        discord.ui.Separator(),
    ]
    if not items:
        children.append(discord.ui.TextDisplay("-# Aucun rappel en attente."))
    else:
        body = "\n\n".join(_format_widget_line(it) for it in items[:15])
        if len(items) > 15:
            body += f"\n\n-# … et {len(items) - 15} de plus."
        children.append(discord.ui.TextDisplay(body))
    return layout_with_commentary(discord.ui.Container(*children), commentary)


async def _resolve_member(ctx, args: dict) -> tuple[Optional[discord.abc.User], Optional[str]]:
    """Résout un membre (user_id / username) ou l'auteur du message. (member, error)."""
    if not ctx or not ctx.trigger_message:
        return None, "Contexte manquant"
    author = ctx.trigger_message.author
    guild = ctx.trigger_message.guild
    uid_str = (args.get("user_id") or "").strip()
    name_q = (args.get("username") or "").strip().lower()
    if not uid_str and not name_q:
        return author, None
    if not guild:
        return None, "Cible autre membre : uniquement sur un serveur"
    member = None
    if uid_str:
        try:
            member = guild.get_member(int(uid_str))
            if not member:
                member = await guild.fetch_member(int(uid_str))
        except (ValueError, discord.NotFound, discord.HTTPException):
            pass
    if not member and name_q:
        member = discord.utils.find(
            lambda m: m.name.lower() == name_q or m.display_name.lower() == name_q,
            guild.members,
        )
        if not member:
            member = discord.utils.find(
                lambda m: name_q in m.name.lower() or name_q in m.display_name.lower(),
                guild.members,
            )
    if not member:
        return None, "Membre introuvable"
    return member, None


def build_reminder_tools(rappels: RappelStore) -> list[Tool]:
    """Construit les outils de planification / gestion des rappels."""

    async def _tool_schedule(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        desc = sanitize_reminder_description(args.get("task_description") or "")
        if not desc:
            return ToolResponseRecord(tc.id, {"error": "Description manquante"}, datetime.now(timezone.utc))

        execute_at_str = (args.get("execute_at") or "").strip()
        if execute_at_str:
            try:
                execute_at = _parse_execute_at(execute_at_str)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"},
                    datetime.now(timezone.utc),
                )
        else:
            total = (args.get("delay_minutes") or 0) + (args.get("delay_hours") or 0) * 60
            execute_at = datetime.now(timezone.utc) + timedelta(minutes=total)

        err = _validate_horizon(execute_at)
        if err:
            return ToolResponseRecord(tc.id, {"error": err}, datetime.now(timezone.utc))
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
        total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
        payload: dict = {
            "success": True,
            "task_id": rid,
            "execute_at": execute_at.isoformat(),
            "delay_minutes": total,
            "recurrence": recurrence,
            "_llm_summary": (
                f"Rappel #{rid} programmé"
                + (f" ({recurrence}, max {RECURRENCE_MAX_DAYS}j)" if recurrence != RECURRENCE_NONE else "")
                + "."
            ),
        }
        if recurrence != RECURRENCE_NONE:
            created = rappels.get(rid)
            if created and created.recurrence_until:
                payload["recurrence_until"] = created.recurrence_until.isoformat()
                payload["_llm_summary"] = (
                    f"Rappel #{rid} récurrent ({recurrence}) jusqu'au "
                    f"{created.recurrence_until.astimezone(PARIS_TZ).strftime('%d/%m/%Y')}."
                )
        return ToolResponseRecord(tc.id, payload, datetime.now(timezone.utc))

    async def _tool_edit(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        tid = args.get("task_id")
        if not tid:
            return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))

        new_desc = args.get("task_description")
        if isinstance(new_desc, str) and new_desc.strip():
            new_desc = sanitize_reminder_description(new_desc) or None
        else:
            new_desc = None

        execute_at = None
        execute_at_str = (args.get("execute_at") or "").strip()
        snooze_min = (args.get("snooze_minutes") or 0) + (args.get("snooze_hours") or 0) * 60

        if execute_at_str:
            try:
                execute_at = _parse_execute_at(execute_at_str)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"},
                    datetime.now(timezone.utc),
                )
        elif snooze_min > 0:
            execute_at = datetime.now(timezone.utc) + timedelta(minutes=snooze_min)

        if execute_at is not None:
            err = _validate_horizon(execute_at)
            if err:
                return ToolResponseRecord(tc.id, {"error": err}, datetime.now(timezone.utc))

        recurrence = args.get("recurrence")
        if isinstance(recurrence, str):
            recurrence = recurrence.strip().lower()
            if recurrence not in VALID_RECURRENCES:
                recurrence = None
        else:
            recurrence = None

        if new_desc is None and execute_at is None and recurrence is None:
            return ToolResponseRecord(
                tc.id,
                {"error": "Rien à modifier (fournis description, execute_at, snooze_minutes/hours ou recurrence)"},
                datetime.now(timezone.utc),
            )

        if execute_at is not None and snooze_min > 0 and not execute_at_str:
            new_at = rappels.snooze(int(tid), ctx.trigger_message.author.id, int(snooze_min))
            if new_at is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Rappel introuvable ou pas le tien"},
                    datetime.now(timezone.utc),
                )
            if new_desc or recurrence:
                rappels.edit(int(tid), ctx.trigger_message.author.id, description=new_desc, recurrence=recurrence)
        else:
            ok = rappels.edit(
                int(tid),
                ctx.trigger_message.author.id,
                description=new_desc,
                execute_at=execute_at,
                recurrence=recurrence,
            )
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Rappel introuvable, déjà passé, ou pas le tien"},
                    datetime.now(timezone.utc),
                )

        return ToolResponseRecord(
            tc.id,
            {"success": True, "task_id": int(tid), "_llm_summary": f"Rappel #{int(tid)} modifié."},
            datetime.now(timezone.utc),
        )

    async def _tool_list_reminders(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        reminders = rappels.get_user_rappels(ctx.trigger_message.author.id)
        if not reminders:
            return ToolResponseRecord(
                tc.id,
                {"reminders": [], "_llm_summary": "Aucun rappel en attente."},
                datetime.now(timezone.utc),
            )
        items = [_serialize_rappel(r) for r in reminders]
        return ToolResponseRecord(tc.id, {
            "reminders": items,
            "_llm_summary": f"{len(items)} rappel(s) en attente.",
        }, datetime.now(timezone.utc))

    async def _tool_show_reminders(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        member, err = await _resolve_member(ctx, tc.arguments or {})
        if err or member is None:
            return ToolResponseRecord(tc.id, {"error": err or "Membre introuvable"}, datetime.now(timezone.utc))
        pending = rappels.get_user_rappels(member.id)
        items = [_serialize_rappel(r) for r in pending]
        name = getattr(member, "display_name", None) or member.name
        return ToolResponseRecord(tc.id, {
            "_tool": "show_reminders",
            "user_id": str(member.id),
            "display_name": name,
            "count": len(items),
            "reminders": items,
            "_llm_summary": (
                f"Widget rappels de {name} affiché ({len(items)})."
                if items
                else f"Aucun rappel en attente pour {name}."
            ),
        }, datetime.now(timezone.utc))

    async def _tool_cancel(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        tid = tc.arguments.get("task_id")
        if not tid or not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))
        ok = rappels.cancel(int(tid), ctx.trigger_message.author.id)
        if not ok:
            return ToolResponseRecord(
                tc.id,
                {
                    "error": "Rappel introuvable, déjà passé/annulé, ou pas le tien. "
                    "Appelle list_reminders pour les IDs à jour (un rappel récurrent = un seul ID)."
                },
                datetime.now(timezone.utc),
            )
        return ToolResponseRecord(tc.id, {
            "success": True,
            "task_id": int(tid),
            "_llm_summary": f"Rappel #{int(tid)} annulé (y compris s'il était récurrent).",
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="schedule_reminder",
            description=(
                "Programme un rappel. Utilise execute_at (ISO 8601) pour une date absolue "
                "(ex. '2026-12-24T17:00:00' — fuseau Europe/Paris si naïf), "
                "ou delay_minutes/delay_hours pour un délai relatif. execute_at est prioritaire. "
                f"Horizon max : {REMINDER_MAX_DAYS} jours. "
                f"recurrence='daily' ou 'weekly' : répète pendant max {RECURRENCE_MAX_DAYS} jours "
                "puis s'arrête (par défaut 'none')."
            ),
            properties={
                "task_description": {
                    "type": "string",
                    "description": (
                        "Contenu du rappel tel qu'il s'affichera à l'heure H — le FAIT / l'événement seul. "
                        "INTERDIT : préfixer par « Rappeler que », « Rappelle-moi de », « N'oublie pas de », "
                        "« Rappel : ». Pas de référence temporelle relative ('demain', 'ce soir'). "
                        "Ex. OK : 'Anniversaire de Enzo', 'Appeler le médecin', 'Sortir les poubelles'. "
                        "Ex. KO : 'Rappeler que c'est l'anniversaire de Enzo', 'Ne pas oublier d'appeler'."
                    ),
                },
                "execute_at": {"type": "string", "description": "Date/heure absolue ISO 8601 (prioritaire sur les délais)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes (si pas de execute_at)"},
                "delay_hours": {"type": "integer", "description": "Délai en heures (si pas de execute_at)"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_RECURRENCES),
                    "description": (
                        f"Répétition : none (défaut), daily ou weekly "
                        f"(série limitée à {RECURRENCE_MAX_DAYS} jours)"
                    ),
                },
            },
            function=_tool_schedule,
        ),
        Tool(
            name="list_reminders",
            description=(
                "Liste JSON des rappels en attente de l'utilisateur qui parle "
                "(pour edit/cancel — pas d'affichage salon). "
                "Pour montrer les rappels dans le salon → show_reminders."
            ),
            properties={},
            function=_tool_list_reminders,
        ),
        Tool(
            name="show_reminders",
            description=(
                "Affiche un widget lecture seule des rappels en attente d'une personne dans le salon. "
                "Sans user_id/username → ceux de l'auteur. "
                "À utiliser pour « montre mes rappels », « les rappels de Bob », etc. "
                "Ne gère pas (pas d'annulation) — pour ça : /rappels ou cancel_reminder."
            ),
            properties={
                "user_id": {
                    "type": "string",
                    "description": "Id Discord du membre (optionnel, défaut = auteur)",
                },
                "username": {
                    "type": "string",
                    "description": "Nom / pseudo Discord du membre (optionnel)",
                },
            },
            optional_props=["user_id", "username"],
            function=_tool_show_reminders,
        ),
        Tool(
            name="edit_reminder",
            description=(
                "Modifie un rappel existant : description, date (execute_at ISO 8601), "
                "récurrence, ou report (snooze_minutes/hours). "
                "Ne renseigne que les champs à changer. list_reminders d'abord si l'ID est inconnu."
            ),
            properties={
                "task_id": {"type": "integer", "description": "ID du rappel à modifier"},
                "task_description": {"type": "string", "description": "Nouvelle description (optionnel)"},
                "execute_at": {"type": "string", "description": "Nouvelle date/heure ISO 8601, fuseau Paris par défaut (optionnel)"},
                "snooze_minutes": {"type": "integer", "description": "Reporter de N minutes à partir de maintenant (optionnel)"},
                "snooze_hours": {"type": "integer", "description": "Reporter de N heures à partir de maintenant (optionnel)"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_RECURRENCES),
                    "description": "Nouvelle récurrence : none, daily ou weekly (optionnel)",
                },
            },
            function=_tool_edit,
        ),
        Tool(
            name="cancel_reminder",
            description=(
                "Annule un rappel par son ID (y compris récurrent : stoppe toute la série). "
                "list_reminders d'abord si l'ID est inconnu. Ne recrée pas un rappel pour 'annuler'."
            ),
            properties={"task_id": {"type": "integer", "description": "ID du rappel"}},
            function=_tool_cancel,
        ),
    ]
