"""Outils LLM liés aux tâches planifiées."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import discord

from common.discord_ui import layout_with_commentary
from common.emojis import REPEAT_REMINDER, SMALL_TASK
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.tasks import (
    SCHEDULE_ONCE,
    STATUS_PAUSED,
    STATUS_PENDING,
    TASK_INSTRUCTION_MAX,
    TASK_MAX_DAYS,
    TASK_MAX_PENDING,
    TASK_MAX_RECURRING,
    TASK_MIN_MINUTES,
    VALID_SCHEDULES,
    ScheduledTask,
    TaskStore,
    format_schedule,
    normalize_weekdays,
    snap_execute_at,
)
from common.timezones import PARIS_TZ

TASK_MAX_MINUTES = TASK_MAX_DAYS * 24 * 60

def sanitize_task_instruction(text: str) -> str:
    """Normalise la consigne (trim + plafond), sans retirer « Rappelle… »."""
    return (text or "").strip()[:TASK_INSTRUCTION_MAX]


def _parse_execute_at(execute_at_str: str) -> datetime:
    execute_at = datetime.fromisoformat(execute_at_str)
    if execute_at.tzinfo is None:
        execute_at = execute_at.replace(tzinfo=PARIS_TZ)
    return execute_at.astimezone(timezone.utc)


def _validate_horizon(execute_at: datetime, *, require_min: bool = True) -> str | None:
    total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
    if require_min and total < TASK_MIN_MINUTES:
        return "Date trop proche (minimum 2 min)"
    if total > TASK_MAX_MINUTES:
        return f"Date trop lointaine (max {TASK_MAX_DAYS} jours)"
    return None


def _serialize_task(t: ScheduledTask) -> dict:
    item = {
        "id": t.id,
        "title": t.title or t.instruction,
        "instruction": t.instruction,
        "execute_at": t.execute_at.isoformat(),
        "execute_at_ts": int(t.execute_at.timestamp()),
        "schedule_kind": t.schedule_kind,
        "weekdays": t.weekdays,
        "time_of_day": t.time_of_day,
        "status": t.status,
        "schedule_label": format_schedule(t),
        "deliver_dm": bool(t.deliver_dm),
    }
    if t.until_at:
        item["until_at"] = t.until_at.isoformat()
        item["until_at_ts"] = int(t.until_at.timestamp())
    if t.last_error:
        item["last_error"] = t.last_error
    return item


def _format_widget_line(item: dict) -> str:
    ts = item["execute_at_ts"]
    desc = item["instruction"]
    kind = item.get("schedule_kind") or SCHEDULE_ONCE
    status = item.get("status") or STATUS_PENDING
    status_bit = " · en pause" if status == STATUS_PAUSED else ""
    if kind != SCHEDULE_ONCE:
        until = ""
        if item.get("until_at_ts"):
            until = f" · jusqu'au <t:{item['until_at_ts']}:d>"
        dest = " · MP" if item.get("deliver_dm") else ""
        return (
            f"-# {REPEAT_REMINDER} · <t:{ts}:f> · "
            f"{item.get('schedule_label', kind)}{until}{dest}{status_bit}\n› {desc}"
        )
    dest = " · MP" if item.get("deliver_dm") else ""
    return f"-# <t:{ts}:f> (<t:{ts}:R>){dest}{status_bit}\n› {desc}"


async def _send_dm_confirm(
    user: discord.abc.User,
    *,
    label: str,
    instruction: str,
    execute_at: datetime,
) -> str | None:
    """Confirme en MP. None si OK, sinon message d'erreur (MP fermés, etc.)."""
    ts = int(execute_at.timestamp())
    desc = (instruction or "").strip()
    if len(desc) > 120:
        desc = desc[:119] + "…"
    body = (
        f"{SMALL_TASK} **Tâche confirmée** — {desc}\n"
        f"-# {label} · <t:{ts}:f> (<t:{ts}:R>)"
    )
    try:
        await user.send(body, allowed_mentions=discord.AllowedMentions.none())
        return None
    except (discord.Forbidden, discord.HTTPException):
        return (
            "Impossible d'envoyer en MP (MP fermés ou bot bloqué). "
            "Tâche annulée. Ouvre tes MP avec moi, ou programme-la dans le salon."
        )


def build_tasks_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if "error" in data or "display_name" not in data:
        return None
    name = data["display_name"]
    items = data.get("tasks") or []
    quota_n = sum(
        1 for it in items
        if (it.get("status") or STATUS_PENDING) in (STATUS_PENDING, STATUS_PAUSED)
    )
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## Tâches · {name} · {quota_n}/{TASK_MAX_PENDING}"),
        discord.ui.Separator(),
    ]
    if not items:
        children.append(discord.ui.TextDisplay("-# Aucune tâche en attente."))
    else:
        body = "\n\n".join(_format_widget_line(it) for it in items)
        children.append(discord.ui.TextDisplay(body))
    return layout_with_commentary(discord.ui.Container(*children), commentary)


async def _resolve_member(ctx, args: dict) -> tuple[Optional[discord.abc.User], Optional[str]]:
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


def build_task_tools(store: TaskStore) -> list[Tool]:
    """Construit les outils de planification / gestion des tâches."""

    async def _tool_schedule(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments
        instruction = sanitize_task_instruction(args.get("instruction") or "")
        if not instruction:
            return ToolResponseRecord(tc.id, {"error": "Instruction manquante"}, datetime.now(timezone.utc))

        kind = (args.get("recurrence") or SCHEDULE_ONCE).strip().lower()
        if kind not in VALID_SCHEDULES:
            kind = SCHEDULE_ONCE
        days = normalize_weekdays(args.get("weekdays") or "")
        time_of_day = (args.get("time") or "").strip()

        execute_at_str = (args.get("execute_at") or "").strip()
        execute_at = None
        if execute_at_str:
            try:
                execute_at = _parse_execute_at(execute_at_str)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"},
                    datetime.now(timezone.utc),
                )
        elif kind == SCHEDULE_ONCE:
            total = (args.get("delay_minutes") or 0) + (args.get("delay_hours") or 0) * 60
            execute_at = datetime.now(timezone.utc) + timedelta(minutes=max(total, 0))

        until_at = None
        until_str = (args.get("until") or "").strip()
        if until_str:
            try:
                until_at = _parse_execute_at(until_str)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "Format until invalide (ISO 8601 attendu)"},
                    datetime.now(timezone.utc),
                )

        if kind != SCHEDULE_ONCE:
            if not time_of_day and execute_at is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Heure requise (time=HH:MM) pour une tâche répétitive."},
                    datetime.now(timezone.utc),
                )
            execute_at = snap_execute_at(
                kind=kind,
                weekdays=days,
                time_of_day=time_of_day,
                execute_at=execute_at,
                until_at=until_at,
            )
            if execute_at is None:
                return ToolResponseRecord(
                    tc.id,
                    {"error": "Aucune occurrence à venir (date de fin trop tôt, ou aucun jour valide)."},
                    datetime.now(timezone.utc),
                )
        elif execute_at is None:
            return ToolResponseRecord(
                tc.id, {"error": "Date manquante (execute_at ou delay)."},
                datetime.now(timezone.utc),
            )

        err = _validate_horizon(execute_at, require_min=(kind == SCHEDULE_ONCE))
        if err:
            return ToolResponseRecord(tc.id, {"error": err}, datetime.now(timezone.utc))
        if store.count_active(ctx.trigger_message.author.id) >= TASK_MAX_PENDING:
            return ToolResponseRecord(
                tc.id, {"error": f"Limite atteinte ({TASK_MAX_PENDING} tâches). Annule-en une d'abord."},
                datetime.now(timezone.utc),
            )
        if kind != SCHEDULE_ONCE:
            n_rep = store.count_active_recurring(ctx.trigger_message.author.id)
            if n_rep >= TASK_MAX_RECURRING:
                return ToolResponseRecord(
                    tc.id,
                    {"error": (
                        f"Limite atteinte ({TASK_MAX_RECURRING} tâches répétitives). "
                        "Passe-en une en unique ou annule-en une."
                    )},
                    datetime.now(timezone.utc),
                )

        via = (args.get("via") or "").strip().lower()
        if not via:
            via = "dm" if ctx.trigger_message.guild is None else "channel"
        deliver_dm = via in ("dm", "mp", "private")

        title = (args.get("title") or "").strip()
        guild = ctx.trigger_message.guild
        tid = store.add(
            channel_id=ctx.trigger_message.channel.id,
            user_id=ctx.trigger_message.author.id,
            guild_id=guild.id if guild else 0,
            instruction=instruction,
            execute_at=execute_at,
            title=title,
            schedule_kind=kind,
            weekdays=days,
            time_of_day=time_of_day,
            until_at=until_at,
            message_id=ctx.trigger_message.id,
            deliver_dm=deliver_dm,
        )
        created = store.get(tid)
        label = format_schedule(created) if created else kind
        dest = "MP" if deliver_dm else "salon"
        if deliver_dm:
            dm_err = await _send_dm_confirm(
                ctx.trigger_message.author,
                label=label,
                instruction=instruction,
                execute_at=execute_at,
            )
            if dm_err:
                store.cancel(tid, ctx.trigger_message.author.id)
                return ToolResponseRecord(
                    tc.id, {"error": dm_err}, datetime.now(timezone.utc),
                )
        return ToolResponseRecord(tc.id, {
            "success": True,
            "task_id": tid,
            "execute_at": execute_at.isoformat(),
            "schedule": label,
            "via": dest,
            "_llm_summary": f"Tâche programmée ({label}, {dest}).",
        }, datetime.now(timezone.utc))

    async def _tool_manage(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments or {}
        action = (args.get("action") or "list").strip().lower()
        user_id = ctx.trigger_message.author.id

        if action == "list":
            tasks = store.get_user_tasks(user_id)
            items = [_serialize_task(t) for t in tasks]
            return ToolResponseRecord(tc.id, {
                "tasks": items,
                "_llm_summary": f"{len(items)} tâche(s) active(s)." if items else "Aucune tâche en attente.",
            }, datetime.now(timezone.utc))

        tid = args.get("task_id")
        if not tid:
            return ToolResponseRecord(
                tc.id,
                {"error": "task_id manquant. Appelle manage_task action=list pour les IDs."},
                datetime.now(timezone.utc),
            )
        tid = int(tid)

        if action == "cancel":
            ok = store.cancel(tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Tâche introuvable ou pas la tienne."},
                    datetime.now(timezone.utc),
                )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche annulée.",
            }, datetime.now(timezone.utc))

        if action == "pause":
            ok = store.pause(tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Impossible de mettre en pause (introuvable, déjà en pause, ou pas la tienne)."},
                    datetime.now(timezone.utc),
                )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche en pause.",
            }, datetime.now(timezone.utc))

        if action == "resume":
            ok = store.resume(tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Impossible de reprendre (pas en pause, ou pas la tienne)."},
                    datetime.now(timezone.utc),
                )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche reprise.",
            }, datetime.now(timezone.utc))

        if action == "skip":
            nxt = store.skip_next(tid, user_id)
            if nxt is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Pas de prochaine occurrence (tâche unique, ou introuvable)."},
                    datetime.now(timezone.utc),
                )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid, "execute_at": nxt.isoformat(),
                "_llm_summary": "Prochaine occurrence sautée.",
            }, datetime.now(timezone.utc))

        if action == "edit":
            new_instr = args.get("instruction")
            if isinstance(new_instr, str) and new_instr.strip():
                new_instr = sanitize_task_instruction(new_instr) or None
            else:
                new_instr = None
            execute_at = None
            execute_at_str = (args.get("execute_at") or "").strip()
            if execute_at_str:
                try:
                    execute_at = _parse_execute_at(execute_at_str)
                except ValueError:
                    return ToolResponseRecord(
                        tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"},
                        datetime.now(timezone.utc),
                    )
            kind = args.get("recurrence")
            if isinstance(kind, str):
                kind = kind.strip().lower()
                if kind not in VALID_SCHEDULES:
                    kind = None
            else:
                kind = None
            current = store.get(tid)
            rec_kind = kind or (current.schedule_kind if current else SCHEDULE_ONCE)
            if execute_at is not None and rec_kind == SCHEDULE_ONCE:
                err = _validate_horizon(execute_at)
                if err:
                    return ToolResponseRecord(tc.id, {"error": err}, datetime.now(timezone.utc))
            if kind in (SCHEDULE_DAILY, SCHEDULE_WEEKLY):
                already = bool(current and current.schedule_kind != SCHEDULE_ONCE)
                n_rep = store.count_active_recurring(user_id, exclude_id=tid)
                if not already and n_rep >= TASK_MAX_RECURRING:
                    return ToolResponseRecord(
                        tc.id,
                        {"error": (
                            f"Limite atteinte ({TASK_MAX_RECURRING} tâches répétitives). "
                            "Passe-en une en unique ou annule-en une."
                        )},
                        datetime.now(timezone.utc),
                    )
            days_raw = args.get("weekdays")
            days = normalize_weekdays(days_raw) if days_raw else None
            time_of_day = args.get("time")
            if isinstance(time_of_day, str):
                time_of_day = time_of_day.strip() or None
            else:
                time_of_day = None
            until_at = None
            until_str = (args.get("until") or "").strip()
            if until_str:
                try:
                    until_at = _parse_execute_at(until_str)
                except ValueError:
                    return ToolResponseRecord(
                        tc.id, {"error": "Format until invalide"},
                        datetime.now(timezone.utc),
                    )
            via_raw = args.get("via")
            deliver_dm = None
            if isinstance(via_raw, str) and via_raw.strip():
                v = via_raw.strip().lower()
                if v in ("dm", "mp", "private"):
                    deliver_dm = True
                elif v in ("channel", "salon"):
                    deliver_dm = False
            if deliver_dm is True and not (current and current.deliver_dm):
                preview = current
                dm_err = await _send_dm_confirm(
                    ctx.trigger_message.author,
                    label=format_schedule(preview) if preview else "MP",
                    instruction=(new_instr or (preview.instruction if preview else "")),
                    execute_at=execute_at or (preview.execute_at if preview else datetime.now(timezone.utc)),
                )
                if dm_err:
                    return ToolResponseRecord(
                        tc.id,
                        {"error": dm_err.replace("Tâche annulée.", "Passage en MP annulé.")},
                        datetime.now(timezone.utc),
                    )
            ok = store.edit(
                tid, user_id,
                instruction=new_instr,
                execute_at=execute_at,
                schedule_kind=kind,
                weekdays=days,
                time_of_day=time_of_day,
                until_at=until_at,
                deliver_dm=deliver_dm,
            )
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Tâche introuvable, déjà passée, ou pas la tienne."},
                    datetime.now(timezone.utc),
                )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche modifiée.",
            }, datetime.now(timezone.utc))

        return ToolResponseRecord(
            tc.id,
            {"error": "action inconnue (list|edit|pause|resume|skip|cancel)"},
            datetime.now(timezone.utc),
        )

    async def _tool_show(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        member, err = await _resolve_member(ctx, tc.arguments or {})
        if err or member is None:
            return ToolResponseRecord(tc.id, {"error": err or "Membre introuvable"}, datetime.now(timezone.utc))
        pending = store.get_user_tasks(member.id)
        items = [_serialize_task(t) for t in pending]
        name = getattr(member, "display_name", None) or member.name
        return ToolResponseRecord(tc.id, {
            "_tool": "show_tasks",
            "user_id": str(member.id),
            "display_name": name,
            "count": len(items),
            "tasks": items,
            "_llm_summary": (
                f"Widget tâches de {name} affiché ({len(items)})."
                if items else f"Aucune tâche en attente pour {name}."
            ),
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="schedule_task",
            description=(
                "Programme une tâche que tu exécuteras à l'heure H "
                "(rappel, météo, recherche…). "
                f"Max {TASK_MAX_PENDING} tâches par personne, dont {TASK_MAX_RECURRING} répétitives (daily/weekly). "
                f"execute_at ISO 8601 (Paris si naïf) prioritaire, sinon delay_minutes/hours. "
                f"Max {TASK_MAX_DAYS}j. recurrence once|daily|weekly ; "
                "weekly : weekdays=mon,tue,wed,thu,fri et time=HH:MM. until optionnel. "
                "Heure déjà passée → prochaine occ. (ne refuse pas). "
                "via=dm UNIQUEMENT si la personne dit clairement MP, DM ou message privé. "
                "Interdit de déduire via=dm de « donne-moi », « envoie-moi » ou d'un briefing perso. "
                "Défaut : channel (salon)."
            ),
            properties={
                "instruction": {
                    "type": "string",
                    "description": (
                        "Consigne à EXÉCUTER à l'heure H, pas un méta-rappel. "
                        "OK : « Rappelle d'aller à la salle et donne la météo à Paris ». "
                        "KO : « Rappeler que… », « demain/ce soir »."
                    ),
                },
                "title": {"type": "string", "description": "Libellé court UI (optionnel)"},
                "execute_at": {"type": "string", "description": "Date/heure ISO 8601 (prioritaire)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes"},
                "delay_hours": {"type": "integer", "description": "Délai en heures"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_SCHEDULES),
                    "description": "once|daily|weekly",
                },
                "weekdays": {
                    "type": "string",
                    "description": "Jours weekly, virgules : mon,tue,wed,thu,fri,sat,sun (ex. wed,fri)",
                },
                "time": {"type": "string", "description": "Heure Paris HH:MM pour daily/weekly"},
                "until": {"type": "string", "description": "Fin de série ISO 8601 (optionnel)"},
                "via": {
                    "type": "string",
                    "enum": ["channel", "dm"],
                    "description": "channel = salon (défaut). dm = MP, seulement si demandé explicitement (MP/DM/message privé).",
                },
            },
            optional_props=["title", "execute_at", "delay_minutes", "delay_hours", "recurrence", "weekdays", "time", "until", "via"],
            function=_tool_schedule,
        ),
        Tool(
            name="manage_task",
            description=(
                "Gère tes tâches : list (IDs), edit, pause, resume, skip (saute la prochaine), cancel. "
                "list si l'ID est inconnu."
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": ["list", "edit", "pause", "resume", "skip", "cancel"],
                    "description": "Action à faire",
                },
                "task_id": {"type": "integer", "description": "ID (sauf list)"},
                "instruction": {"type": "string", "description": "Nouvelle consigne (edit)"},
                "execute_at": {"type": "string", "description": "Nouvelle date ISO 8601 (edit)"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_SCHEDULES),
                    "description": "once|daily|weekly (edit)",
                },
                "weekdays": {"type": "string", "description": "Jours weekly, virgules (edit)"},
                "time": {"type": "string", "description": "HH:MM Paris (edit)"},
                "until": {"type": "string", "description": "Fin de série ISO 8601 (edit)"},
                "via": {
                    "type": "string",
                    "enum": ["channel", "dm"],
                    "description": "dm seulement si demandé explicitement (MP/DM/message privé)",
                },
            },
            optional_props=["task_id", "instruction", "execute_at", "recurrence", "weekdays", "time", "until", "via"],
            function=_tool_manage,
        ),
        Tool(
            name="show_tasks",
            description=(
                "Widget lecture seule des tâches d'une personne dans le salon "
                "(défaut = auteur). Gestion → /taches ou manage_task."
            ),
            properties={
                "user_id": {"type": "string", "description": "Id Discord (optionnel, défaut = auteur)"},
                "username": {"type": "string", "description": "Pseudo Discord (optionnel)"},
            },
            optional_props=["user_id", "username"],
            function=_tool_show,
        ),
    ]
