"""Outils LLM liés aux tâches planifiées (horloge, écoute, veille)."""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

import discord

from common.discord_ui import layout_with_commentary, member_accent_colour, member_accent_value
from common.emojis import SMALL_TASK
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.task_recipe import (
    build_event_trigger,
    build_recipe,
    build_watch_trigger,
    human_status_line,
    keyword_is_specific,
    natural_summary,
    normalize_author_filter,
    pattern_ok,
    quick_delivery_mode,
)
from common.tasks import (
    EVENT_COOLDOWN_DEFAULT,
    EVENT_COOLDOWN_MIN,
    EVENT_MAX_FIRES_DEFAULT,
    EVENT_TTL_DEFAULT_DAYS,
    EVENT_TTL_MAX_DAYS,
    KIND_AT,
    KIND_EVENT,
    KIND_RECURRING,
    KIND_WATCH,
    SCHEDULE_DAILY,
    SCHEDULE_ONCE,
    SCHEDULE_WEEKLY,
    SCOPE_CHANNEL,
    SCOPE_GUILD,
    STATUS_DRAFT,
    STATUS_PAUSED,
    STATUS_PENDING,
    STATUS_RUNNING,
    TASK_INSTRUCTION_MAX,
    TASK_MAX_DAYS,
    TASK_MAX_EVENT,
    TASK_MAX_EVENT_CHANNELS,
    TASK_MAX_GUILD_SCOPE_PER_GUILD,
    TASK_MAX_GUILD_SCOPE_PER_USER,
    TASK_MAX_PENDING,
    TASK_MAX_RECURRING,
    TASK_MAX_WATCH,
    TASK_MIN_MINUTES,
    TASK_MIN_SECONDS,
    VALID_SCHEDULES,
    WATCH_INTERVAL_MIN_MINUTES,
    ScheduledTask,
    TaskStore,
    format_schedule,
    normalize_weekdays,
    snap_execute_at,
)
from common.timezones import PARIS_TZ
from common.layout_kit import sep_wide

TASK_MAX_MINUTES = TASK_MAX_DAYS * 24 * 60


def sanitize_task_instruction(text: str) -> str:
    """Normalise la consigne (trim + plafond), sans retirer « Rappelle… »."""
    return (text or "").strip()[:TASK_INSTRUCTION_MAX]


_QUOTE_RE = re.compile(r"[«\"“]([^»\"”]{1,80})[»\"”]")
_SELF_AUTHOR_RE = re.compile(
    r"\b(?:quand|d[eè]s que|la prochaine fois que|si)\s+je\s+"
    r"(?:dis|dirai|parle|parlerai|écris|ecris|mentionne|mentionnerai)\b"
    r"|\bje\s+dis\b.+\b(?:préviens|previens|dis[- ]moi|ping)\b"
    r"|\b(?:préviens|previens|dis[- ]moi|ping(?:ue)?[- ]?moi).+\bquand je\b",
    re.IGNORECASE,
)


def _extract_quoted(text: str) -> list[str]:
    return [m.group(1).strip() for m in _QUOTE_RE.finditer(text or "") if m.group(1).strip()]


def _pick_event_pattern(pattern: str, topic: str, user_text: str, instruction: str) -> str:
    """Mot-clé concret : arg GPT, sinon mot après « dis/dit » entre guillemets."""
    if pattern and pattern_ok(pattern) is None:
        return pattern.strip()
    # « … que je dis 'Singe' » / « dit le mot « Singe » »
    m = re.search(
        r"(?:dis|dit|parle(?:r)?\s+de|mot)\s+[«\"“']([^»\"”']{2,24})[»\"”']",
        user_text or "",
        re.IGNORECASE,
    )
    if m and pattern_ok(m.group(1).strip()) is None:
        return m.group(1).strip()
    for src in (topic, user_text):
        for q in _extract_quoted(src):
            if pattern_ok(q) is None:
                return q
    return (pattern or "").strip()


def _clean_event_instruction(instruction: str, user_text: str, pattern: str) -> str:
    """Garde le message à poster (ex. STOPPPPP), pas la méta « Répondre X quand Y »."""
    instr = (instruction or "").strip()
    # « tu peux me dire 'STOPPPPP' … » / « dis-moi "…" »
    m = re.search(
        r"(?:me\s+dire|dis[- ]moi|répond(?:re|s)?|envoyer)\s+[«\"“']([^»\"”']{1,80})[»\"”']",
        user_text or "",
        re.IGNORECASE,
    )
    if m:
        q = m.group(1).strip()
        if q and (not pattern or q.casefold() != pattern.casefold()):
            return q
    for q in _extract_quoted(user_text):
        if pattern and q.casefold() == pattern.casefold():
            continue
        if 1 <= len(q) <= 80:
            return q
    # Instruction GPT trop méta : « Répondre « STOPPPPP » quand… »
    low = instr.casefold()
    if low.startswith(("répondre", "repondre", "dire ", "envoyer", "écrire", "ecrire")):
        for q in _extract_quoted(instr):
            if pattern and q.casefold() == pattern.casefold():
                continue
            if 1 <= len(q) <= 80:
                return q
    return instr


def _infer_author_filter(user_text: str, args: dict, owner_id: int) -> str:
    """self si « quand je dis… », sinon arg GPT, sinon not_self."""
    if args.get("author") is not None:
        return normalize_author_filter(args.get("author"), owner_id=owner_id)
    if _SELF_AUTHOR_RE.search(user_text or ""):
        return "self"
    return "not_self"


def _parse_execute_at(execute_at_str: str) -> datetime:
    execute_at = datetime.fromisoformat(execute_at_str)
    if execute_at.tzinfo is None:
        execute_at = execute_at.replace(tzinfo=PARIS_TZ)
    return execute_at.astimezone(timezone.utc)


def _validate_horizon(execute_at: datetime, *, require_min: bool = True) -> str | None:
    delta = (execute_at - datetime.now(timezone.utc)).total_seconds()
    if require_min and delta < TASK_MIN_SECONDS:
        return f"Date trop proche (minimum {TASK_MIN_MINUTES} min)"
    total = int(delta / 60)
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
        "kind": t.kind,
        "scope": t.scope,
        "fires_count": t.fires_count,
        "max_fires": t.max_fires,
        "cooldown_seconds": t.cooldown_seconds,
        "human_status": human_status_line(t),
    }
    if t.until_at:
        item["until_at"] = t.until_at.isoformat()
        item["until_at_ts"] = int(t.until_at.timestamp())
    if t.expires_at:
        item["expires_at"] = t.expires_at.isoformat()
        item["expires_at_ts"] = int(t.expires_at.timestamp())
    if t.last_error:
        item["last_error"] = t.last_error
    return item


def _format_widget_line(item: dict) -> str:
    ts = item["execute_at_ts"]
    desc = item["instruction"]
    kind = item.get("schedule_kind") or SCHEDULE_ONCE
    status = item.get("status") or STATUS_PENDING
    status_bit = " · en pause" if status == STATUS_PAUSED else ""
    task_kind = item.get("kind") or KIND_AT
    if task_kind in (KIND_EVENT, KIND_WATCH):
        label = "Écoute" if task_kind == KIND_EVENT else "Veille"
        return f"**{label}** {item.get('human_status') or ''}{status_bit}\n› {desc}"
    if kind != SCHEDULE_ONCE:
        until = ""
        if item.get("until_at_ts"):
            until = f" · jusqu'au <t:{item['until_at_ts']}:d>"
        dest = " · MP" if item.get("deliver_dm") else ""
        return (
            f"-# <t:{ts}:f> · "
            f"{item.get('schedule_label', kind)}{until}{dest}{status_bit}\n› {desc}"
        )
    dest = " · MP" if item.get("deliver_dm") else ""
    return f"-# <t:{ts}:f> (<t:{ts}:R>){dest}{status_bit}\n› {desc}"


def _clip_text(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _llm_task_line(item: dict) -> str:
    """Ligne vue par le modèle : l'ID est indispensable pour annuler / modifier."""
    kind = item.get("kind") or KIND_AT
    label = _clip_text(item.get("title") or item.get("instruction") or "", 70)
    flags = ""
    if item.get("status") == STATUS_PAUSED:
        flags += " [en pause]"
    if item.get("status") == STATUS_DRAFT:
        flags += " [brouillon]"
    if item.get("deliver_dm"):
        flags += " [MP]"
    if kind in (KIND_EVENT, KIND_WATCH):
        return f"#{item['id']} « {label} » · {item.get('human_status') or kind}{flags}"
    when = datetime.fromtimestamp(item["execute_at_ts"], PARIS_TZ).strftime("%d/%m %H:%M")
    sk = item.get("schedule_kind") or SCHEDULE_ONCE
    sched = "une fois" if sk == SCHEDULE_ONCE else item.get("schedule_label", sk)
    return f"#{item['id']} « {label} » · {sched} · prochaine {when}{flags}"


def _llm_tasks_summary(items: list[dict], header: str) -> str:
    if not items:
        return "Aucune tâche en attente."
    return header + "\n" + "\n".join(_llm_task_line(it) for it in items)


def _coerce_task_id(raw) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) or None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else None


def _accent_kwargs(accent) -> dict:
    if isinstance(accent, discord.Colour) and accent.value:
        return {"accent_colour": accent}
    if isinstance(accent, int) and accent:
        return {"accent_colour": discord.Colour(accent)}
    return {}


async def _send_dm_confirm(
    user: discord.abc.User,
    *,
    label: str,
    instruction: str,
    execute_at: datetime,
) -> str | None:
    """Confirme en MP. None si OK, sinon message d'erreur (MP fermés, etc.)."""
    ts = int(execute_at.timestamp())
    desc = " ".join((instruction or "").split())
    if len(desc) > 160:
        desc = desc[:159] + "…"
    head = f"**Programmé** · *{desc}*" if desc else "**Programmé**"
    bits = [f"{SMALL_TASK} <t:{ts}:f>", f"<t:{ts}:R>"]
    if label and label != "une fois":
        bits.append(label)
    bits.append("en MP")
    foot = f"-# {' · '.join(bits)}"
    view = discord.ui.LayoutView(timeout=None)
    view.add_item(discord.ui.Container(
        discord.ui.TextDisplay(head),
        discord.ui.TextDisplay(foot),
        **_accent_kwargs(member_accent_colour(user)),
    ))
    try:
        await user.send(view=view, allowed_mentions=discord.AllowedMentions.none())
        return None
    except (discord.Forbidden, discord.HTTPException):
        return (
            "Impossible d'envoyer en MP (MP fermés ou bot bloqué). "
            "Tâche annulée. Ouvre tes MP avec moi, ou programme-la dans le salon."
        )


def _task_execute_ts(data: dict) -> Optional[int]:
    ts = data.get("execute_at_ts")
    if isinstance(ts, int) and ts > 0:
        return ts
    raw = (data.get("execute_at") or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PARIS_TZ)
    return int(dt.timestamp())


def make_schedule_widget_builder(store: TaskStore):
    """Widget schedule_task : carte horloge ou ConfirmView event/watch."""

    def build_scheduled_task_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
        if not isinstance(data, dict) or data.get("error") or not data.get("success"):
            return None
        if data.get("needs_confirm") and data.get("task_id"):
            from cogs.chat.views import ConfirmTaskCreateView
            task = store.get(int(data["task_id"]))
            if task is None:
                return None
            quotas = data.get("quotas") or store.quota_summary(task.user_id)
            accent = data.get("accent_colour")
            colour = discord.Colour(accent) if isinstance(accent, int) and accent else None
            return ConfirmTaskCreateView(
                store, task, quotas=quotas, accent_colour=colour,
                commentary=commentary,
                price=data.get("current_price"),
            )
        ts = _task_execute_ts(data)
        if ts is None:
            return None
        desc = " ".join((data.get("instruction") or data.get("title") or "").split())
        if len(desc) > 160:
            desc = desc[:159] + "…"
        head = f"**Programmé** · *{desc}*" if desc else "**Programmé**"
        via = (data.get("via") or "").strip().lower()
        dest = "en MP" if (data.get("deliver_dm") or via in ("mp", "dm", "private")) else "sur ce salon"
        foot = f"-# {SMALL_TASK} <t:{ts}:f> · <t:{ts}:R> · {dest}"
        accent = data.get("accent_colour")
        container = discord.ui.Container(
            discord.ui.TextDisplay(head),
            discord.ui.TextDisplay(foot),
            **_accent_kwargs(accent),
        )
        return layout_with_commentary(container, commentary)

    return build_scheduled_task_view


# Compat import sites that still expect the bare name.
def build_scheduled_task_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if not isinstance(data, dict) or data.get("error") or not data.get("success"):
        return None
    if data.get("needs_confirm"):
        return None
    ts = _task_execute_ts(data)
    if ts is None:
        return None
    desc = " ".join((data.get("instruction") or data.get("title") or "").split())
    if len(desc) > 160:
        desc = desc[:159] + "…"
    head = f"**Programmé** · *{desc}*" if desc else "**Programmé**"
    via = (data.get("via") or "").strip().lower()
    dest = "en MP" if (data.get("deliver_dm") or via in ("mp", "dm", "private")) else "sur ce salon"
    foot = f"-# {SMALL_TASK} <t:{ts}:f> · <t:{ts}:R> · {dest}"
    return layout_with_commentary(
        discord.ui.Container(
            discord.ui.TextDisplay(head),
            discord.ui.TextDisplay(foot),
            **_accent_kwargs(data.get("accent_colour")),
        ),
        commentary,
    )


def build_tasks_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if "error" in data or "display_name" not in data:
        return None
    name = data["display_name"]
    items = data.get("tasks") or []
    quota = data.get("quotas") or {}
    if quota:
        meta = (
            f"{quota.get('event', 0)}/{quota.get('max_event', TASK_MAX_EVENT)} écoutes · "
            f"{quota.get('watch', 0)}/{quota.get('max_watch', TASK_MAX_WATCH)} veille"
        )
    else:
        meta = f"{len(items)} tâche{'s' if len(items) != 1 else ''}"
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## Tâches · {name}"),
        discord.ui.TextDisplay(f"-# {meta}"),
        sep_wide(),
    ]
    if not items:
        children.append(discord.ui.TextDisplay(
            "-# Rien en cours. Dis « préviens-moi si… » ou « surveille ce prix »."
        ))
    else:
        body = "\n\n".join(_format_widget_line(it) for it in items)
        children.append(discord.ui.TextDisplay(body))
    view = layout_with_commentary(
        discord.ui.Container(*children, **_accent_kwargs(data.get("accent_colour"))),
        commentary,
    )
    uid = data.get("user_id")
    if uid:
        from cogs.chat.views import TasksManageButton
        view.add_item(discord.ui.ActionRow(TasksManageButton(int(uid))))
    return view


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


async def _resolve_task(store: TaskStore, user_id: int, args: dict) -> tuple[Optional[ScheduledTask], str]:
    """Cible d'un manage_task : task_id (« 12 », « #12 »), sinon `query`, sinon l'unique tâche."""
    tasks = await asyncio.to_thread(store.get_user_tasks, user_id)
    items = [_serialize_task(t) for t in tasks]
    listing = "\n".join(_llm_task_line(it) for it in items)
    tid = _coerce_task_id(args.get("task_id"))
    if tid is not None:
        for t in tasks:
            if t.id == tid:
                return t, ""
        running = await asyncio.to_thread(store.get, tid)
        if running is not None and running.user_id == user_id and running.status == STATUS_RUNNING:
            return running, ""
        hint = f" Tes tâches :\n{listing}" if listing else " Tu n'as aucune tâche active."
        return None, f"Tâche #{tid} introuvable parmi tes tâches actives.{hint}"
    query = " ".join(str(args.get("query") or "").split()).casefold()
    if query:
        hits = [t for t in tasks if query in f"{t.title} {t.instruction}".casefold()]
        if len(hits) == 1:
            return hits[0], ""
        if not hits:
            hint = f" Tes tâches :\n{listing}" if listing else ""
            return None, f"Aucune tâche ne correspond à « {query} ».{hint}"
        lines = "\n".join(_llm_task_line(_serialize_task(t)) for t in hits)
        return None, f"Plusieurs tâches correspondent, précise l'ID :\n{lines}"
    if len(tasks) == 1:
        return tasks[0], ""
    if not tasks:
        return None, "Tu n'as aucune tâche active."
    return None, f"Précise laquelle (task_id) :\n{listing}"


def _member_is_mod(member) -> bool:
    perms = getattr(member, "guild_permissions", None)
    return bool(perms and perms.manage_guild)


def _parse_ttl_days(raw) -> int:
    try:
        days = int(raw)
    except (TypeError, ValueError):
        days = EVENT_TTL_DEFAULT_DAYS
    return max(1, min(EVENT_TTL_MAX_DAYS, days))


def _text_flags(*texts: str) -> dict[str, bool]:
    """Indices lexicaux du rythme voulu (filet quand JEV est absent)."""
    blob = " ".join(t for t in texts if t).casefold()
    once = any(p in blob for p in (
        "une seule fois", "juste une fois", "une fois seulement", "qu'une fois",
        "la prochaine fois", "dès la prochaine", "des la prochaine",
    )) or (
        "une fois" in blob
        and "des fois" not in blob
        and "quelques fois" not in blob
        and "plusieurs fois" not in blob
    )
    return {
        "one_shot": once,
        "short_term": any(p in blob for p in (
            "ce soir", "aujourd'hui", "cette nuit", "dans la soirée",
            "ce week-end", "ce weekend", "demain",
        )),
        "tonight": "ce soir" in blob or "aujourd" in blob or "cette nuit" in blob,
        "week": any(p in blob for p in (
            "cette semaine", "la semaine", "quelques jours", "d'ici peu",
        )),
        "long_term": any(p in blob for p in (
            "longtemps", "un mois", "pendant un mois", "promo", "solde",
            "black friday", "restock", "baisse", "baisser",
        )),
        "urgent": any(p in blob for p in (
            "dès que", "des que", "au plus vite", "vite", "urgent", "immédiat",
            "tout de suite", "direct",
        )),
        "low_noise": any(p in blob for p in (
            "pas trop souvent", "sans spam", "tranquille", "de temps en temps",
            "pas toutes les",
        )),
        "gaming": any(p in blob for p in (
            "ranked", "rank", "lobby", "duo", "scrim", "custom", "valorant",
            "league", "lol ", "cs2", "game", "jouer", "partie",
        )),
    }


def _infer_trigger_limits(
    kind: str,
    *texts: str,
    jev_flags: Optional[dict[str, bool]] = None,
) -> dict:
    """Choisit cooldown / max_fires / ttl : le membre ne les dicte pas.

    JEV (si dispo) et indices lexicaux sont combinés (OU) ; défauts sinon.
    """
    f = _text_flags(*texts)
    for key, val in (jev_flags or {}).items():
        if val:
            f[key] = True
    cd_h = EVENT_COOLDOWN_DEFAULT // 3600
    max_fires = EVENT_MAX_FIRES_DEFAULT
    ttl_days = EVENT_TTL_DEFAULT_DAYS
    interval = WATCH_INTERVAL_MIN_MINUTES

    if f["one_shot"]:
        max_fires, cd_h = 1, 1
        ttl_days = 2 if f["short_term"] else 7
    elif f["short_term"]:
        ttl_days = 1 if f["tonight"] else 2
        max_fires, cd_h = 3, 1
    elif f["week"]:
        ttl_days, max_fires, cd_h = 7, 5, 2
    elif f["long_term"]:
        ttl_days = 21 if kind == KIND_WATCH else 14
        max_fires, cd_h = 5, 3
    elif f["gaming"]:
        ttl_days, max_fires, cd_h = 7, 5, 3

    if f["urgent"] and not f["one_shot"]:
        cd_h = 1
        max_fires = max(max_fires, 5)
    if f["low_noise"]:
        cd_h = max(cd_h, 6)
        max_fires = min(max_fires, 3)

    if kind == KIND_WATCH:
        cd_h = max(cd_h, 6)
        if f["short_term"]:
            ttl_days = min(ttl_days, 3)
            max_fires = min(max_fires, 3)
        elif f["long_term"]:
            ttl_days = max(ttl_days, 14)

    return {
        "cooldown_hours": max(1, min(24, cd_h)),
        "max_fires": max(1, min(20, max_fires)),
        "ttl_days": max(1, min(EVENT_TTL_MAX_DAYS, ttl_days)),
        "interval_minutes": max(WATCH_INTERVAL_MIN_MINUTES, interval),
    }


async def _delivery_mode(typesafe, instruction: str) -> str:
    """Regex d'abord, JEV seulement si ambigu ; sans verdict, GPT rédige (comportement sûr)."""
    quick = quick_delivery_mode(instruction)
    if quick:
        return quick
    if typesafe is None:
        return "generate"
    return (await typesafe.classify_alert_mode(instruction)) or "generate"


def _watch_var_key(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"watch:price:{h}"


def _discord_client(ctx):
    try:
        return ctx.trigger_message._state._get_client()
    except Exception:
        return None


def _quota_hint(store: TaskStore, user_id: int, kind: str | None = None) -> str:
    """Liste courte des tâches du membre : GPT peut proposer laquelle libérer."""
    tasks = store.get_user_tasks(user_id)
    if kind:
        tasks = [t for t in tasks if t.kind == kind]
    if not tasks:
        return ""
    lines = "\n".join(_llm_task_line(_serialize_task(t)) for t in tasks[:6])
    return (
        "\nTâches concernées (propose d'annuler la moins utile avec manage_task cancel, "
        f"puis relance) :\n{lines}"
    )


async def _fetch_page_price(
    ctx, store: TaskStore, guild_id: int, url: str, typesafe=None,
) -> tuple[Optional[float], str, str]:
    """(prix, ancre, erreur FR). Un fetch consomme le budget web du serveur."""
    from common.task_watch import pick_price

    if not await asyncio.to_thread(store.consume_watch_budget, guild_id, n=1):
        return None, "", "Quota de veille du serveur atteint pour aujourd'hui, réessaie demain."
    client = _discord_client(ctx)
    web = client.get_cog("Web") if client is not None else None
    if web is None or not hasattr(web, "_crawl_page"):
        return None, "", "Lecture de page indisponible pour le moment."
    try:
        text = await asyncio.to_thread(web._crawl_page, url) or ""
    except Exception:
        text = ""
    price, anchor = await pick_price(text[:8000], typesafe)
    if price is None:
        return None, "", (
            "Je ne trouve pas de prix en € sur cette page (site protégé ou prix chargé en JS). "
            "Essaie une autre URL de la même offre."
        )
    return price, anchor, ""


async def _tool_schedule_event_watch(
    tc: ToolCallRecord,
    ctx,
    store: TaskStore,
    *,
    kind: str,
    typesafe=None,
) -> ToolResponseRecord:
    msg = ctx.trigger_message
    guild = msg.guild
    author = msg.author
    now = datetime.now(timezone.utc)
    args = tc.arguments or {}
    if guild is None:
        return ToolResponseRecord(tc.id, {
            "error": "Écoute / veille : uniquement sur un serveur.",
        }, now)
    instruction = sanitize_task_instruction(args.get("instruction") or "")
    if not instruction:
        instruction = (
            "Préviens-moi."
            if kind == KIND_EVENT
            else "Préviens-moi que le prix a bougé."
        )

    # Un nouveau brouillon remplace l'ancien (annulé seulement une fois la demande validée,
    # pour qu'un refus ne détruise pas le brouillon en cours). L'ancien ne compte pas au quota.
    old_draft = await asyncio.to_thread(store.latest_draft, author.id, kind)
    old_id = old_draft.id if old_draft else 0

    if await asyncio.to_thread(store.count_active, author.id, exclude_id=old_id) >= TASK_MAX_PENDING:
        hint = await asyncio.to_thread(_quota_hint, store, author.id)
        return ToolResponseRecord(tc.id, {
            "error": f"Tu as déjà {TASK_MAX_PENDING} tâches (max).{hint}",
        }, now)

    title = (args.get("title") or "").strip()
    user_text = (getattr(msg, "clean_content", None) or msg.content or "")[:400]
    jev_flags = None
    rx = _text_flags(user_text, instruction, title)
    rx_conclusive = any(rx[k] for k in ("one_shot", "short_term", "week", "long_term"))
    if typesafe is not None and not rx_conclusive:  # regex suffit quand le rythme est explicite
        jev_flags = await typesafe.infer_task_flags(
            f"{user_text}\n{instruction}", kind=kind,
        )
    suggested = _infer_trigger_limits(
        kind, user_text, instruction, title, jev_flags=jev_flags,
    )
    if args.get("ttl_days") is not None or args.get("expire_days") is not None:
        ttl_days = _parse_ttl_days(args.get("ttl_days") or args.get("expire_days"))
    else:
        ttl_days = suggested["ttl_days"]
    expires_at = now + timedelta(days=ttl_days)
    accent = member_accent_value(author)
    deliver_dm = (args.get("via") or "").strip().lower() in ("dm", "mp", "private")

    def _int_arg(name: str, default: int) -> int:
        if args.get(name) is None:
            return default
        try:
            return int(args[name])
        except (TypeError, ValueError):
            return default

    current_price: Optional[float] = None

    if kind == KIND_EVENT:
        n_ev = await asyncio.to_thread(
            store.count_kind, author.id, KIND_EVENT, exclude_id=old_id,
        )
        if n_ev >= TASK_MAX_EVENT:
            hint = await asyncio.to_thread(_quota_hint, store, author.id, KIND_EVENT)
            return ToolResponseRecord(tc.id, {
                "error": f"Tu as déjà {TASK_MAX_EVENT} écoutes (max).{hint}",
            }, now)
        topic = (args.get("topic") or "").strip()[:160]
        pattern = _pick_event_pattern(
            (args.get("pattern") or args.get("keyword") or "").strip(),
            topic, user_text, instruction,
        )
        instruction = _clean_event_instruction(instruction, user_text, pattern) or instruction
        jev_on = typesafe is not None and getattr(typesafe, "enabled", False)
        if not pattern and not (topic and jev_on):
            return ToolResponseRecord(tc.id, {
                "error": (
                    "Il me faut un mot-clé précis (pattern)"
                    + (" ou une description du sujet (topic)." if jev_on else ".")
                ),
            }, now)
        if pattern:
            err_pat = pattern_ok(pattern)
            if err_pat:
                return ToolResponseRecord(tc.id, {
                    "error": f"{err_pat} Reformule avec un mot-clé plus précis"
                    + (" ou décris juste le sujet (topic)." if jev_on else "."),
                }, now)
            # JEV : mot banal sans sujet précis → alertes en rafale. Fail-open.
            if (
                typesafe is not None
                and not topic
                and not keyword_is_specific(pattern)  # regex : mot distinctif → pas de JEV
                and await typesafe.is_keyword_too_generic(pattern, user_text)
            ):
                return ToolResponseRecord(tc.id, {
                    "error": (
                        f"« {pattern} » est trop courant pour une écoute fiable. "
                        "Propose un mot plus précis ou décris le sujet avec topic."
                    ),
                }, now)
        # Topic trop « scénario » alors qu'on a un mot-clé : on le simplifie.
        if pattern and topic and (
            len(topic) > 40
            or "dit le mot" in topic.casefold()
            or pattern.casefold() in topic.casefold()
        ):
            topic = ""
        dup = await asyncio.to_thread(
            store.find_event_duplicate, author.id, guild.id, pattern or topic,
        )
        if dup is not None:
            return ToolResponseRecord(tc.id, {
                "error": (
                    f"Tu écoutes déjà « {pattern or topic} » (tâche #{dup.id}). "
                    "Dis-lui qu'elle est active ; pour la changer, utilise manage_task edit."
                ),
            }, now)

        want_guild = bool(args.get("guild_scope") or args.get("scope") == "guild")
        explicit = bool(args.get("explicit_guild_scope"))
        if want_guild and not explicit:
            return ToolResponseRecord(tc.id, {
                "error": (
                    "Scope serveur réservé aux modos, et seulement si le membre le demande "
                    "clairement (« tout le serveur », « tous les salons »). "
                    "Sinon, écoute juste ce salon."
                ),
            }, now)
        scope = SCOPE_CHANNEL
        channel_ids = [msg.channel.id]
        if want_guild and explicit:
            if not _member_is_mod(author):
                return ToolResponseRecord(tc.id, {
                    "error": (
                        "Scope serveur réservé aux modos (permission Gérer le serveur). "
                        "Propose d'écouter seulement ce salon."
                    ),
                }, now)
            total_g, mine_g = await asyncio.to_thread(
                store.count_guild_scope_events, guild.id, user_id=author.id,
            )
            if mine_g >= TASK_MAX_GUILD_SCOPE_PER_USER:
                return ToolResponseRecord(tc.id, {
                    "error": "Tu as déjà 1 écoute serveur (max pour un modo).",
                }, now)
            if total_g >= TASK_MAX_GUILD_SCOPE_PER_GUILD:
                return ToolResponseRecord(tc.id, {
                    "error": (
                        f"Ce serveur a déjà {TASK_MAX_GUILD_SCOPE_PER_GUILD} écoutes "
                        "serveur actives (max)."
                    ),
                }, now)
            scope = SCOPE_GUILD
            channel_ids = []
        else:
            extra = args.get("channel_ids") or args.get("channels") or []
            if isinstance(extra, str):
                extra = [p.strip() for p in extra.split(",") if p.strip()]
            ids: list[int] = [msg.channel.id]
            for raw in extra:
                try:
                    cid = int(raw)
                except (TypeError, ValueError):
                    continue
                if cid not in ids:
                    ids.append(cid)
            channel_ids = ids[:TASK_MAX_EVENT_CHANNELS]
            for cid in list(channel_ids):
                ch = guild.get_channel(cid)
                if ch is None:
                    continue
                if not ch.permissions_for(author).view_channel:
                    channel_ids = [c for c in channel_ids if c != cid]
            if not channel_ids:
                return ToolResponseRecord(tc.id, {
                    "error": "Aucun salon visible pour cette écoute.",
                }, now)

        if args.get("cooldown_seconds") is not None:
            cd = _int_arg("cooldown_seconds", suggested["cooldown_hours"] * 3600)
        elif args.get("cooldown_hours") is not None:
            try:
                cd = int(float(args["cooldown_hours"]) * 3600)
            except (TypeError, ValueError):
                cd = suggested["cooldown_hours"] * 3600
        else:
            cd = suggested["cooldown_hours"] * 3600
        cd = max(EVENT_COOLDOWN_MIN, cd)
        max_fires = max(1, min(20, _int_arg("max_fires", suggested["max_fires"])))

        author_filter = _infer_author_filter(user_text, args, author.id)
        trigger = build_event_trigger(
            pattern=pattern,
            channel_ids=channel_ids or [msg.channel.id],
            author=author_filter,
            scope=scope,
            aliases=args.get("aliases"),
            intent=user_text,
            topic=topic,
            owner_id=author.id,
        )
        recipe = build_recipe(
            say=instruction, ping=True, mode=await _delivery_mode(typesafe, instruction),
        )
        await asyncio.to_thread(store.cancel_drafts, author.id, kind)
        tid = await asyncio.to_thread(
            store.add,
            channel_id=msg.channel.id,
            user_id=author.id,
            guild_id=guild.id,
            instruction=instruction,
            execute_at=now,
            title=title or f'Écoute « {pattern or topic[:40]} »',
            kind=KIND_EVENT,
            trigger=trigger,
            recipe=recipe,
            scope=scope,
            channel_ids=channel_ids or [msg.channel.id],
            cooldown_seconds=cd,
            max_fires=max_fires,
            expires_at=expires_at,
            status=STATUS_DRAFT,
            message_id=msg.id,
            deliver_dm=deliver_dm,
        )
    else:
        n_w = await asyncio.to_thread(
            store.count_kind, author.id, KIND_WATCH, exclude_id=old_id,
        )
        if n_w >= TASK_MAX_WATCH:
            hint = await asyncio.to_thread(_quota_hint, store, author.id, KIND_WATCH)
            return ToolResponseRecord(tc.id, {
                "error": (
                    "Tu as déjà une veille (1 max, vérif toutes les 6 h min, expire sous 30 j)."
                    f"{hint}"
                ),
            }, now)
        url = (args.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ToolResponseRecord(tc.id, {
                "error": (
                    "URL manquante ou invalide. Si le membre n'en a pas donné, "
                    "trouve la page produit avec search_web puis relance avec l'URL."
                ),
            }, now)
        # On lit la page maintenant : prouve que le prix est lisible + donne le seuil auto.
        current_price, anchor, fetch_err = await _fetch_page_price(
            ctx, store, guild.id, url, typesafe,
        )
        if current_price is None:
            return ToolResponseRecord(tc.id, {"error": fetch_err}, now)
        op = (args.get("op") or "lt").strip().lower()
        if args.get("threshold") is not None or args.get("price") is not None:
            try:
                threshold = float(
                    args.get("threshold") if args.get("threshold") is not None else args.get("price")
                )
            except (TypeError, ValueError):
                return ToolResponseRecord(tc.id, {"error": "Seuil de prix invalide."}, now)
        else:
            # « préviens-moi si ça baisse » : seuil auto = -10 % (ou -5 € mini sous 50 €).
            try:
                pct = float(args.get("drop_percent") or 10)
            except (TypeError, ValueError):
                pct = 10.0
            pct = max(2.0, min(60.0, pct))
            threshold = round(current_price * (1 - pct / 100), 2)
            op = "lt"
        interval = max(
            WATCH_INTERVAL_MIN_MINUTES,
            _int_arg("interval_minutes", suggested["interval_minutes"]),
        )
        max_fires_w = max(1, min(20, _int_arg("max_fires", suggested["max_fires"])))
        var_key = _watch_var_key(url)
        await asyncio.to_thread(
            store.set_var, guild.id, author.id, var_key, f"{current_price:.2f}",
        )
        trigger = build_watch_trigger(
            url=url,
            threshold=threshold,
            op=op,
            interval_minutes=interval,
            var_key=var_key,
            anchor=anchor,
        )
        recipe = build_recipe(
            say=instruction, ping=True, mode=await _delivery_mode(typesafe, instruction),
        )
        await asyncio.to_thread(store.cancel_drafts, author.id, kind)
        tid = await asyncio.to_thread(
            store.add,
            channel_id=msg.channel.id,
            user_id=author.id,
            guild_id=guild.id,
            instruction=instruction,
            execute_at=now + timedelta(minutes=interval),
            title=title or f"Veille ≤ {threshold:g}€",
            kind=KIND_WATCH,
            trigger=trigger,
            recipe=recipe,
            scope=SCOPE_CHANNEL,
            channel_ids=[msg.channel.id],
            cooldown_seconds=max(EVENT_COOLDOWN_MIN, suggested["cooldown_hours"] * 3600),
            max_fires=max_fires_w,
            expires_at=expires_at,
            status=STATUS_DRAFT,
            message_id=msg.id,
            deliver_dm=deliver_dm,
        )

    created = await asyncio.to_thread(store.get, tid)
    quotas = await asyncio.to_thread(store.quota_summary, author.id)
    payload = _serialize_task(created) if created else {"id": tid}
    summary_txt = natural_summary(created, price=current_price) if created else ""
    kind_fr = "écoute" if kind == KIND_EVENT else "veille"
    return ToolResponseRecord(tc.id, {
        "_tool": "schedule_task",
        "success": True,
        "needs_confirm": True,
        "task_id": tid,
        "kind": kind,
        "quotas": quotas,
        "current_price": current_price,
        "natural_summary": summary_txt,
        **payload,
        "accent_colour": accent,
        "_llm_summary": (
            f"Brouillon {kind_fr} #{tid} prêt, en attente de confirmation (boutons affichés ; "
            f"si le membre répond oui/ok/vas-y, appelle manage_task action=confirm). "
            f"Résumé : {summary_txt} "
            "Dis-le en une phrase naturelle, sans lister les plafonds comme un règlement."
        ),
    }, now)


def build_task_tools(store: TaskStore, typesafe=None) -> list[Tool]:
    """Construit les outils de planification / gestion des tâches."""

    async def _tool_schedule(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments or {}
        instruction = sanitize_task_instruction(args.get("instruction") or "")

        task_kind = (args.get("kind") or "").strip().lower()
        if task_kind in (KIND_EVENT, "listen", "message"):
            return await _tool_schedule_event_watch(
                tc, ctx, store, kind=KIND_EVENT, typesafe=typesafe,
            )
        if task_kind in (KIND_WATCH, "url", "price"):
            return await _tool_schedule_event_watch(
                tc, ctx, store, kind=KIND_WATCH, typesafe=typesafe,
            )

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
        if await asyncio.to_thread(store.count_active, ctx.trigger_message.author.id) >= TASK_MAX_PENDING:
            return ToolResponseRecord(
                tc.id, {
                    "error": (
                        f"Tu as déjà {TASK_MAX_PENDING} tâches (max). "
                        "Annule-en une avec /taches."
                    ),
                },
                datetime.now(timezone.utc),
            )
        if kind != SCHEDULE_ONCE:
            n_rep = await asyncio.to_thread(
                store.count_active_recurring, ctx.trigger_message.author.id,
            )
            if n_rep >= TASK_MAX_RECURRING:
                return ToolResponseRecord(
                    tc.id,
                    {"error": (
                        f"Tu as déjà {TASK_MAX_RECURRING} tâches répétitives (max). "
                        "Passe-en une en unique ou annule-en une avec /taches."
                    )},
                    datetime.now(timezone.utc),
                )

        via = (args.get("via") or "").strip().lower()
        if not via:
            via = "dm" if ctx.trigger_message.guild is None else "channel"
        deliver_dm = via in ("dm", "mp", "private")

        title = (args.get("title") or "").strip()
        guild = ctx.trigger_message.guild
        task_kind_store = KIND_RECURRING if kind != SCHEDULE_ONCE else KIND_AT
        tid = await asyncio.to_thread(
            store.add,
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
            kind=task_kind_store,
        )
        created = await asyncio.to_thread(store.get, tid)
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
                await asyncio.to_thread(store.cancel, tid, ctx.trigger_message.author.id)
                return ToolResponseRecord(
                    tc.id, {"error": dm_err}, datetime.now(timezone.utc),
                )
        payload = _serialize_task(created) if created else {
            "id": tid,
            "title": title or instruction,
            "instruction": instruction,
            "execute_at": execute_at.isoformat(),
            "execute_at_ts": int(execute_at.timestamp()),
            "schedule_kind": kind,
            "weekdays": days,
            "time_of_day": time_of_day,
            "status": STATUS_PENDING,
            "schedule_label": label,
            "deliver_dm": deliver_dm,
            "kind": task_kind_store,
        }
        accent = member_accent_value(ctx.trigger_message.author)
        quotas = await asyncio.to_thread(store.quota_summary, ctx.trigger_message.author.id)
        near = quotas["total"] >= TASK_MAX_PENDING - 1
        quota_note = (
            f" Quotas : {quotas['total']}/{quotas['max_total']} tâches."
            if near else ""
        )
        return ToolResponseRecord(tc.id, {
            "_tool": "schedule_task",
            "success": True,
            "task_id": tid,
            "schedule": label,
            "via": dest,
            "quotas": quotas,
            **payload,
            "accent_colour": accent,
            "_llm_summary": (
                f"Tâche #{tid} programmée ({label}, {dest}) : "
                f"{_llm_task_line(payload)}.{quota_note}"
            ),
        }, datetime.now(timezone.utc))

    async def _tool_manage(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message:
            return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
        args = tc.arguments or {}
        action = (args.get("action") or "list").strip().lower()
        user_id = ctx.trigger_message.author.id

        def _invalidate(guild_id: int) -> None:
            try:
                client = ctx.trigger_message._state._get_client()
            except Exception:
                return
            cog = client.get_cog("Chat")
            if cog is not None and hasattr(cog, "event_triggers"):
                cog.event_triggers.invalidate(guild_id)

        if action == "list":
            tasks = await asyncio.to_thread(store.get_user_tasks, user_id)
            items = [_serialize_task(t) for t in tasks]
            quotas = await asyncio.to_thread(store.quota_summary, user_id)
            return ToolResponseRecord(tc.id, {
                "tasks": items,
                "quotas": quotas,
                "_llm_summary": _llm_tasks_summary(
                    items,
                    (
                        f"{len(items)} tâche(s) · "
                        f"{quotas['total']}/{quotas['max_total']} · "
                        f"Écoute {quotas['event']}/{quotas['max_event']} · "
                        f"Veille {quotas['watch']}/{quotas['max_watch']} "
                        f"(utilise l'ID #n pour agir) :"
                    ),
                ),
            }, datetime.now(timezone.utc))

        if action == "cancel_all":
            n = await asyncio.to_thread(store.cancel_all, user_id)
            g = ctx.trigger_message.guild
            if g:
                _invalidate(g.id)
            return ToolResponseRecord(tc.id, {
                "success": True, "cancelled": n,
                "_llm_summary": f"{n} tâche(s) annulée(s)." if n else "Aucune tâche à annuler.",
            }, datetime.now(timezone.utc))

        if action == "confirm":
            tid_arg = _coerce_task_id(args.get("task_id"))
            draft = (
                await asyncio.to_thread(store.get, tid_arg) if tid_arg
                else await asyncio.to_thread(store.latest_draft, user_id)
            )
            if draft is None or draft.user_id != user_id or draft.status != STATUS_DRAFT:
                return ToolResponseRecord(tc.id, {
                    "error": "Aucun brouillon à confirmer (expiré après 10 min ? Recrée-le).",
                }, datetime.now(timezone.utc))
            ok = await asyncio.to_thread(store.confirm_draft, draft.id, user_id)
            if not ok:
                return ToolResponseRecord(tc.id, {
                    "error": "Brouillon expiré ou déjà confirmé.",
                }, datetime.now(timezone.utc))
            if draft.guild_id:
                _invalidate(draft.guild_id)
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": draft.id,
                "_llm_summary": (
                    f"Tâche #{draft.id} activée. {natural_summary(draft)} "
                    "Confirme-le brièvement, sans réciter les plafonds."
                ),
            }, datetime.now(timezone.utc))

        target, why = await _resolve_task(store, user_id, args)
        if target is None:
            return ToolResponseRecord(tc.id, {"error": why}, datetime.now(timezone.utc))
        tid = target.id

        if action == "cancel":
            ok = await asyncio.to_thread(store.cancel, tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Tâche introuvable ou pas la tienne."},
                    datetime.now(timezone.utc),
                )
            if target.guild_id:
                _invalidate(target.guild_id)
            left = await asyncio.to_thread(store.count_active, user_id)
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": (
                    f"Tâche #{tid} « {_clip_text(target.title or target.instruction, 60)} » annulée. "
                    f"Il te reste {left} tâche(s) active(s)."
                ),
            }, datetime.now(timezone.utc))

        if action == "pause":
            ok = await asyncio.to_thread(store.pause, tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Impossible de mettre en pause (introuvable, déjà en pause, ou pas la tienne)."},
                    datetime.now(timezone.utc),
                )
            if target.guild_id:
                _invalidate(target.guild_id)
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche en pause.",
            }, datetime.now(timezone.utc))

        if action == "resume":
            ok = await asyncio.to_thread(store.resume, tid, user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Impossible de reprendre (pas en pause, ou pas la tienne)."},
                    datetime.now(timezone.utc),
                )
            if target.guild_id:
                _invalidate(target.guild_id)
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": "Tâche reprise.",
            }, datetime.now(timezone.utc))

        if action == "skip":
            nxt = await asyncio.to_thread(store.skip_next, tid, user_id)
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
            current = await asyncio.to_thread(store.get, tid)
            rec_kind = kind or (current.schedule_kind if current else SCHEDULE_ONCE)
            if execute_at is not None and rec_kind == SCHEDULE_ONCE:
                err = _validate_horizon(execute_at)
                if err:
                    return ToolResponseRecord(tc.id, {"error": err}, datetime.now(timezone.utc))
            if kind in (SCHEDULE_DAILY, SCHEDULE_WEEKLY):
                already = bool(current and current.schedule_kind != SCHEDULE_ONCE)
                n_rep = await asyncio.to_thread(
                    store.count_active_recurring, user_id, exclude_id=tid,
                )
                if not already and n_rep >= TASK_MAX_RECURRING:
                    return ToolResponseRecord(
                        tc.id,
                        {"error": (
                            f"Tu as déjà {TASK_MAX_RECURRING} tâches répétitives (max). "
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

            pattern = args.get("pattern") or args.get("keyword")
            if isinstance(pattern, str) and pattern.strip():
                err_pat = pattern_ok(pattern.strip())
                if err_pat:
                    return ToolResponseRecord(tc.id, {"error": err_pat}, datetime.now(timezone.utc))
            else:
                pattern = None

            cooldown_seconds = None
            if args.get("cooldown_seconds") is not None:
                try:
                    cooldown_seconds = int(args["cooldown_seconds"])
                except (TypeError, ValueError):
                    pass
            elif args.get("cooldown_hours") is not None:
                try:
                    cooldown_seconds = int(float(args["cooldown_hours"]) * 3600)
                except (TypeError, ValueError):
                    pass
            if cooldown_seconds is not None and cooldown_seconds < EVENT_COOLDOWN_MIN:
                return ToolResponseRecord(tc.id, {
                    "error": f"Cooldown event : minimum {EVENT_COOLDOWN_MIN // 3600} h.",
                }, datetime.now(timezone.utc))

            max_fires = None
            if args.get("max_fires") is not None:
                try:
                    max_fires = int(args["max_fires"])
                except (TypeError, ValueError):
                    pass

            ttl_days_edit = None
            if args.get("ttl_days") is not None:
                try:
                    ttl_days_edit = int(args["ttl_days"])
                except (TypeError, ValueError):
                    pass

            threshold = None
            if args.get("threshold") is not None:
                try:
                    threshold = float(args["threshold"])
                except (TypeError, ValueError):
                    return ToolResponseRecord(tc.id, {
                        "error": "Seuil invalide.",
                    }, datetime.now(timezone.utc))

            ok = await asyncio.to_thread(
                store.edit,
                tid, user_id,
                instruction=new_instr,
                execute_at=execute_at,
                schedule_kind=kind,
                weekdays=days,
                time_of_day=time_of_day,
                until_at=until_at,
                deliver_dm=deliver_dm,
                pattern=pattern,
                cooldown_seconds=cooldown_seconds,
                max_fires=max_fires,
                threshold=threshold,
                ttl_days=ttl_days_edit,
            )
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Tâche introuvable, déjà passée, ou pas la tienne."},
                    datetime.now(timezone.utc),
                )
            if target.guild_id:
                _invalidate(target.guild_id)
            fresh = await asyncio.to_thread(store.get, tid)
            recap = natural_summary(fresh) if fresh and fresh.kind in (KIND_EVENT, KIND_WATCH) else ""
            note = " Brouillon toujours en attente de confirmation." if (
                fresh and fresh.status == STATUS_DRAFT
            ) else ""
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": tid,
                "_llm_summary": f"Tâche modifiée. {recap}{note}".strip(),
            }, datetime.now(timezone.utc))

        return ToolResponseRecord(
            tc.id,
            {"error": "action inconnue (list|edit|confirm|pause|resume|skip|cancel|cancel_all)"},
            datetime.now(timezone.utc),
        )

    async def _tool_show(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        member, err = await _resolve_member(ctx, tc.arguments or {})
        if err or member is None:
            return ToolResponseRecord(tc.id, {"error": err or "Membre introuvable"}, datetime.now(timezone.utc))
        pending = await asyncio.to_thread(store.get_user_tasks, member.id)
        items = [_serialize_task(t) for t in pending]
        quotas = await asyncio.to_thread(store.quota_summary, member.id)
        name = getattr(member, "display_name", None) or member.name
        return ToolResponseRecord(tc.id, {
            "_tool": "show_tasks",
            "user_id": str(member.id),
            "display_name": name,
            "count": len(items),
            "tasks": items,
            "quotas": quotas,
            "accent_colour": member_accent_value(ctx.trigger_message.author),
            "_llm_summary": (
                _llm_tasks_summary(items, f"Widget tâches de {name} affiché ({len(items)}) :")
                if items else f"Aucune tâche en attente pour {name}."
            ),
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="schedule_task",
            description=(
                "Programme une tâche. Kinds : at/recurring (horloge), event (écoute mot-clé), "
                "watch (veille prix URL). "
                f"Max {TASK_MAX_PENDING} tâches, dont {TASK_MAX_EVENT} écoutes, "
                f"{TASK_MAX_WATCH} veille, {TASK_MAX_RECURRING} répétitives. "
                "event/watch → brouillon + Confirm View (le membre clique Confirmer, ou dit oui → "
                "manage_task confirm). "
                "TU gères tout : le membre ne précise presque jamais cooldown / max / durée. "
                "Omets cooldown_hours / max_fires / ttl_days : l'outil les déduit (JEV + contexte) "
                "— ne les passe que si le membre a été explicite. Ne demande JAMAIS « combien de fois ? ». "
                "Écoute : si le membre cite un mot (« Singe », « ranked ») → pattern = ce mot "
                "(obligatoire). topic seulement pour un sujet flou sans mot exact. "
                "instruction = le message exact à envoyer (« STOPPPPP »), PAS une méta "
                "« Répondre X quand Y ». author=self si « quand JE dis… », sinon not_self. "
                "Veille : si pas d'URL, trouve la page produit avec search_web ; si pas de seuil, "
                "omets threshold (seuil auto -10 %, ou drop_percent). La page est lue à la création. "
                "Défauts si indécis : cd 3 h, 5 alertes, 7 j (veille : check ≥6 h). "
                "Horloge : execute_at ISO / delay ; recurrence once|daily|weekly. "
                f"Min ~{TASK_MIN_MINUTES} min, max {TASK_MAX_DAYS}j. "
                "via=dm seulement si MP/DM demandé clairement. "
                "Scope serveur (event) : modos + explicit_guild_scope=true seulement."
            ),
            properties={
                "instruction": {
                    "type": "string",
                    "description": (
                        "Consigne à exécuter / message d'alerte. "
                        "OK : « Rappelle d'aller à la salle ». "
                        "Pour event/watch : ce que tu diras en alertant."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["at", "recurring", "event", "watch"],
                    "description": "at=unique, recurring=via recurrence, event=écoute, watch=prix URL",
                },
                "title": {"type": "string", "description": "Libellé court UI (optionnel)"},
                "execute_at": {"type": "string", "description": "Date/heure ISO 8601 (horloge)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes"},
                "delay_hours": {"type": "integer", "description": "Délai en heures"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_SCHEDULES),
                    "description": "once|daily|weekly (horloge)",
                },
                "weekdays": {
                    "type": "string",
                    "description": "Jours weekly : mon,tue,…",
                },
                "time": {"type": "string", "description": "HH:MM Paris daily/weekly"},
                "until": {"type": "string", "description": "Fin de série ISO"},
                "via": {
                    "type": "string",
                    "enum": ["channel", "dm"],
                    "description": "channel défaut ; dm si demandé explicitement",
                },
                "pattern": {
                    "type": "string",
                    "description": "Mot-clé écoute (3–24 car., pas générique)",
                },
                "keyword": {"type": "string", "description": "Alias de pattern"},
                "topic": {
                    "type": "string",
                    "description": (
                        "Écoute : sujet décrit en une phrase (« quelqu'un propose une partie "
                        "de ranked »). Détecté par le sens (JEV), sans mot-clé exact. "
                        "Préfère topic dès que le sujet peut se dire de plusieurs façons."
                    ),
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1–3 variantes du mot-clé (event), mêmes règles que pattern",
                },
                "url": {"type": "string", "description": "URL à surveiller (watch)"},
                "threshold": {
                    "type": "number",
                    "description": "Seuil prix € (watch). Omettre = -10 % du prix actuel.",
                },
                "drop_percent": {
                    "type": "number",
                    "description": "Baisse en % pour le seuil auto (watch, défaut 10)",
                },
                "op": {
                    "type": "string",
                    "enum": ["lt", "lte", "gt", "change"],
                    "description": "Comparaison prix (défaut lt)",
                },
                "interval_minutes": {
                    "type": "integer",
                    "description": f"Intervalle veille (min {WATCH_INTERVAL_MIN_MINUTES})",
                },
                "cooldown_hours": {
                    "type": "number",
                    "description": (
                        "Cooldown en heures (min 1). À INFÉRER du contexte si non dit "
                        "(urgent→1, ranked/gaming→3, sans spam→6). Omettre = auto."
                    ),
                },
                "max_fires": {
                    "type": "integer",
                    "description": (
                        "Max alertes puis stop. À INFÉRER : « une fois»→1, ce soir→3, "
                        "défaut 5. Omettre = auto."
                    ),
                },
                "ttl_days": {
                    "type": "integer",
                    "description": (
                        f"Durée de vie 1–{EVENT_TTL_MAX_DAYS} j. À INFÉRER : ce soir→1, "
                        f"semaine→7, promo/long→14–21, défaut {EVENT_TTL_DEFAULT_DAYS}. Omettre = auto."
                    ),
                },
                "guild_scope": {
                    "type": "boolean",
                    "description": "Écoute tout le serveur (modos + demande explicite)",
                },
                "explicit_guild_scope": {
                    "type": "boolean",
                    "description": "true seulement si le membre a dit clairement « tout le serveur »",
                },
                "author": {
                    "type": "string",
                    "enum": ["any", "not_self", "self"],
                    "description": (
                        "Qui déclenche : self = le membre lui-même (« quand JE dis… »), "
                        "not_self = les autres (défaut), any = tout le monde. "
                        "À INFÉRER : « quand je dis X » → self."
                    ),
                },
            },
            optional_props=[
                "title", "execute_at", "delay_minutes", "delay_hours", "recurrence",
                "weekdays", "time", "until", "via", "kind", "pattern", "keyword",
                "aliases", "drop_percent", "topic",
                "url", "threshold", "op", "interval_minutes", "cooldown_hours",
                "max_fires", "ttl_days", "guild_scope", "explicit_guild_scope", "author",
            ],
            function=_tool_schedule,
        ),
        Tool(
            name="manage_task",
            description=(
                "Gère tes tâches : list, edit, confirm, pause, resume, skip, cancel, cancel_all. "
                "confirm = active le brouillon écoute/veille quand le membre dit oui/ok/vas-y "
                "(sans task_id : le dernier brouillon). cancel sur un brouillon = il refuse. "
                "Edit event/watch (même brouillon) : pattern, cooldown_hours, max_fires, "
                "ttl_days, threshold. Si le membre dit « plutôt moins/plus… », édite toi-même. "
                "Refus explicites si quotas / champs hors limites."
            ),
            properties={
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "edit", "confirm", "pause", "resume", "skip", "cancel", "cancel_all",
                    ],
                    "description": "Action à faire",
                },
                "task_id": {"type": "integer", "description": "ID #n"},
                "query": {"type": "string", "description": "Mots pour viser la tâche"},
                "instruction": {"type": "string", "description": "Nouvelle consigne (edit)"},
                "execute_at": {"type": "string", "description": "Nouvelle date ISO"},
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_SCHEDULES),
                    "description": "once|daily|weekly",
                },
                "weekdays": {"type": "string", "description": "Jours weekly"},
                "time": {"type": "string", "description": "HH:MM Paris"},
                "until": {"type": "string", "description": "Fin de série ISO"},
                "via": {
                    "type": "string",
                    "enum": ["channel", "dm"],
                    "description": "dm seulement si demandé explicitement",
                },
                "pattern": {"type": "string", "description": "Nouveau mot-clé (event)"},
                "cooldown_hours": {"type": "number", "description": "Nouveau cooldown (event)"},
                "max_fires": {"type": "integer", "description": "Nouveau max alertes"},
                "ttl_days": {"type": "integer", "description": "Nouvelle durée de vie en jours (event/watch)"},
                "threshold": {"type": "number", "description": "Nouveau seuil € (watch)"},
            },
            optional_props=[
                "task_id", "query", "instruction", "execute_at", "recurrence",
                "weekdays", "time", "until", "via", "pattern", "cooldown_hours",
                "max_fires", "ttl_days", "threshold",
            ],
            function=_tool_manage,
        ),
        Tool(
            name="show_tasks",
            description=(
                "Widget lecture seule des tâches d'une personne "
                "(défaut = auteur). Gestion → /taches ou manage_task."
            ),
            properties={
                "user_id": {"type": "string", "description": "Id Discord (optionnel)"},
                "username": {"type": "string", "description": "Pseudo Discord (optionnel)"},
            },
            optional_props=["user_id", "username"],
            function=_tool_show,
        ),
    ]
