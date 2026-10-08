"""Tâches planifiées — store SQLite + worker asyncio (1 exécution LLM à la fois)."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

from common.timezones import PARIS_TZ

logger = logging.getLogger("MARIA.Tasks")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "tasks.db"
OLD_RAPPELS_DB = DATA_DIR / "rappels.db"

SCHEDULE_ONCE = "once"
SCHEDULE_DAILY = "daily"
SCHEDULE_WEEKLY = "weekly"
VALID_SCHEDULES = (SCHEDULE_ONCE, SCHEDULE_DAILY, SCHEDULE_WEEKLY)

KIND_AT = "at"
KIND_RECURRING = "recurring"
KIND_EVENT = "event"
KIND_WATCH = "watch"
VALID_KINDS = (KIND_AT, KIND_RECURRING, KIND_EVENT, KIND_WATCH)

SCOPE_CHANNEL = "channel"
SCOPE_GUILD = "guild"

STATUS_PENDING = "pending"
STATUS_PAUSED = "paused"
STATUS_RUNNING = "running"
STATUS_ARMED = "armed"
STATUS_DRAFT = "draft"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_PAUSED, STATUS_RUNNING, STATUS_ARMED)
QUOTA_STATUSES = (STATUS_PENDING, STATUS_PAUSED, STATUS_ARMED, STATUS_DRAFT)

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAYS_FR = {
    "mon": "lundi",
    "tue": "mardi",
    "wed": "mercredi",
    "thu": "jeudi",
    "fri": "vendredi",
    "sat": "samedi",
    "sun": "dimanche",
}

MAX_SEND_RETRIES = 3
# Backoff croissant appliqué par TaskStore.retry_later, indexé par le compteur
# de retries après incrément (1ère → 2e → 3e tentative).
_RETRY_BACKOFF_SECONDS = {1: 30, 2: 120, 3: 300}
_RETRY_BACKOFF_SECONDS_MAX = 300
TASK_MAX_PENDING = 8
TASK_MAX_RECURRING = 2
TASK_MAX_EVENT = 2
TASK_MAX_WATCH = 1
TASK_MAX_EVENT_CHANNELS = 2
TASK_MAX_GUILD_SCOPE_PER_USER = 1
TASK_MAX_GUILD_SCOPE_PER_GUILD = 3
EVENT_COOLDOWN_MIN = 3600
EVENT_COOLDOWN_DEFAULT = 3 * 3600
EVENT_MAX_FIRES_DEFAULT = 5
EVENT_TTL_DEFAULT_DAYS = 7
EVENT_TTL_MAX_DAYS = 30
WATCH_INTERVAL_MIN_MINUTES = 6 * 60
WATCH_WEB_BUDGET_GUILD_DAY = 40
FIRE_STORM_MAX_PER_HOUR = 3
DRAFT_TTL_MINUTES = 10
MEMBER_VARS_MAX_KEYS = 20
MEMBER_VARS_MAX_BYTES = 1024
MEMBER_VARS_TTL_DAYS = 30
# Events : pas claimés par l'horloge.
EVENT_SENTINEL_AT = datetime(2099, 1, 1, tzinfo=timezone.utc)
TASK_MIN_MINUTES = 1
TASK_MIN_SECONDS = 45
TASK_MAX_DAYS = 365
TASK_INSTRUCTION_MAX = 500
TASK_TITLE_MAX = 80
TASK_RUN_KEEP = 45
TASK_RUN_SUMMARY_MAX = 280


@dataclass
class ScheduledTask:
    id: int
    channel_id: int
    user_id: int
    guild_id: int
    instruction: str
    execute_at: datetime
    title: str = ""
    schedule_kind: str = SCHEDULE_ONCE
    weekdays: list[str] = field(default_factory=list)
    time_of_day: str = ""
    until_at: Optional[datetime] = None
    status: str = STATUS_PENDING
    retries: int = 0
    last_error: str = ""
    message_id: int = 0
    created_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    deliver_dm: bool = False
    kind: str = KIND_AT
    trigger_json: str = "{}"
    recipe_json: str = "{}"
    scope: str = SCOPE_CHANNEL
    channel_ids_json: str = "[]"
    cooldown_seconds: int = 0
    max_fires: int = 0
    fires_count: int = 0
    expires_at: Optional[datetime] = None
    last_fired_at: Optional[datetime] = None

    @property
    def channel_ids(self) -> list[int]:
        try:
            raw = json.loads(self.channel_ids_json or "[]")
            if isinstance(raw, list) and raw:
                return [int(x) for x in raw]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return [self.channel_id] if self.channel_id else []

    @property
    def trigger(self) -> dict:
        try:
            data = json.loads(self.trigger_json or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    @property
    def recipe(self) -> dict:
        try:
            data = json.loads(self.recipe_json or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_optional_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    return _as_utc(datetime.fromisoformat(raw))


def normalize_weekdays(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [p.strip().lower()[:3] for p in raw.replace(";", ",").split(",") if p.strip()]
    out: list[str] = []
    seen: set[str] = set()
    aliases = {
        "lundi": "mon", "mardi": "tue", "mercredi": "wed", "jeudi": "thu",
        "vendredi": "fri", "samedi": "sat", "dimanche": "sun",
        "monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
        "friday": "fri", "saturday": "sat", "sunday": "sun",
    }
    for item in raw:
        token = str(item).strip().lower()
        token = aliases.get(token, token[:3])
        if token in WEEKDAYS and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def normalize_time_of_day(raw: str, fallback: Optional[datetime] = None) -> str:
    text = (raw or "").strip()
    if text:
        try:
            parts = text.replace("h", ":").split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
        except (ValueError, IndexError):
            pass
    if fallback is not None:
        local = _as_utc(fallback).astimezone(PARIS_TZ)
        return local.strftime("%H:%M")
    return "09:00"


def _parse_hhmm(time_of_day: str) -> tuple[int, int]:
    h, m = time_of_day.split(":")
    return int(h), int(m)


def format_schedule(task: ScheduledTask) -> str:
    """Libellé français court (UI)."""
    if task.schedule_kind == SCHEDULE_ONCE:
        return "une fois"
    tod = task.time_of_day or "—"
    if task.schedule_kind == SCHEDULE_DAILY:
        return f"tous les jours à {tod}"
    days = task.weekdays or []
    labels = [WEEKDAYS_FR.get(d, d) for d in days]
    if not labels:
        return f"hebdo à {tod}"
    if len(labels) == 1:
        return f"chaque {labels[0]} à {tod}"
    return f"{' et '.join(labels)} à {tod}"


def next_occurrence(
    task: ScheduledTask,
    *,
    after: Optional[datetime] = None,
) -> Optional[datetime]:
    """Prochaine occurrence strictement après `after` (défaut: maintenant)."""
    after_utc = _as_utc(after or datetime.now(timezone.utc))
    until = _as_utc(task.until_at) if task.until_at else None
    kind = task.schedule_kind
    if kind == SCHEDULE_ONCE:
        return None

    tod = task.time_of_day or normalize_time_of_day("", task.execute_at)
    hour, minute = _parse_hhmm(tod)
    start = after_utc.astimezone(PARIS_TZ)

    if kind == SCHEDULE_DAILY:
        wanted = set(WEEKDAYS)
    else:
        wanted = set(task.weekdays) or {WEEKDAYS[start.weekday()]}

    day = start.date()
    for _ in range(400):
        cand_local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=PARIS_TZ)
        cand_utc = cand_local.astimezone(timezone.utc)
        wd = WEEKDAYS[cand_local.weekday()]
        if wd in wanted and cand_utc > after_utc:
            if until is not None and cand_utc > until:
                return None
            return cand_utc
        day = day + timedelta(days=1)
    return None


def snap_execute_at(
    *,
    kind: str,
    weekdays: Optional[list[str]] = None,
    time_of_day: str = "",
    execute_at: Optional[datetime] = None,
    until_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Première occurrence assez loin (≈1 min souple), calée sur l'heure / les jours.

    Une daily/weekly demandée à une heure déjà passée avance au prochain créneau
    au lieu d'échouer (« trop proche »).
    """
    now_utc = _as_utc(now or datetime.now(timezone.utc))
    min_at = now_utc + timedelta(seconds=TASK_MIN_SECONDS)
    if kind == SCHEDULE_ONCE or kind not in VALID_SCHEDULES:
        return _as_utc(execute_at) if execute_at is not None else None
    days = normalize_weekdays(weekdays)
    tod = normalize_time_of_day(time_of_day, execute_at)
    dummy = ScheduledTask(
        id=0,
        channel_id=0,
        user_id=0,
        guild_id=0,
        instruction="x",
        execute_at=_as_utc(execute_at) if execute_at is not None else min_at,
        schedule_kind=kind,
        weekdays=days,
        time_of_day=tod,
        until_at=until_at,
    )
    return next_occurrence(dummy, after=min_at - timedelta(seconds=1))


def _paris_date(dt: datetime):
    return _as_utc(dt).astimezone(PARIS_TZ).date()


def _end_of_paris_day(now: datetime) -> datetime:
    """Dernier instant (UTC) de la journée calendaire Paris de `now`."""
    local = _as_utc(now).astimezone(PARIS_TZ)
    start = datetime(local.year, local.month, local.day, tzinfo=PARIS_TZ)
    return (start + timedelta(days=1)).astimezone(timezone.utc) - timedelta(microseconds=1)


def already_ran_today(task: ScheduledTask, now: Optional[datetime] = None) -> bool:
    """Série (daily/weekly) déjà exécutée aujourd'hui (jour Paris) : pas de 2e passage."""
    if task.schedule_kind == SCHEDULE_ONCE or task.last_run_at is None:
        return False
    return _paris_date(task.last_run_at) == _paris_date(now or datetime.now(timezone.utc))


def _infer_kind(schedule_kind: str, kind: str) -> str:
    k = (kind or "").strip()
    if k in VALID_KINDS:
        return k
    if schedule_kind in (SCHEDULE_DAILY, SCHEDULE_WEEKLY):
        return KIND_RECURRING
    return KIND_AT


def _row_to_task(r: sqlite3.Row) -> ScheduledTask:
    keys = r.keys()
    sk = (r["schedule_kind"] if "schedule_kind" in keys else SCHEDULE_ONCE) or SCHEDULE_ONCE
    kind_raw = (r["kind"] if "kind" in keys else "") or ""
    return ScheduledTask(
        id=r["id"],
        channel_id=r["channel_id"],
        user_id=r["user_id"],
        guild_id=r["guild_id"] or 0,
        instruction=r["instruction"] or "",
        execute_at=_as_utc(datetime.fromisoformat(r["execute_at"])),
        title=(r["title"] if "title" in keys else "") or "",
        schedule_kind=sk,
        weekdays=normalize_weekdays(r["weekdays"] if "weekdays" in keys else "[]"),
        time_of_day=(r["time_of_day"] if "time_of_day" in keys else "") or "",
        until_at=_parse_optional_dt(r["until_at"] if "until_at" in keys else None),
        status=(r["status"] if "status" in keys else STATUS_PENDING) or STATUS_PENDING,
        retries=(r["retries"] if "retries" in keys else 0) or 0,
        last_error=(r["last_error"] if "last_error" in keys else "") or "",
        message_id=(r["message_id"] if "message_id" in keys else 0) or 0,
        created_at=_parse_optional_dt(r["created_at"] if "created_at" in keys else None),
        last_run_at=_parse_optional_dt(r["last_run_at"] if "last_run_at" in keys else None),
        deliver_dm=bool((r["deliver_dm"] if "deliver_dm" in keys else 0) or 0),
        kind=_infer_kind(sk, kind_raw),
        trigger_json=(r["trigger_json"] if "trigger_json" in keys else None) or "{}",
        recipe_json=(r["recipe_json"] if "recipe_json" in keys else None) or "{}",
        scope=(r["scope"] if "scope" in keys else None) or SCOPE_CHANNEL,
        channel_ids_json=(r["channel_ids"] if "channel_ids" in keys else None) or "[]",
        cooldown_seconds=int((r["cooldown_seconds"] if "cooldown_seconds" in keys else 0) or 0),
        max_fires=int((r["max_fires"] if "max_fires" in keys else 0) or 0),
        fires_count=int((r["fires_count"] if "fires_count" in keys else 0) or 0),
        expires_at=_parse_optional_dt(r["expires_at"] if "expires_at" in keys else None),
        last_fired_at=_parse_optional_dt(r["last_fired_at"] if "last_fired_at" in keys else None),
    )


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id   INTEGER NOT NULL,
                user_id      INTEGER NOT NULL,
                guild_id     INTEGER DEFAULT 0,
                title        TEXT DEFAULT '',
                instruction  TEXT NOT NULL,
                execute_at   TEXT NOT NULL,
                schedule_kind TEXT DEFAULT 'once',
                weekdays     TEXT DEFAULT '[]',
                time_of_day  TEXT DEFAULT '',
                until_at     TEXT,
                status       TEXT DEFAULT 'pending',
                retries      INTEGER DEFAULT 0,
                last_error   TEXT DEFAULT '',
                message_id   INTEGER DEFAULT 0,
                created_at   TEXT,
                last_run_at  TEXT,
                deliver_dm   INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        alters = {
            "deliver_dm": "INTEGER NOT NULL DEFAULT 0",
            "kind": "TEXT DEFAULT 'at'",
            "trigger_json": "TEXT DEFAULT '{}'",
            "recipe_json": "TEXT DEFAULT '{}'",
            "scope": "TEXT DEFAULT 'channel'",
            "channel_ids": "TEXT DEFAULT '[]'",
            "cooldown_seconds": "INTEGER DEFAULT 0",
            "max_fires": "INTEGER DEFAULT 0",
            "fires_count": "INTEGER DEFAULT 0",
            "expires_at": "TEXT",
            "last_fired_at": "TEXT",
        }
        for col, decl in alters.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {decl}")
        # Backfill kind depuis schedule_kind.
        conn.execute(
            "UPDATE tasks SET kind=? WHERE (kind IS NULL OR kind='' OR kind='at') "
            "AND schedule_kind IN (?, ?)",
            (KIND_RECURRING, SCHEDULE_DAILY, SCHEDULE_WEEKLY),
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_at ON tasks(status, execute_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_kind_status ON tasks(kind, status, guild_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS member_vars (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                key      TEXT NOT NULL,
                value_json TEXT NOT NULL DEFAULT '',
                updated_at TEXT,
                ttl_expires_at TEXT,
                PRIMARY KEY (guild_id, user_id, key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watch_web_budget (
                guild_id INTEGER NOT NULL,
                day      TEXT NOT NULL,
                fetches  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, day)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id  INTEGER NOT NULL,
                ran_at   TEXT NOT NULL,
                summary  TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_runs_task ON task_runs(task_id, ran_at)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        # Crash pendant une exécution → remettre en file.
        conn.execute(
            "UPDATE tasks SET status = ? WHERE status = ?",
            (STATUS_PENDING, STATUS_RUNNING),
        )
        migrated = conn.execute(
            "SELECT value FROM meta WHERE key = 'rappels_migrated'"
        ).fetchone()
        if not migrated:
            n = _migrate_rappels(conn)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('rappels_migrated', ?)",
                (str(n),),
            )
            if n:
                logger.info("Migration rappels → tâches : %d ligne(s)", n)


def _migrate_rappels(conn: sqlite3.Connection) -> int:
    if not OLD_RAPPELS_DB.exists():
        return 0
    old = sqlite3.connect(str(OLD_RAPPELS_DB), timeout=30.0)
    old.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "rappels" not in tables:
            return 0
        cols = {r[1] for r in old.execute("PRAGMA table_info(rappels)").fetchall()}
        rows = old.execute(
            "SELECT * FROM rappels WHERE status IN ('pending', 'failed')"
        ).fetchall()
        count = 0
        for r in rows:
            execute_at = _as_utc(datetime.fromisoformat(r["execute_at"]))
            rec = (r["recurrence"] if "recurrence" in cols else "none") or "none"
            if rec == "daily":
                kind = SCHEDULE_DAILY
            elif rec == "weekly":
                kind = SCHEDULE_WEEKLY
            else:
                kind = SCHEDULE_ONCE
            until = None
            if "recurrence_until" in cols and r["recurrence_until"]:
                until = r["recurrence_until"]
            local = execute_at.astimezone(PARIS_TZ)
            weekdays = json.dumps([WEEKDAYS[local.weekday()]]) if kind == SCHEDULE_WEEKLY else "[]"
            time_of_day = local.strftime("%H:%M") if kind != SCHEDULE_ONCE else ""
            desc = (r["description"] or "").strip()
            instruction = f"Rappelle : {desc}" if desc else "Rappelle à l'utilisateur."
            title = desc[:TASK_TITLE_MAX]
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO tasks (
                    id, channel_id, user_id, guild_id, title, instruction, execute_at,
                    schedule_kind, weekdays, time_of_day, until_at, status, retries,
                    last_error, message_id, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    r["id"],
                    r["channel_id"],
                    r["user_id"],
                    0,
                    title,
                    instruction,
                    execute_at.isoformat(),
                    kind,
                    weekdays,
                    time_of_day,
                    until,
                    STATUS_PENDING if r["status"] == "pending" else STATUS_FAILED,
                    (r["retries"] if "retries" in cols else 0) or 0,
                    "",
                    r["message_id"] or 0,
                    now_iso,
                ),
            )
            count += 1
        max_id = conn.execute("SELECT MAX(id) FROM tasks").fetchone()[0]
        if max_id:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO sqlite_sequence(name, seq) VALUES ('tasks', ?)",
                    (max_id,),
                )
            except sqlite3.Error:
                pass
        return count
    except sqlite3.Error as e:
        logger.error("Migration rappels échouée : %s", e)
        return 0
    finally:
        old.close()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class TaskStore:
    def __init__(self):
        _init_db()

    def add(
        self,
        *,
        channel_id: int,
        user_id: int,
        guild_id: int,
        instruction: str,
        execute_at: datetime,
        title: str = "",
        schedule_kind: str = SCHEDULE_ONCE,
        weekdays: Optional[list[str]] = None,
        time_of_day: str = "",
        until_at: Optional[datetime] = None,
        message_id: int = 0,
        deliver_dm: bool = False,
        kind: str = KIND_AT,
        trigger: Optional[dict] = None,
        recipe: Optional[dict] = None,
        scope: str = SCOPE_CHANNEL,
        channel_ids: Optional[list[int]] = None,
        cooldown_seconds: int = 0,
        max_fires: int = 0,
        expires_at: Optional[datetime] = None,
        status: str = STATUS_PENDING,
    ) -> int:
        if schedule_kind not in VALID_SCHEDULES:
            schedule_kind = SCHEDULE_ONCE
        kind = kind if kind in VALID_KINDS else _infer_kind(schedule_kind, kind)
        if kind == KIND_EVENT:
            execute_at = EVENT_SENTINEL_AT
            status = status if status in (STATUS_DRAFT, STATUS_ARMED, STATUS_PAUSED) else STATUS_ARMED
            schedule_kind = SCHEDULE_ONCE
        elif kind == KIND_WATCH:
            schedule_kind = SCHEDULE_ONCE
            status = status if status in (STATUS_DRAFT, STATUS_PENDING, STATUS_PAUSED) else STATUS_PENDING
        execute_at = _as_utc(execute_at)
        days = normalize_weekdays(weekdays)
        if schedule_kind == SCHEDULE_WEEKLY and not days:
            days = [WEEKDAYS[execute_at.astimezone(PARIS_TZ).weekday()]]
        tod = ""
        if schedule_kind != SCHEDULE_ONCE and kind == KIND_RECURRING:
            tod = normalize_time_of_day(time_of_day, execute_at)
            snapped = snap_execute_at(
                kind=schedule_kind,
                weekdays=days,
                time_of_day=tod,
                execute_at=execute_at,
                until_at=until_at,
            )
            if snapped is not None:
                execute_at = snapped
        title = (title or instruction).strip()[:TASK_TITLE_MAX]
        instruction = (instruction or "").strip()[:TASK_INSTRUCTION_MAX]
        now_iso = datetime.now(timezone.utc).isoformat()
        until_iso = _as_utc(until_at).isoformat() if until_at else None
        exp_iso = _as_utc(expires_at).isoformat() if expires_at else None
        chans = channel_ids if channel_ids else [channel_id]
        with _db() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (
                    channel_id, user_id, guild_id, title, instruction, execute_at,
                    schedule_kind, weekdays, time_of_day, until_at, status,
                    message_id, created_at, deliver_dm,
                    kind, trigger_json, recipe_json, scope, channel_ids,
                    cooldown_seconds, max_fires, fires_count, expires_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    channel_id, user_id, guild_id, title, instruction,
                    execute_at.isoformat(), schedule_kind, json.dumps(days),
                    tod, until_iso, status, message_id, now_iso,
                    1 if deliver_dm else 0,
                    kind,
                    json.dumps(trigger or {}, ensure_ascii=False),
                    json.dumps(recipe or {}, ensure_ascii=False),
                    scope if scope in (SCOPE_CHANNEL, SCOPE_GUILD) else SCOPE_CHANNEL,
                    json.dumps([int(c) for c in chans]),
                    int(cooldown_seconds or 0),
                    int(max_fires or 0),
                    0,
                    exp_iso,
                ),
            )
            return int(cur.lastrowid)

    def get(self, task_id: int) -> Optional[ScheduledTask]:
        with _db() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def get_user_tasks(self, user_id: int) -> list[ScheduledTask]:
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE user_id=? AND status IN (?, ?, ?, ?, ?)
                ORDER BY execute_at
                """,
                (
                    user_id, STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED,
                    STATUS_ARMED, STATUS_DRAFT,
                ),
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def count_active(self, user_id: int, *, exclude_id: int = 0) -> int:
        with _db() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND id != ? AND status IN ({','.join('?' * len(QUOTA_STATUSES))})
                """,
                (user_id, exclude_id, *QUOTA_STATUSES),
            ).fetchone()
        return row[0] if row else 0

    def count_active_recurring(self, user_id: int, *, exclude_id: int = 0) -> int:
        with _db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND status IN (?, ?, ?)
                  AND (kind=? OR schedule_kind IN (?, ?))
                  AND id != ?
                """,
                (
                    user_id, STATUS_PENDING, STATUS_PAUSED, STATUS_ARMED,
                    KIND_RECURRING, SCHEDULE_DAILY, SCHEDULE_WEEKLY, exclude_id,
                ),
            ).fetchone()
        return row[0] if row else 0

    def count_kind(self, user_id: int, kind: str, *, exclude_id: int = 0) -> int:
        with _db() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND kind=? AND id != ?
                  AND status IN ({','.join('?' * len(QUOTA_STATUSES))})
                """,
                (user_id, kind, exclude_id, *QUOTA_STATUSES),
            ).fetchone()
        return row[0] if row else 0

    def count_guild_scope_events(self, guild_id: int, *, user_id: int = 0) -> tuple[int, int]:
        """(total guild-scope events actifs, ceux du user)."""
        with _db() as conn:
            total = conn.execute(
                f"""
                SELECT COUNT(*) FROM tasks
                WHERE guild_id=? AND kind=? AND scope=?
                  AND status IN ({','.join('?' * len(QUOTA_STATUSES))})
                """,
                (guild_id, KIND_EVENT, SCOPE_GUILD, *QUOTA_STATUSES),
            ).fetchone()[0]
            mine = 0
            if user_id:
                mine = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM tasks
                    WHERE guild_id=? AND user_id=? AND kind=? AND scope=?
                      AND status IN ({','.join('?' * len(QUOTA_STATUSES))})
                    """,
                    (guild_id, user_id, KIND_EVENT, SCOPE_GUILD, *QUOTA_STATUSES),
                ).fetchone()[0]
        return int(total or 0), int(mine or 0)

    def quota_summary(self, user_id: int) -> dict[str, int]:
        tasks = self.get_user_tasks(user_id)
        active = [t for t in tasks if t.status in QUOTA_STATUSES]
        return {
            "total": len(active),
            "max_total": TASK_MAX_PENDING,
            "event": sum(1 for t in active if t.kind == KIND_EVENT),
            "max_event": TASK_MAX_EVENT,
            "watch": sum(1 for t in active if t.kind == KIND_WATCH),
            "max_watch": TASK_MAX_WATCH,
            "recurring": sum(
                1 for t in active
                if t.kind == KIND_RECURRING or t.schedule_kind in (SCHEDULE_DAILY, SCHEDULE_WEEKLY)
            ),
            "max_recurring": TASK_MAX_RECURRING,
        }

    def list_armed_events(self, guild_id: int) -> list[ScheduledTask]:
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE guild_id=? AND kind=? AND status=?
                """,
                (guild_id, KIND_EVENT, STATUS_ARMED),
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def latest_draft(self, user_id: int, kind: Optional[str] = None) -> Optional[ScheduledTask]:
        sql = "SELECT * FROM tasks WHERE user_id=? AND status=?"
        params: list = [user_id, STATUS_DRAFT]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        with _db() as conn:
            row = conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()
        return _row_to_task(row) if row else None

    def cancel_drafts(self, user_id: int, kind: str) -> int:
        """Un nouveau brouillon remplace les anciens (pas de quota gâché par un doublon)."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE user_id=? AND kind=? AND status=?",
                (STATUS_CANCELLED, user_id, kind, STATUS_DRAFT),
            )
            return cur.rowcount

    def find_event_duplicate(
        self, user_id: int, guild_id: int, pattern: str,
    ) -> Optional[ScheduledTask]:
        """Écoute active du même membre sur le même mot-clé (hors brouillon)."""
        want = (pattern or "").strip().casefold()
        if not want:
            return None
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE user_id=? AND guild_id=? AND kind=? AND status IN (?, ?)
                """,
                (user_id, guild_id, KIND_EVENT, STATUS_ARMED, STATUS_PAUSED),
            ).fetchall()
        for row in rows:
            t = _row_to_task(row)
            key = t.trigger.get("pattern") or t.trigger.get("topic") or ""
            if str(key).strip().casefold() == want:
                return t
        return None

    def expire_overdue(self) -> int:
        """Termine les écoutes / veilles dont la durée de vie est dépassée (libère le quota)."""
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            cur = conn.execute(
                """
                UPDATE tasks SET status=?, last_run_at=?
                WHERE kind IN (?, ?) AND status IN (?, ?, ?)
                  AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (
                    STATUS_COMPLETED, now, KIND_EVENT, KIND_WATCH,
                    STATUS_ARMED, STATUS_PENDING, STATUS_PAUSED, now,
                ),
            )
            return cur.rowcount

    def confirm_draft(self, task_id: int, user_id: int) -> bool:
        task = self.get(task_id)
        if task is None or task.user_id != user_id or task.status != STATUS_DRAFT:
            return False
        new_status = STATUS_ARMED if task.kind == KIND_EVENT else STATUS_PENDING
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE id=? AND user_id=? AND status=?",
                (new_status, task_id, user_id, STATUS_DRAFT),
            )
            return cur.rowcount > 0

    def purge_expired_drafts(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=DRAFT_TTL_MINUTES)).isoformat()
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE status=? AND created_at < ?",
                (STATUS_CANCELLED, STATUS_DRAFT, cutoff),
            )
            return cur.rowcount

    def touch_last_fired(self, task_id: int) -> None:
        """Pose last_fired_at (anti-storm) sans incrémenter fires_count."""
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                "UPDATE tasks SET last_fired_at=? WHERE id=?",
                (now, task_id),
            )

    def record_fire(self, task_id: int) -> Optional[ScheduledTask]:
        """Incrémente fires ; complete si max atteint ou expire. Retourne tâche à jour."""
        now = datetime.now(timezone.utc)
        with _db() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return None
            task = _row_to_task(row)
            fires = task.fires_count + 1
            done = False
            if task.max_fires and fires >= task.max_fires:
                done = True
            if task.expires_at and now >= task.expires_at:
                done = True
            status = STATUS_COMPLETED if done else (
                STATUS_ARMED if task.kind == KIND_EVENT else STATUS_PENDING
            )
            next_at = task.execute_at
            if task.kind == KIND_WATCH and not done:
                mins = int(task.trigger.get("interval_minutes") or WATCH_INTERVAL_MIN_MINUTES)
                next_at = now + timedelta(minutes=max(WATCH_INTERVAL_MIN_MINUTES, mins))
            conn.execute(
                """
                UPDATE tasks SET fires_count=?, last_fired_at=?, last_run_at=?,
                    status=?, execute_at=?, retries=0, last_error=''
                WHERE id=?
                """,
                (
                    fires, now.isoformat(), now.isoformat(),
                    status, _as_utc(next_at).isoformat(), task_id,
                ),
            )
        return self.get(task_id)

    def reschedule_watch(self, task_id: int, *, fired: bool) -> Optional[ScheduledTask]:
        """Après un poll watch : alerte (fired) ou simple report du prochain check."""
        if fired:
            return self.record_fire(task_id)
        now = datetime.now(timezone.utc)
        task = self.get(task_id)
        if task is None:
            return None
        if task.expires_at and now >= task.expires_at:
            with _db() as conn:
                conn.execute(
                    "UPDATE tasks SET status=?, last_run_at=? WHERE id=?",
                    (STATUS_COMPLETED, now.isoformat(), task_id),
                )
            return self.get(task_id)
        mins = int(task.trigger.get("interval_minutes") or WATCH_INTERVAL_MIN_MINUTES)
        next_at = now + timedelta(minutes=max(WATCH_INTERVAL_MIN_MINUTES, mins))
        with _db() as conn:
            conn.execute(
                """
                UPDATE tasks SET status=?, execute_at=?, last_run_at=?,
                    retries=0, last_error=''
                WHERE id=?
                """,
                (STATUS_PENDING, next_at.isoformat(), now.isoformat(), task_id),
            )
        return self.get(task_id)

    def fires_last_hour(self, user_id: int) -> int:
        since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with _db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND kind IN (?, ?) AND last_fired_at >= ?
                """,
                (user_id, KIND_EVENT, KIND_WATCH, since),
            ).fetchone()
        return int(row[0] if row else 0)

    def consume_watch_budget(self, guild_id: int, *, n: int = 1) -> bool:
        """True si le fetch est autorisé (incrémente le compteur jour Paris)."""
        day = datetime.now(timezone.utc).astimezone(PARIS_TZ).strftime("%Y-%m-%d")
        with _db() as conn:
            row = conn.execute(
                "SELECT fetches FROM watch_web_budget WHERE guild_id=? AND day=?",
                (guild_id, day),
            ).fetchone()
            cur = int(row[0] if row else 0)
            if cur + n > WATCH_WEB_BUDGET_GUILD_DAY:
                return False
            conn.execute(
                """
                INSERT INTO watch_web_budget(guild_id, day, fetches) VALUES (?,?,?)
                ON CONFLICT(guild_id, day) DO UPDATE SET fetches = fetches + ?
                """,
                (guild_id, day, n, n),
            )
        return True

    def get_var(self, guild_id: int, user_id: int, key: str) -> Optional[str]:
        with _db() as conn:
            row = conn.execute(
                "SELECT value_json, ttl_expires_at FROM member_vars "
                "WHERE guild_id=? AND user_id=? AND key=?",
                (guild_id, user_id, key),
            ).fetchone()
        if not row:
            return None
        exp = _parse_optional_dt(row["ttl_expires_at"])
        if exp and exp < datetime.now(timezone.utc):
            self.delete_var(guild_id, user_id, key)
            return None
        return row["value_json"]

    def set_var(self, guild_id: int, user_id: int, key: str, value: str) -> str | None:
        """None si OK, sinon erreur FR."""
        key = (key or "").strip()[:80]
        if not key:
            return "Clé vide."
        val = (value or "")[:MEMBER_VARS_MAX_BYTES]
        low = val.casefold()
        if any(s in low for s in ("api_key", "token", "bearer ", "sk-", "password")):
            return "Valeur refusée (ressemble à un secret)."
        with _db() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM member_vars WHERE guild_id=? AND user_id=?",
                (guild_id, user_id),
            ).fetchone()[0]
            exists = conn.execute(
                "SELECT 1 FROM member_vars WHERE guild_id=? AND user_id=? AND key=?",
                (guild_id, user_id, key),
            ).fetchone()
            if not exists and n >= MEMBER_VARS_MAX_KEYS:
                return f"Limite de variables atteinte ({MEMBER_VARS_MAX_KEYS})."
            ttl = (
                datetime.now(timezone.utc) + timedelta(days=MEMBER_VARS_TTL_DAYS)
            ).isoformat()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO member_vars(guild_id, user_id, key, value_json, updated_at, ttl_expires_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(guild_id, user_id, key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at,
                    ttl_expires_at=excluded.ttl_expires_at
                """,
                (guild_id, user_id, key, val, now, ttl),
            )
        return None

    def delete_var(self, guild_id: int, user_id: int, key: str) -> None:
        with _db() as conn:
            conn.execute(
                "DELETE FROM member_vars WHERE guild_id=? AND user_id=? AND key=?",
                (guild_id, user_id, key),
            )

    def list_vars(self, guild_id: int, user_id: int) -> list[tuple[str, str]]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT key, value_json, ttl_expires_at FROM member_vars "
                "WHERE guild_id=? AND user_id=? ORDER BY key",
                (guild_id, user_id),
            ).fetchall()
        out: list[tuple[str, str]] = []
        now = datetime.now(timezone.utc)
        for r in rows:
            exp = _parse_optional_dt(r["ttl_expires_at"])
            if exp and exp < now:
                continue
            out.append((r["key"], r["value_json"]))
        return out

    def list_runs(self, task_id: int, *, limit: int = TASK_RUN_KEEP) -> list[tuple[datetime, str]]:
        """Exécutions passées, plus anciennes d'abord."""
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT ran_at, summary FROM task_runs
                WHERE task_id=?
                ORDER BY ran_at DESC
                LIMIT ?
                """,
                (task_id, max(1, min(limit, TASK_RUN_KEEP))),
            ).fetchall()
        out: list[tuple[datetime, str]] = []
        for r in reversed(rows):
            dt = _parse_optional_dt(r["ran_at"])
            if dt is None:
                continue
            summary = (r["summary"] or "").strip()
            if summary:
                out.append((dt, summary))
        return out

    def append_run(self, task_id: int, summary: str) -> None:
        text = (summary or "").strip()[:TASK_RUN_SUMMARY_MAX]
        if not text:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                "INSERT INTO task_runs (task_id, ran_at, summary) VALUES (?,?,?)",
                (task_id, now_iso, text),
            )
            rows = conn.execute(
                "SELECT id FROM task_runs WHERE task_id=? ORDER BY ran_at DESC, id DESC",
                (task_id,),
            ).fetchall()
            extra = [r["id"] for r in rows[TASK_RUN_KEEP:]]
            if extra:
                conn.execute(
                    f"DELETE FROM task_runs WHERE id IN ({','.join('?' * len(extra))})",
                    extra,
                )

    def claim_due(self) -> Optional[ScheduledTask]:
        """Passe la plus ancienne tâche due en running. None si rien.

        Garde-fou : une tâche répétitive déjà exécutée aujourd'hui (jour Paris) n'est jamais
        relancée dans la même journée (édition d'heure, reprise, rattrapage…) : elle est
        repoussée à sa prochaine occurrence des jours suivants.
        """
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        with _db() as conn:
            for _ in range(50):
                row = conn.execute(
                    """
                    SELECT * FROM tasks
                    WHERE status=? AND execute_at <= ? AND kind != ?
                    ORDER BY execute_at LIMIT 1
                    """,
                    (STATUS_PENDING, now, KIND_EVENT),
                ).fetchone()
                if not row:
                    return None
                task = _row_to_task(row)
                if task.retries == 0 and already_ran_today(task, now_dt):
                    nxt = next_occurrence(task, after=_end_of_paris_day(now_dt))
                    if nxt is None:
                        conn.execute(
                            "UPDATE tasks SET status=? WHERE id=?",
                            (STATUS_COMPLETED, task.id),
                        )
                    else:
                        conn.execute(
                            "UPDATE tasks SET execute_at=? WHERE id=?",
                            (nxt.isoformat(), task.id),
                        )
                    logger.info(
                        "Tâche #%s déjà exécutée aujourd'hui : reportée au %s",
                        task.id, nxt.isoformat() if nxt else "— (terminée)",
                    )
                    continue
                cur = conn.execute(
                    "UPDATE tasks SET status=? WHERE id=? AND status=?",
                    (STATUS_RUNNING, row["id"], STATUS_PENDING),
                )
                if cur.rowcount == 0:
                    return None
                break
            else:
                return None
        return self.get(row["id"])

    def get_next_due_at(self) -> Optional[datetime]:
        with _db() as conn:
            row = conn.execute(
                "SELECT execute_at FROM tasks WHERE status=? ORDER BY execute_at LIMIT 1",
                (STATUS_PENDING,),
            ).fetchone()
        return _as_utc(datetime.fromisoformat(row[0])) if row else None

    def mark_done(self, task_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, last_run_at=?, retries=0, last_error='' "
                "WHERE id=? AND status=?",
                (STATUS_COMPLETED, now, task_id, STATUS_RUNNING),
            )

    def mark_failed(self, task_id: int, error: str = "") -> None:
        with _db() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, last_error=? WHERE id=?",
                (STATUS_FAILED, (error or "")[:300], task_id),
            )

    def reschedule_after_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Après un fire réussi : prochaine occ. ou completed. Retourne la date ou None."""
        now = datetime.now(timezone.utc)
        # Série : la prochaine occurrence est toujours un jour (Paris) ultérieur.
        nxt = next_occurrence(task, after=_end_of_paris_day(now))
        now_iso = now.isoformat()
        with _db() as conn:
            if nxt is None:
                conn.execute(
                    "UPDATE tasks SET status=?, last_run_at=?, retries=0, last_error='' "
                    "WHERE id=?",
                    (STATUS_COMPLETED, now_iso, task.id),
                )
                return None
            conn.execute(
                """
                UPDATE tasks SET status=?, execute_at=?, last_run_at=?,
                    retries=0, last_error=''
                WHERE id=?
                """,
                (STATUS_PENDING, nxt.isoformat(), now_iso, task.id),
            )
        return nxt

    def retry_later(self, task_id: int, error: str) -> int:
        """Remet pending + incrément retries, avec un backoff croissant sur execute_at.

        Sans ce délai, `TaskWorker._loop` reclaim immédiatement une tâche due après un
        échec : les MAX_SEND_RETRIES tentatives s'enchaîneraient quasi instantanément.
        """
        with _db() as conn:
            conn.execute(
                """
                UPDATE tasks SET status=?, retries=retries+1, last_error=?
                WHERE id=?
                """,
                (STATUS_PENDING, (error or "")[:300], task_id),
            )
            row = conn.execute("SELECT retries FROM tasks WHERE id=?", (task_id,)).fetchone()
            retries = row[0] if row else MAX_SEND_RETRIES
            delay_seconds = _RETRY_BACKOFF_SECONDS.get(retries, _RETRY_BACKOFF_SECONDS_MAX)
            next_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            conn.execute(
                "UPDATE tasks SET execute_at=? WHERE id=?",
                (next_at.isoformat(), task_id),
            )
        return retries

    def edit(
        self,
        task_id: int,
        user_id: int,
        *,
        instruction: Optional[str] = None,
        title: Optional[str] = None,
        execute_at: Optional[datetime] = None,
        schedule_kind: Optional[str] = None,
        weekdays: Optional[list[str]] = None,
        time_of_day: Optional[str] = None,
        until_at: Optional[datetime] = None,
        clear_until: bool = False,
        deliver_dm: Optional[bool] = None,
        pattern: Optional[str] = None,
        cooldown_seconds: Optional[int] = None,
        max_fires: Optional[int] = None,
        threshold: Optional[float] = None,
        ttl_days: Optional[int] = None,
    ) -> bool:
        current = self.get(task_id)
        if current is None or current.user_id != user_id:
            return False
        if current.status not in (
            STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED, STATUS_ARMED, STATUS_DRAFT,
        ):
            return False
        sets: list[str] = []
        params: list[object] = []
        if instruction is not None:
            sets.append("instruction=?")
            params.append(instruction.strip()[:TASK_INSTRUCTION_MAX])
            if not title:
                sets.append("title=?")
                params.append(instruction.strip()[:TASK_TITLE_MAX])
        instr_changed = (
            instruction is not None
            and instruction.strip() != (current.instruction or "").strip()
        )
        if title is not None:
            sets.append("title=?")
            params.append(title.strip()[:TASK_TITLE_MAX])
        new_exec = _as_utc(execute_at) if execute_at is not None else None
        kind = schedule_kind
        if kind is not None and kind not in VALID_SCHEDULES:
            kind = None
        days = normalize_weekdays(weekdays) if weekdays is not None else None
        effective_kind = kind or current.schedule_kind
        will_snap = effective_kind != SCHEDULE_ONCE and (
            new_exec is not None or time_of_day is not None or days is not None or kind is not None
        )
        if new_exec is not None and not will_snap:
            sets.append("execute_at=?")
            params.append(new_exec.isoformat())
            sets.append("retries=?")
            params.append(0)
            if current.status == STATUS_FAILED:
                sets.append("status=?")
                params.append(STATUS_PENDING)
        if kind is not None:
            sets.append("schedule_kind=?")
            params.append(kind)
        if days is not None:
            sets.append("weekdays=?")
            params.append(json.dumps(days))
        if time_of_day is not None:
            sets.append("time_of_day=?")
            params.append(normalize_time_of_day(time_of_day, new_exec or current.execute_at))
        if clear_until:
            sets.append("until_at=?")
            params.append(None)
        elif until_at is not None:
            sets.append("until_at=?")
            params.append(_as_utc(until_at).isoformat())
        if deliver_dm is not None:
            sets.append("deliver_dm=?")
            params.append(1 if deliver_dm else 0)
        if cooldown_seconds is not None:
            sets.append("cooldown_seconds=?")
            params.append(max(EVENT_COOLDOWN_MIN, int(cooldown_seconds)))
        if max_fires is not None:
            sets.append("max_fires=?")
            params.append(max(1, int(max_fires)))
        if ttl_days is not None and current.kind in (KIND_EVENT, KIND_WATCH):
            days = max(1, min(EVENT_TTL_MAX_DAYS, int(ttl_days)))
            sets.append("expires_at=?")
            params.append((datetime.now(timezone.utc) + timedelta(days=days)).isoformat())
        trig = dict(current.trigger)
        trig_changed = False
        if pattern is not None and current.kind == KIND_EVENT:
            trig["pattern"] = str(pattern).strip()[:24]
            trig_changed = True
        if threshold is not None and current.kind == KIND_WATCH:
            trig["threshold"] = float(threshold)
            trig_changed = True
        if trig_changed:
            sets.append("trigger_json=?")
            params.append(json.dumps(trig, ensure_ascii=False))
        if will_snap:
            snapped = snap_execute_at(
                kind=effective_kind,
                weekdays=days if days is not None else current.weekdays,
                time_of_day=(
                    normalize_time_of_day(time_of_day, new_exec or current.execute_at)
                    if time_of_day is not None else current.time_of_day
                ),
                execute_at=new_exec or current.execute_at,
                until_at=(
                    None if clear_until else (until_at if until_at is not None else current.until_at)
                ),
            )
            if snapped is not None:
                sets.append("execute_at=?")
                params.append(snapped.isoformat())
                sets.append("retries=?")
                params.append(0)
                if current.status == STATUS_FAILED:
                    sets.append("status=?")
                    params.append(STATUS_PENDING)
        if not sets:
            return False
        params.extend([task_id, user_id])
        with _db() as conn:
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id=? AND user_id=?",
                params,
            )
            if cur.rowcount > 0 and instr_changed:
                conn.execute("DELETE FROM task_runs WHERE task_id=?", (task_id,))
            return cur.rowcount > 0

    def pause(self, task_id: int, user_id: int) -> bool:
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE id=? AND user_id=? AND status IN (?, ?)",
                (STATUS_PAUSED, task_id, user_id, STATUS_PENDING, STATUS_ARMED),
            )
            return cur.rowcount > 0

    def resume(self, task_id: int, user_id: int) -> bool:
        task = self.get(task_id)
        if task is None or task.user_id != user_id or task.status != STATUS_PAUSED:
            return False
        if task.kind == KIND_EVENT:
            with _db() as conn:
                cur = conn.execute(
                    "UPDATE tasks SET status=?, retries=0 WHERE id=? AND user_id=?",
                    (STATUS_ARMED, task_id, user_id),
                )
                return cur.rowcount > 0
        nxt = task.execute_at
        if nxt <= datetime.now(timezone.utc) and task.schedule_kind != SCHEDULE_ONCE:
            nxt = next_occurrence(task, after=datetime.now(timezone.utc)) or nxt
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=?, execute_at=?, retries=0 WHERE id=? AND user_id=?",
                (STATUS_PENDING, _as_utc(nxt).isoformat(), task_id, user_id),
            )
            return cur.rowcount > 0

    def skip_next(self, task_id: int, user_id: int) -> Optional[datetime]:
        task = self.get(task_id)
        if task is None or task.user_id != user_id:
            return None
        if task.status not in (STATUS_PENDING, STATUS_PAUSED):
            return None
        nxt = next_occurrence(task, after=task.execute_at)
        if nxt is None:
            return None
        with _db() as conn:
            conn.execute(
                "UPDATE tasks SET execute_at=?, retries=0 WHERE id=? AND user_id=?",
                (nxt.isoformat(), task_id, user_id),
            )
        return nxt

    def cancel(self, task_id: int, user_id: int) -> bool:
        with _db() as conn:
            cur = conn.execute(
                """
                UPDATE tasks SET status=?
                WHERE id=? AND user_id=? AND status IN (?, ?, ?, ?, ?, ?)
                """,
                (
                    STATUS_CANCELLED, task_id, user_id,
                    STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED, STATUS_RUNNING,
                    STATUS_ARMED, STATUS_DRAFT,
                ),
            )
            return cur.rowcount > 0

    def cancel_all(self, user_id: int) -> int:
        with _db() as conn:
            cur = conn.execute(
                """
                UPDATE tasks SET status=?
                WHERE user_id=? AND status IN (?, ?, ?, ?, ?)
                """,
                (
                    STATUS_CANCELLED, user_id,
                    STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED,
                    STATUS_ARMED, STATUS_DRAFT,
                ),
            )
            return cur.rowcount

    def still_running(self, task_id: int) -> bool:
        with _db() as conn:
            row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        return bool(row and row["status"] == STATUS_RUNNING)


class TaskWorker:
    def __init__(self, store: TaskStore, executor: Callable):
        self.store = store
        self.executor = executor
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("TaskWorker démarré")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self.store.purge_expired_drafts)
                await asyncio.to_thread(self.store.expire_overdue)
                claimed = await asyncio.to_thread(self.store.claim_due)
                if claimed is not None:
                    async with self._lock:
                        await self._run_one(claimed)
                    continue
                next_at = await asyncio.to_thread(self.store.get_next_due_at)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("TaskWorker : itération échouée, nouvel essai dans 30 s", exc_info=True)
                await asyncio.sleep(30)
                continue
            if next_at:
                delay = (next_at - datetime.now(timezone.utc)).total_seconds()
                delay = min(max(delay, 5), 300)
            else:
                delay = 60
            await asyncio.sleep(delay)

    async def _run_one(self, task: ScheduledTask) -> None:
        # Event : déclenché hors horloge (listener). Watch / at / recurring : claim_due.
        if task.kind == KIND_EVENT:
            return
        if not await asyncio.to_thread(self.store.still_running, task.id):
            return
        try:
            await self.executor(task)
        except Exception as e:
            attempts = await asyncio.to_thread(self.store.retry_later, task.id, str(e))
            if attempts >= MAX_SEND_RETRIES:
                logger.error("Tâche #%s abandonnée après %s tentatives: %s", task.id, attempts, e)
                await asyncio.to_thread(self.store.mark_failed, task.id, str(e))
            else:
                logger.warning(
                    "Tâche #%s échec (tentative %s/%s): %s",
                    task.id, attempts, MAX_SEND_RETRIES, e,
                )
            return
        if task.kind == KIND_WATCH:
            # L'executor appelle reschedule_watch ; filet si encore running.
            still = await asyncio.to_thread(self.store.still_running, task.id)
            if still:
                await asyncio.to_thread(self.store.reschedule_watch, task.id, fired=False)
            return
        if not await asyncio.to_thread(self.store.still_running, task.id):
            return
        await asyncio.to_thread(self.store.reschedule_after_run, task)
