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

STATUS_PENDING = "pending"
STATUS_PAUSED = "paused"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = (STATUS_PENDING, STATUS_PAUSED, STATUS_RUNNING)

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
TASK_MAX_PENDING = 10
TASK_MAX_RECURRING = 3
TASK_MIN_MINUTES = 2
TASK_MAX_DAYS = 365
TASK_INSTRUCTION_MAX = 500
TASK_TITLE_MAX = 80


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
    """Première occurrence assez loin (min 2 min), calée sur l'heure / les jours.

    Une daily/weekly demandée à une heure déjà passée avance au prochain créneau
    au lieu d'échouer (« trop proche »).
    """
    now_utc = _as_utc(now or datetime.now(timezone.utc))
    min_at = now_utc + timedelta(minutes=TASK_MIN_MINUTES)
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


def _row_to_task(r: sqlite3.Row) -> ScheduledTask:
    keys = r.keys()
    return ScheduledTask(
        id=r["id"],
        channel_id=r["channel_id"],
        user_id=r["user_id"],
        guild_id=r["guild_id"] or 0,
        instruction=r["instruction"] or "",
        execute_at=_as_utc(datetime.fromisoformat(r["execute_at"])),
        title=(r["title"] if "title" in keys else "") or "",
        schedule_kind=(r["schedule_kind"] if "schedule_kind" in keys else SCHEDULE_ONCE) or SCHEDULE_ONCE,
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
        if "deliver_dm" not in cols:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN deliver_dm INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_at ON tasks(status, execute_at)"
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
    ) -> int:
        if schedule_kind not in VALID_SCHEDULES:
            schedule_kind = SCHEDULE_ONCE
        execute_at = _as_utc(execute_at)
        days = normalize_weekdays(weekdays)
        if schedule_kind == SCHEDULE_WEEKLY and not days:
            days = [WEEKDAYS[execute_at.astimezone(PARIS_TZ).weekday()]]
        tod = ""
        if schedule_kind != SCHEDULE_ONCE:
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
        with _db() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (
                    channel_id, user_id, guild_id, title, instruction, execute_at,
                    schedule_kind, weekdays, time_of_day, until_at, status,
                    message_id, created_at, deliver_dm
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    channel_id, user_id, guild_id, title, instruction,
                    execute_at.isoformat(), schedule_kind, json.dumps(days),
                    tod, until_iso, STATUS_PENDING, message_id, now_iso,
                    1 if deliver_dm else 0,
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
                WHERE user_id=? AND status IN (?, ?, ?)
                ORDER BY execute_at
                """,
                (user_id, STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED),
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def count_active(self, user_id: int) -> int:
        with _db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND status IN (?, ?)
                """,
                (user_id, STATUS_PENDING, STATUS_PAUSED),
            ).fetchone()
        return row[0] if row else 0

    def count_active_recurring(self, user_id: int, *, exclude_id: int = 0) -> int:
        with _db() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE user_id=? AND status IN (?, ?)
                  AND schedule_kind IN (?, ?)
                  AND id != ?
                """,
                (
                    user_id, STATUS_PENDING, STATUS_PAUSED,
                    SCHEDULE_DAILY, SCHEDULE_WEEKLY, exclude_id,
                ),
            ).fetchone()
        return row[0] if row else 0

    def claim_due(self) -> Optional[ScheduledTask]:
        """Passe la plus ancienne tâche due en running. None si rien."""
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            row = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status=? AND execute_at <= ?
                ORDER BY execute_at LIMIT 1
                """,
                (STATUS_PENDING, now),
            ).fetchone()
            if not row:
                return None
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE id=? AND status=?",
                (STATUS_RUNNING, row["id"], STATUS_PENDING),
            )
            if cur.rowcount == 0:
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
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, last_run_at=?, last_error=? WHERE id=?",
                (STATUS_FAILED, now, (error or "")[:300], task_id),
            )

    def reschedule_after_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Après un fire réussi : prochaine occ. ou completed. Retourne la date ou None."""
        now = datetime.now(timezone.utc)
        nxt = next_occurrence(task, after=now)
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
        """Remet pending + incrément retries. Retourne le nouveau compteur."""
        with _db() as conn:
            conn.execute(
                """
                UPDATE tasks SET status=?, retries=retries+1, last_error=?
                WHERE id=?
                """,
                (STATUS_PENDING, (error or "")[:300], task_id),
            )
            row = conn.execute("SELECT retries FROM tasks WHERE id=?", (task_id,)).fetchone()
        return row[0] if row else MAX_SEND_RETRIES

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
    ) -> bool:
        current = self.get(task_id)
        if current is None or current.user_id != user_id:
            return False
        if current.status not in (STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED):
            return False
        sets: list[str] = []
        params: list[object] = []
        if instruction is not None:
            sets.append("instruction=?")
            params.append(instruction.strip()[:TASK_INSTRUCTION_MAX])
            if not title:
                sets.append("title=?")
                params.append(instruction.strip()[:TASK_TITLE_MAX])
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
            return cur.rowcount > 0

    def pause(self, task_id: int, user_id: int) -> bool:
        with _db() as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE id=? AND user_id=? AND status=?",
                (STATUS_PAUSED, task_id, user_id, STATUS_PENDING),
            )
            return cur.rowcount > 0

    def resume(self, task_id: int, user_id: int) -> bool:
        task = self.get(task_id)
        if task is None or task.user_id != user_id or task.status != STATUS_PAUSED:
            return False
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
                WHERE id=? AND user_id=? AND status IN (?, ?, ?, ?)
                """,
                (
                    STATUS_CANCELLED, task_id, user_id,
                    STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED, STATUS_RUNNING,
                ),
            )
            return cur.rowcount > 0

    def cancel_all(self, user_id: int) -> int:
        with _db() as conn:
            cur = conn.execute(
                """
                UPDATE tasks SET status=?
                WHERE user_id=? AND status IN (?, ?, ?)
                """,
                (STATUS_CANCELLED, user_id, STATUS_PENDING, STATUS_PAUSED, STATUS_FAILED),
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
            claimed = self.store.claim_due()
            if claimed is not None:
                async with self._lock:
                    await self._run_one(claimed)
                continue
            next_at = self.store.get_next_due_at()
            if next_at:
                delay = (next_at - datetime.now(timezone.utc)).total_seconds()
                delay = min(max(delay, 5), 300)
            else:
                delay = 60
            await asyncio.sleep(delay)

    async def _run_one(self, task: ScheduledTask) -> None:
        if not self.store.still_running(task.id):
            return
        try:
            await self.executor(task)
        except Exception as e:
            attempts = self.store.retry_later(task.id, str(e))
            if attempts >= MAX_SEND_RETRIES:
                logger.error("Tâche #%s abandonnée après %s tentatives: %s", task.id, attempts, e)
                self.store.mark_failed(task.id, str(e))
            else:
                logger.warning(
                    "Tâche #%s échec (tentative %s/%s): %s",
                    task.id, attempts, MAX_SEND_RETRIES, e,
                )
            return
        if not self.store.still_running(task.id):
            return
        self.store.reschedule_after_run(task)
