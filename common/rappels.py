"""Système de rappels simple."""

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger("MARIA.Rappels")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "rappels.db"

# Récurrences supportées
RECURRENCE_NONE = "none"
RECURRENCE_DAILY = "daily"
RECURRENCE_WEEKLY = "weekly"
VALID_RECURRENCES = (RECURRENCE_NONE, RECURRENCE_DAILY, RECURRENCE_WEEKLY)

# Types de rappels
KIND_PERSONAL = "personal"
KIND_EVENT = "event"

# Nombre maximum de tentatives d'envoi avant abandon
MAX_SEND_RETRIES = 3

# Une série récurrente s'arrête 30 jours après la 1re occurrence.
RECURRENCE_MAX_DAYS = 30

REPEAT_EMOJI = "<:repeat_small:1529199171029041283>"


@dataclass
class Rappel:
    id: int
    channel_id: int
    user_id: int
    description: str
    execute_at: datetime
    message_id: int = 0
    recurrence: str = RECURRENCE_NONE
    kind: str = KIND_PERSONAL
    retries: int = 0
    recurrence_until: Optional[datetime] = None


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rappels (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                user_id    INTEGER NOT NULL,
                description TEXT NOT NULL,
                execute_at TEXT NOT NULL,
                message_id INTEGER DEFAULT 0,
                status     TEXT DEFAULT 'pending'
            )
            """
        )
        # Migration douce : ajoute les colonnes manquantes sur une base existante.
        existing = _column_names(conn, "rappels")
        if "recurrence" not in existing:
            conn.execute(f"ALTER TABLE rappels ADD COLUMN recurrence TEXT DEFAULT '{RECURRENCE_NONE}'")
        if "kind" not in existing:
            conn.execute(f"ALTER TABLE rappels ADD COLUMN kind TEXT DEFAULT '{KIND_PERSONAL}'")
        if "retries" not in existing:
            conn.execute("ALTER TABLE rappels ADD COLUMN retries INTEGER DEFAULT 0")
        if "recurrence_until" not in existing:
            conn.execute("ALTER TABLE rappels ADD COLUMN recurrence_until TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rappels_status_at ON rappels(status, execute_at)"
        )
        n = _backfill_recurrence_limits(conn)
        if n:
            logger.info("Migration rappels : %d série(s) récurrente(s) bornée(s) à %dj", n, RECURRENCE_MAX_DAYS)


def _backfill_recurrence_limits(conn: sqlite3.Connection) -> int:
    """Pose une fin de série (now+30j) sur les récurrents pending sans borne.

    Les anciennes séries tournaient à l'infini ; on les coupe désormais.
    """
    now = datetime.now(timezone.utc)
    until_iso = (now + timedelta(days=RECURRENCE_MAX_DAYS)).isoformat()
    rows = conn.execute(
        """
        SELECT id, execute_at FROM rappels
        WHERE status = 'pending'
          AND recurrence IS NOT NULL AND recurrence != ?
          AND (recurrence_until IS NULL OR recurrence_until = '')
        """,
        (RECURRENCE_NONE,),
    ).fetchall()
    for r in rows:
        conn.execute(
            "UPDATE rappels SET recurrence_until = ? WHERE id = ?",
            (until_iso, r["id"]),
        )
    # Si la prochaine occurrence est déjà après la borne → terminer la série.
    overdue = conn.execute(
        """
        SELECT id, execute_at, recurrence_until FROM rappels
        WHERE status = 'pending'
          AND recurrence IS NOT NULL AND recurrence != ?
          AND recurrence_until IS NOT NULL AND recurrence_until != ''
        """,
        (RECURRENCE_NONE,),
    ).fetchall()
    stopped = 0
    for r in overdue:
        try:
            ex = _as_utc(datetime.fromisoformat(r["execute_at"]))
            until = _as_utc(datetime.fromisoformat(r["recurrence_until"]))
        except ValueError:
            continue
        if ex > until:
            conn.execute(
                "UPDATE rappels SET status = 'completed' WHERE id = ?",
                (r["id"],),
            )
            stopped += 1
    if stopped:
        logger.info("Migration rappels : %d série(s) déjà hors borne → completed", stopped)
    return len(rows)


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


def _parse_optional_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    return _as_utc(datetime.fromisoformat(raw))


def _row_to_rappel(r: sqlite3.Row) -> Rappel:
    keys = r.keys()
    return Rappel(
        id=r["id"],
        channel_id=r["channel_id"],
        user_id=r["user_id"],
        description=r["description"],
        execute_at=_as_utc(datetime.fromisoformat(r["execute_at"])),
        message_id=r["message_id"] or 0,
        recurrence=(r["recurrence"] if "recurrence" in keys else RECURRENCE_NONE) or RECURRENCE_NONE,
        kind=(r["kind"] if "kind" in keys else KIND_PERSONAL) or KIND_PERSONAL,
        retries=(r["retries"] if "retries" in keys else 0) or 0,
        recurrence_until=_parse_optional_dt(
            r["recurrence_until"] if "recurrence_until" in keys else None
        ),
    )


def compute_recurrence_until(execute_at: datetime) -> datetime:
    """Fin de série : 30 jours après la première occurrence planifiée."""
    return _as_utc(execute_at) + timedelta(days=RECURRENCE_MAX_DAYS)


def next_occurrence(
    execute_at: datetime,
    recurrence: str,
    *,
    after: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> Optional[datetime]:
    """Calcule la prochaine occurrence strictement future d'un rappel récurrent.

    Avance par pas (jour/semaine) jusqu'à dépasser `after` (par défaut maintenant),
    pour rattraper proprement un éventuel retard sans déclencher en rafale.
    Si `until` est fourni, renvoie None dès que la prochaine occurrence le dépasse.
    """
    if recurrence == RECURRENCE_DAILY:
        step = timedelta(days=1)
    elif recurrence == RECURRENCE_WEEKLY:
        step = timedelta(weeks=1)
    else:
        return None
    after = _as_utc(after or datetime.now(timezone.utc))
    until_utc = _as_utc(until) if until is not None else None
    nxt = _as_utc(execute_at)
    # Garde-fou contre une boucle infinie (max ~5 ans de rattrapage)
    for _ in range(2000):
        nxt = nxt + step
        if until_utc is not None and nxt > until_utc:
            return None
        if nxt > after:
            return nxt
    return None


class RappelStore:
    def __init__(self):
        _init_db()

    def add(
        self,
        channel_id: int,
        user_id: int,
        description: str,
        execute_at: datetime,
        message_id: int = 0,
        recurrence: str = RECURRENCE_NONE,
        kind: str = KIND_PERSONAL,
        recurrence_until: Optional[datetime] = None,
    ) -> int:
        if recurrence not in VALID_RECURRENCES:
            recurrence = RECURRENCE_NONE
        execute_at = _as_utc(execute_at)
        until_iso: Optional[str] = None
        if recurrence != RECURRENCE_NONE:
            until = _as_utc(recurrence_until) if recurrence_until else compute_recurrence_until(execute_at)
            until_iso = until.isoformat()
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO rappels (channel_id, user_id, description, execute_at, message_id, "
                "recurrence, kind, recurrence_until)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    channel_id, user_id, description, execute_at.isoformat(),
                    message_id, recurrence, kind, until_iso,
                ),
            )
            return cur.lastrowid

    def get(self, rappel_id: int) -> Optional[Rappel]:
        with _db() as conn:
            row = conn.execute("SELECT * FROM rappels WHERE id=?", (rappel_id,)).fetchone()
        return _row_to_rappel(row) if row else None

    def edit(
        self,
        rappel_id: int,
        user_id: int,
        *,
        description: Optional[str] = None,
        execute_at: Optional[datetime] = None,
        recurrence: Optional[str] = None,
    ) -> bool:
        """Modifie un rappel en attente appartenant à l'utilisateur."""
        current = self.get(rappel_id)
        if current is None or current.user_id != user_id:
            return False

        sets: list[str] = []
        params: list[object] = []
        if description is not None:
            sets.append("description=?")
            params.append(description)

        new_execute = _as_utc(execute_at) if execute_at is not None else None
        if new_execute is not None:
            sets.append("execute_at=?")
            params.append(new_execute.isoformat())

        new_recurrence = recurrence
        if new_recurrence is not None and new_recurrence not in VALID_RECURRENCES:
            new_recurrence = None
        if new_recurrence is not None:
            sets.append("recurrence=?")
            params.append(new_recurrence)
            # Recalcule la fin de série si on (ré)active une récurrence.
            if new_recurrence == RECURRENCE_NONE:
                sets.append("recurrence_until=?")
                params.append(None)
            else:
                base = new_execute or current.execute_at
                sets.append("recurrence_until=?")
                params.append(compute_recurrence_until(base).isoformat())
        elif new_execute is not None and current.recurrence != RECURRENCE_NONE:
            # Nouvelle 1re occurrence → décale la fenêtre de 30j.
            sets.append("recurrence_until=?")
            params.append(compute_recurrence_until(new_execute).isoformat())

        if not sets:
            return False
        params.extend([rappel_id, user_id])
        with _db() as conn:
            cur = conn.execute(
                f"UPDATE rappels SET {', '.join(sets)}"
                f" WHERE id=? AND user_id=? AND status='pending'",
                params,
            )
            return cur.rowcount > 0

    def snooze(self, rappel_id: int, user_id: int, minutes: int) -> Optional[datetime]:
        """Reporte un rappel de `minutes` à partir de maintenant. Retourne la nouvelle date."""
        new_at = datetime.now(timezone.utc) + timedelta(minutes=max(1, minutes))
        with _db() as conn:
            cur = conn.execute(
                "UPDATE rappels SET execute_at=?, retries=0, status='pending'"
                " WHERE id=? AND user_id=? AND status IN ('pending','failed')",
                (new_at.isoformat(), rappel_id, user_id),
            )
            if cur.rowcount > 0:
                return new_at
        return None

    def get_due(self) -> list[Rappel]:
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM rappels WHERE status='pending' AND execute_at <= ? ORDER BY execute_at",
                (now,),
            ).fetchall()
        return [_row_to_rappel(r) for r in rows]

    def get_next_due_at(self) -> Optional[datetime]:
        with _db() as conn:
            row = conn.execute(
                "SELECT execute_at FROM rappels WHERE status='pending' ORDER BY execute_at LIMIT 1"
            ).fetchone()
        return _as_utc(datetime.fromisoformat(row[0])) if row else None

    def mark_done(self, rappel_id: int) -> None:
        with _db() as conn:
            conn.execute(
                "UPDATE rappels SET status='completed' WHERE id=? AND status='pending'",
                (rappel_id,),
            )

    def mark_failed(self, rappel_id: int) -> None:
        with _db() as conn:
            conn.execute(
                "UPDATE rappels SET status='failed' WHERE id=? AND status='pending'",
                (rappel_id,),
            )

    def reschedule(self, rappel_id: int, execute_at: datetime) -> bool:
        """Replanifie un rappel récurrent encore pending. Retourne False si annulé entre-temps."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE rappels SET execute_at=?, retries=0"
                " WHERE id=? AND status='pending'",
                (_as_utc(execute_at).isoformat(), rappel_id),
            )
            return cur.rowcount > 0

    def bump_retry(self, rappel_id: int) -> int:
        """Incrémente le compteur de tentatives et retourne sa nouvelle valeur."""
        with _db() as conn:
            conn.execute(
                "UPDATE rappels SET retries=retries+1 WHERE id=? AND status='pending'",
                (rappel_id,),
            )
            row = conn.execute("SELECT retries FROM rappels WHERE id=?", (rappel_id,)).fetchone()
        return row[0] if row else MAX_SEND_RETRIES

    def cancel(self, rappel_id: int, user_id: int) -> bool:
        """Annule un rappel pending ou failed (séries récurrentes incluses)."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE rappels SET status='cancelled'"
                " WHERE id=? AND user_id=? AND status IN ('pending','failed')",
                (rappel_id, user_id),
            )
            return cur.rowcount > 0

    def cancel_all(self, user_id: int) -> int:
        """Annule tous les rappels pending/failed de l'utilisateur. Renvoie le nombre annulé."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE rappels SET status='cancelled'"
                " WHERE user_id=? AND status IN ('pending','failed')",
                (user_id,),
            )
            return cur.rowcount

    def set_recurrence_until(self, rappel_id: int, until: datetime) -> None:
        with _db() as conn:
            conn.execute(
                "UPDATE rappels SET recurrence_until=? WHERE id=? AND status='pending'",
                (_as_utc(until).isoformat(), rappel_id),
            )

    def count_pending(self, user_id: int) -> int:
        with _db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rappels WHERE user_id=? AND status='pending'",
                (user_id,),
            ).fetchone()
        return row[0] if row else 0

    def get_user_rappels(self, user_id: int) -> list[Rappel]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM rappels WHERE user_id=? AND status='pending' ORDER BY execute_at",
                (user_id,),
            ).fetchall()
        return [_row_to_rappel(r) for r in rows]


class RappelWorker:
    def __init__(self, store: RappelStore, executor: Callable):
        self.store = store
        self.executor = executor
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("RappelWorker démarré")

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
            for r in self.store.get_due():
                if not self._still_pending(r.id):
                    continue
                try:
                    await self.executor(r)
                except Exception as e:
                    attempts = self.store.bump_retry(r.id)
                    if attempts >= MAX_SEND_RETRIES:
                        logger.error(f"Rappel #{r.id} abandonné après {attempts} tentatives: {e}")
                        self.store.mark_failed(r.id)
                    else:
                        logger.warning(
                            f"Rappel #{r.id} échec (tentative {attempts}/{MAX_SEND_RETRIES}): {e}"
                        )
                    continue

                # Annulé pendant l'envoi → ne pas ressusciter la série.
                if not self._still_pending(r.id):
                    continue

                if r.recurrence != RECURRENCE_NONE:
                    until = r.recurrence_until
                    if until is None:
                        until = datetime.now(timezone.utc) + timedelta(days=RECURRENCE_MAX_DAYS)
                        self.store.set_recurrence_until(r.id, until)
                    # Prochaine occurrence déjà hors fenêtre → fin de série.
                    if _as_utc(r.execute_at) > _as_utc(until):
                        self.store.mark_done(r.id)
                        continue
                    nxt = next_occurrence(r.execute_at, r.recurrence, until=until)
                    if nxt:
                        if not self.store.reschedule(r.id, nxt):
                            logger.debug("Rappel #%s non replanifié (annulé entre-temps)", r.id)
                    else:
                        self.store.mark_done(r.id)
                else:
                    self.store.mark_done(r.id)

            next_at = self.store.get_next_due_at()
            if next_at:
                delay = (next_at - datetime.now(timezone.utc)).total_seconds()
                delay = min(max(delay, 10), 300)
            else:
                delay = 60
            await asyncio.sleep(delay)

    def _still_pending(self, rappel_id: int) -> bool:
        with _db() as conn:
            row = conn.execute(
                "SELECT status FROM rappels WHERE id=?", (rappel_id,),
            ).fetchone()
        return bool(row and row["status"] == "pending")
