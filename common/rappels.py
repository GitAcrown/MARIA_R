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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rappels_status_at ON rappels(status, execute_at)"
        )


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


def _row_to_rappel(r: sqlite3.Row) -> Rappel:
    keys = r.keys()
    return Rappel(
        id=r["id"],
        channel_id=r["channel_id"],
        user_id=r["user_id"],
        description=r["description"],
        execute_at=datetime.fromisoformat(r["execute_at"]),
        message_id=r["message_id"] or 0,
        recurrence=(r["recurrence"] if "recurrence" in keys else RECURRENCE_NONE) or RECURRENCE_NONE,
        kind=(r["kind"] if "kind" in keys else KIND_PERSONAL) or KIND_PERSONAL,
        retries=(r["retries"] if "retries" in keys else 0) or 0,
    )


def next_occurrence(execute_at: datetime, recurrence: str, *, after: Optional[datetime] = None) -> Optional[datetime]:
    """Calcule la prochaine occurrence strictement future d'un rappel récurrent.

    Avance par pas (jour/semaine) jusqu'à dépasser `after` (par défaut maintenant),
    pour rattraper proprement un éventuel retard sans déclencher en rafale.
    """
    if recurrence == RECURRENCE_DAILY:
        step = timedelta(days=1)
    elif recurrence == RECURRENCE_WEEKLY:
        step = timedelta(weeks=1)
    else:
        return None
    after = after or datetime.now(timezone.utc)
    nxt = execute_at
    # Garde-fou contre une boucle infinie (max ~5 ans de rattrapage)
    for _ in range(2000):
        nxt = nxt + step
        if nxt > after:
            return nxt
    return nxt


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
    ) -> int:
        if recurrence not in VALID_RECURRENCES:
            recurrence = RECURRENCE_NONE
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO rappels (channel_id, user_id, description, execute_at, message_id, recurrence, kind)"
                " VALUES (?,?,?,?,?,?,?)",
                (channel_id, user_id, description, execute_at.isoformat(), message_id, recurrence, kind),
            )
            return cur.lastrowid

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
        sets: list[str] = []
        params: list[object] = []
        if description is not None:
            sets.append("description=?")
            params.append(description)
        if execute_at is not None:
            sets.append("execute_at=?")
            params.append(execute_at.isoformat())
        if recurrence is not None and recurrence in VALID_RECURRENCES:
            sets.append("recurrence=?")
            params.append(recurrence)
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
                "UPDATE rappels SET execute_at=?, retries=0"
                " WHERE id=? AND user_id=? AND status IN ('pending','failed')",
                (new_at.isoformat(), rappel_id, user_id),
            )
            if cur.rowcount > 0:
                conn.execute("UPDATE rappels SET status='pending' WHERE id=?", (rappel_id,))
                return new_at
        return None

    def get_due(self) -> list[Rappel]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM rappels WHERE status='pending' AND execute_at <= ? ORDER BY execute_at",
                (datetime.now(timezone.utc).isoformat(),),
            ).fetchall()
        return [_row_to_rappel(r) for r in rows]

    def get_next_due_at(self) -> Optional[datetime]:
        with _db() as conn:
            row = conn.execute(
                "SELECT execute_at FROM rappels WHERE status='pending' ORDER BY execute_at LIMIT 1"
            ).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def mark_done(self, rappel_id: int) -> None:
        with _db() as conn:
            conn.execute("UPDATE rappels SET status='completed' WHERE id=?", (rappel_id,))

    def mark_failed(self, rappel_id: int) -> None:
        with _db() as conn:
            conn.execute("UPDATE rappels SET status='failed' WHERE id=?", (rappel_id,))

    def reschedule(self, rappel_id: int, execute_at: datetime) -> None:
        """Replanifie un rappel récurrent (garde le statut pending, remet les retries à 0)."""
        with _db() as conn:
            conn.execute(
                "UPDATE rappels SET execute_at=?, retries=0 WHERE id=?",
                (execute_at.isoformat(), rappel_id),
            )

    def bump_retry(self, rappel_id: int) -> int:
        """Incrémente le compteur de tentatives et retourne sa nouvelle valeur."""
        with _db() as conn:
            conn.execute("UPDATE rappels SET retries=retries+1 WHERE id=?", (rappel_id,))
            row = conn.execute("SELECT retries FROM rappels WHERE id=?", (rappel_id,)).fetchone()
        return row[0] if row else MAX_SEND_RETRIES

    def cancel(self, rappel_id: int, user_id: int) -> bool:
        with _db() as conn:
            cur = conn.execute(
                "UPDATE rappels SET status='cancelled'"
                " WHERE id=? AND user_id=? AND status='pending'",
                (rappel_id, user_id),
            )
            return cur.rowcount > 0

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
                try:
                    await self.executor(r)
                except Exception as e:
                    # Échec d'envoi : retenter jusqu'à MAX_SEND_RETRIES avant d'abandonner.
                    attempts = self.store.bump_retry(r.id)
                    if attempts >= MAX_SEND_RETRIES:
                        logger.error(f"Rappel #{r.id} abandonné après {attempts} tentatives: {e}")
                        self.store.mark_failed(r.id)
                    else:
                        logger.warning(f"Rappel #{r.id} échec (tentative {attempts}/{MAX_SEND_RETRIES}): {e}")
                    continue

                # Succès : récurrent -> replanifier la prochaine occurrence ; sinon terminer.
                if r.recurrence != RECURRENCE_NONE:
                    nxt = next_occurrence(r.execute_at, r.recurrence)
                    if nxt:
                        self.store.reschedule(r.id, nxt)
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
