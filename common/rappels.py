"""Système de rappels simple."""

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger("rappels")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "rappels.db"


@dataclass
class Rappel:
    id: int
    channel_id: int
    user_id: int
    description: str
    execute_at: datetime
    message_id: int = 0


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
    return Rappel(
        id=r["id"],
        channel_id=r["channel_id"],
        user_id=r["user_id"],
        description=r["description"],
        execute_at=datetime.fromisoformat(r["execute_at"]),
        message_id=r["message_id"] or 0,
    )


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
    ) -> int:
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO rappels (channel_id, user_id, description, execute_at, message_id)"
                " VALUES (?,?,?,?,?)",
                (channel_id, user_id, description, execute_at.isoformat(), message_id),
            )
            return cur.lastrowid

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
                    self.store.mark_done(r.id)
                except Exception as e:
                    logger.error(f"Rappel #{r.id}: {e}")
                    self.store.mark_failed(r.id)

            next_at = self.store.get_next_due_at()
            if next_at:
                delay = (next_at - datetime.now(timezone.utc)).total_seconds()
                delay = min(max(delay, 10), 300)
            else:
                delay = 60
            await asyncio.sleep(delay)
