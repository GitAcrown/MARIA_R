"""Stockage relationnel des souvenirs (SQLite)."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("MARIA.Memory.Store")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "memories.db"

CATEGORY_USER = "user"
CATEGORY_SERVER = "server"
CATEGORY_EVENT = "event"
VALID_CATEGORIES = (CATEGORY_USER, CATEGORY_SERVER, CATEGORY_EVENT)

STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

CONFIDENCE_CREATE = 0.35
CONFIDENCE_UPDATE_DELTA = 0.15
CONFIDENCE_CONTRADICT_DELTA = 0.25
CONFIDENCE_ARCHIVE_BELOW = 0.2
CONFIDENCE_DECAY = 0.05
CONFIDENCE_DECAY_ARCHIVE_BELOW = 0.15
DECAY_AFTER_DAYS = 30


@dataclass
class Memory:
    id: str
    category: str
    guild_id: int
    content: str
    created_at: datetime
    confirmed_at: datetime
    confidence: float
    status: str = STATUS_ACTIVE
    user_id: Optional[int] = None
    chroma_id: Optional[str] = None


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                category      TEXT NOT NULL,
                guild_id      INTEGER NOT NULL,
                user_id       INTEGER,
                content       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                confirmed_at  TEXT NOT NULL,
                confidence    REAL NOT NULL,
                status        TEXT NOT NULL DEFAULT 'active',
                chroma_id     TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_guild_status "
            "ON memories(guild_id, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user "
            "ON memories(guild_id, user_id, status)"
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


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_memory(r: sqlite3.Row) -> Memory:
    return Memory(
        id=r["id"],
        category=r["category"],
        guild_id=r["guild_id"],
        user_id=r["user_id"],
        content=r["content"],
        created_at=_parse_dt(r["created_at"]),
        confirmed_at=_parse_dt(r["confirmed_at"]),
        confidence=float(r["confidence"]),
        status=r["status"] or STATUS_ACTIVE,
        chroma_id=r["chroma_id"],
    )


class MemoryStore:
    def __init__(self) -> None:
        _init_db()

    def get(self, memory_id: str) -> Optional[Memory]:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_memory(row) if row else None

    def get_many(self, ids: list[str]) -> list[Memory]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with _db() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def list_for_users(
        self,
        guild_id: int,
        user_ids: set[int],
        *,
        limit: int = 15,
    ) -> list[Memory]:
        """Souvenirs actifs liés aux users du batch + quelques souvenirs serveur."""
        with _db() as conn:
            server_rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE guild_id = ? AND status = ? AND category = ?
                ORDER BY confirmed_at DESC
                LIMIT ?
                """,
                (guild_id, STATUS_ACTIVE, CATEGORY_SERVER, max(3, limit // 3)),
            ).fetchall()
            if not user_ids:
                return [_row_to_memory(r) for r in server_rows][:limit]
            placeholders = ",".join("?" * len(user_ids))
            user_rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE guild_id = ? AND status = ? AND user_id IN ({placeholders})
                ORDER BY confirmed_at DESC
                LIMIT ?
                """,
                (guild_id, STATUS_ACTIVE, *user_ids, limit),
            ).fetchall()
        seen: set[str] = set()
        out: list[Memory] = []
        for r in list(user_rows) + list(server_rows):
            m = _row_to_memory(r)
            if m.id in seen:
                continue
            seen.add(m.id)
            out.append(m)
            if len(out) >= limit:
                break
        return out

    def create(
        self,
        *,
        category: str,
        guild_id: int,
        content: str,
        user_id: Optional[int] = None,
        confidence: float = CONFIDENCE_CREATE,
    ) -> Memory:
        now = datetime.now(timezone.utc)
        mid = str(uuid.uuid4())
        mem = Memory(
            id=mid,
            category=category if category in VALID_CATEGORIES else CATEGORY_USER,
            guild_id=guild_id,
            user_id=user_id,
            content=content.strip(),
            created_at=now,
            confirmed_at=now,
            confidence=max(0.0, min(1.0, confidence)),
            status=STATUS_ACTIVE,
            chroma_id=mid,
        )
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO memories
                (id, category, guild_id, user_id, content, created_at, confirmed_at,
                 confidence, status, chroma_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem.id, mem.category, mem.guild_id, mem.user_id, mem.content,
                    mem.created_at.isoformat(), mem.confirmed_at.isoformat(),
                    mem.confidence, mem.status, mem.chroma_id,
                ),
            )
        return mem

    def update_content(
        self,
        memory_id: str,
        content: str,
        *,
        confidence_delta: float = CONFIDENCE_UPDATE_DELTA,
    ) -> Optional[Memory]:
        mem = self.get(memory_id)
        if mem is None or mem.status != STATUS_ACTIVE:
            return None
        now = datetime.now(timezone.utc)
        new_conf = max(0.0, min(1.0, mem.confidence + confidence_delta))
        with _db() as conn:
            conn.execute(
                """
                UPDATE memories
                SET content = ?, confirmed_at = ?, confidence = ?
                WHERE id = ?
                """,
                (content.strip(), now.isoformat(), new_conf, memory_id),
            )
        return self.get(memory_id)

    def bump_confidence(
        self,
        memory_id: str,
        delta: float = CONFIDENCE_UPDATE_DELTA,
    ) -> Optional[Memory]:
        mem = self.get(memory_id)
        if mem is None or mem.status != STATUS_ACTIVE:
            return None
        now = datetime.now(timezone.utc)
        new_conf = max(0.0, min(1.0, mem.confidence + delta))
        with _db() as conn:
            conn.execute(
                """
                UPDATE memories
                SET confirmed_at = ?, confidence = ?
                WHERE id = ?
                """,
                (now.isoformat(), new_conf, memory_id),
            )
        return self.get(memory_id)

    def contradict(self, memory_id: str) -> Optional[Memory]:
        mem = self.get(memory_id)
        if mem is None or mem.status != STATUS_ACTIVE:
            return None
        new_conf = max(0.0, mem.confidence - CONFIDENCE_CONTRADICT_DELTA)
        if new_conf < CONFIDENCE_ARCHIVE_BELOW:
            self.archive(memory_id)
            return self.get(memory_id)
        with _db() as conn:
            conn.execute(
                "UPDATE memories SET confidence = ? WHERE id = ?",
                (new_conf, memory_id),
            )
        return self.get(memory_id)

    def archive(self, memory_id: str) -> None:
        with _db() as conn:
            conn.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (STATUS_ARCHIVED, memory_id),
            )

    def apply_decay(self) -> list[str]:
        """Baisse la confiance des souvenirs non confirmés depuis DECAY_AFTER_DAYS.

        Renvoie les ids archivés (à retirer de Chroma).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=DECAY_AFTER_DAYS)
        archived: list[str] = []
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT id, confidence FROM memories
                WHERE status = ? AND confirmed_at < ?
                """,
                (STATUS_ACTIVE, cutoff.isoformat()),
            ).fetchall()
            for r in rows:
                new_conf = max(0.0, float(r["confidence"]) - CONFIDENCE_DECAY)
                if new_conf < CONFIDENCE_DECAY_ARCHIVE_BELOW:
                    conn.execute(
                        "UPDATE memories SET confidence = ?, status = ? WHERE id = ?",
                        (new_conf, STATUS_ARCHIVED, r["id"]),
                    )
                    archived.append(r["id"])
                else:
                    conn.execute(
                        "UPDATE memories SET confidence = ? WHERE id = ?",
                        (new_conf, r["id"]),
                    )
        if archived:
            logger.info("Decay: %d souvenir(s) archivé(s)", len(archived))
        return archived
