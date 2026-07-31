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
STATUS_PENDING = "pending"
STATUS_ARCHIVED = "archived"

CONFIDENCE_PENDING = 0.2
# Collectif (server/event) : actif dès la 1re capture, seuil RAG = 0.3.
CONFIDENCE_COLLECTIVE = 0.5
# Fait perso dit directement à MARIA (mention / reply) — confiance élevée, actif tout de suite.
CONFIDENCE_DIRECT = 0.75
# Faits immuables affirmés clairement (anniv…) — actifs tout de suite, hors decay.
CONFIDENCE_STABLE = 0.99
CONFIDENCE_UPDATE_DELTA = 0.15
CONFIDENCE_CONTRADICT_DELTA = 0.25
CONFIDENCE_ARCHIVE_BELOW = 0.2
CONFIDENCE_DECAY = 0.05
CONFIDENCE_DECAY_ARCHIVE_BELOW = 0.15
DECAY_AFTER_DAYS = 30
# 2e observation → promotion pending → active (évite les one-shots type « running gag »)
HITS_TO_PROMOTE = 2
# 21j : laisse le temps à un signal qui revient (ex. météo d'une ville) de se confirmer
# une 2e fois sans traîner indéfiniment en base.
PENDING_EXPIRE_DAYS = 21


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
    hits: int = 1


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
        # Migration douce : compteur d'observations (tampon pending → active).
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "hits" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN hits INTEGER NOT NULL DEFAULT 1")


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
    keys = r.keys()
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
        hits=int(r["hits"]) if "hits" in keys and r["hits"] is not None else 1,
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
        """Actifs + pending (tampon) pour l'agent : users globaux + serveur local."""
        statuses = (STATUS_ACTIVE, STATUS_PENDING)
        with _db() as conn:
            server_rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE guild_id = ? AND status IN (?, ?) AND category IN (?, ?)
                ORDER BY
                    CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                    confirmed_at DESC
                LIMIT ?
                """,
                (
                    guild_id, STATUS_ACTIVE, STATUS_PENDING,
                    CATEGORY_SERVER, CATEGORY_EVENT, max(8, limit // 2),
                ),
            ).fetchall()
            if not user_ids:
                return [_row_to_memory(r) for r in server_rows][:limit]
            placeholders = ",".join("?" * len(user_ids))
            user_rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE status IN (?, ?) AND category = ? AND user_id IN ({placeholders})
                ORDER BY
                    CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                    confirmed_at DESC
                LIMIT ?
                """,
                (STATUS_ACTIVE, STATUS_PENDING, CATEGORY_USER, *user_ids, limit),
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

    def list_for_user(
        self,
        guild_id: int,
        user_id: int,
        *,
        limit: int = 40,
        include_server: bool = False,
    ) -> list[Memory]:
        """Souvenirs perso d'un membre (globaux) (+ optionnellement serveur local)."""
        with _db() as conn:
            user_rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE status = ? AND user_id = ? AND category = ?
                ORDER BY confidence DESC, confirmed_at DESC
                LIMIT ?
                """,
                (STATUS_ACTIVE, user_id, CATEGORY_USER, limit),
            ).fetchall()
            server_rows = []
            if include_server:
                server_rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE guild_id = ? AND status = ? AND category = ?
                    ORDER BY confidence DESC, confirmed_at DESC
                    LIMIT ?
                    """,
                    (guild_id, STATUS_ACTIVE, CATEGORY_SERVER, min(10, limit // 2)),
                ).fetchall()
        seen: set[str] = set()
        out: list[Memory] = []
        for r in list(user_rows) + list(server_rows):
            m = _row_to_memory(r)
            if m.id in seen:
                continue
            seen.add(m.id)
            out.append(m)
        return out

    def list_server(
        self,
        guild_id: int,
        *,
        limit: int = 40,
    ) -> list[Memory]:
        """Souvenirs collectifs du serveur (server + event, sans mémoires user)."""
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE guild_id = ? AND status = ? AND category IN (?, ?)
                ORDER BY confidence DESC, confirmed_at DESC
                LIMIT ?
                """,
                (guild_id, STATUS_ACTIVE, CATEGORY_SERVER, CATEGORY_EVENT, limit),
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def list_recent(
        self,
        guild_id: int,
        *,
        category: Optional[str] = None,
        limit: int = 25,
        include_pending: bool = True,
    ) -> list[Memory]:
        """Derniers souvenirs créés sur ce guild (user + server + event), created_at DESC."""
        limit = max(1, min(limit, 40))
        statuses = (STATUS_ACTIVE, STATUS_PENDING) if include_pending else (STATUS_ACTIVE,)
        status_ph = ",".join("?" * len(statuses))
        with _db() as conn:
            if category and category in VALID_CATEGORIES:
                rows = conn.execute(
                    f"""
                    SELECT * FROM memories
                    WHERE guild_id = ? AND status IN ({status_ph}) AND category = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (guild_id, *statuses, category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT * FROM memories
                    WHERE guild_id = ? AND status IN ({status_ph})
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (guild_id, *statuses, limit),
                ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def search_active(
        self,
        guild_id: int,
        *,
        query: str = "",
        category: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Recherche textuelle sur les souvenirs actifs (user globaux + serveur local).

        - category=user : perso (tous serveurs), filtre user_id optionnel
        - category=server|event : ce guild uniquement
        - category=None : user (globaux) + server/event du guild
        """
        q = (query or "").strip().lower()
        limit = max(1, min(limit, 30))
        with _db() as conn:
            rows: list = []
            if category == CATEGORY_USER or category is None:
                sql = """
                    SELECT * FROM memories
                    WHERE status = ? AND category = ?
                """
                params: list = [STATUS_ACTIVE, CATEGORY_USER]
                if user_id is not None:
                    sql += " AND user_id = ?"
                    params.append(user_id)
                if q:
                    sql += " AND lower(content) LIKE ?"
                    params.append(f"%{q}%")
                sql += " ORDER BY confidence DESC, confirmed_at DESC LIMIT ?"
                params.append(limit if category == CATEGORY_USER else limit)
                rows.extend(conn.execute(sql, params).fetchall())

            if category in (CATEGORY_SERVER, CATEGORY_EVENT) or category is None:
                cats = (
                    (category,)
                    if category in (CATEGORY_SERVER, CATEGORY_EVENT)
                    else (CATEGORY_SERVER, CATEGORY_EVENT)
                )
                placeholders = ",".join("?" * len(cats))
                sql = f"""
                    SELECT * FROM memories
                    WHERE guild_id = ? AND status = ? AND category IN ({placeholders})
                """
                params = [guild_id, STATUS_ACTIVE, *cats]
                if q:
                    sql += " AND lower(content) LIKE ?"
                    params.append(f"%{q}%")
                sql += " ORDER BY confidence DESC, confirmed_at DESC LIMIT ?"
                params.append(limit)
                rows.extend(conn.execute(sql, params).fetchall())

        seen: set[str] = set()
        out: list[Memory] = []
        for r in rows:
            m = _row_to_memory(r)
            if m.id in seen:
                continue
            # Si query fournie et category None, on a pu doubler les LIMIT — refiltre.
            if q and q not in m.content.lower():
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
        confidence: float = CONFIDENCE_PENDING,
        status: str = STATUS_PENDING,
    ) -> Memory:
        now = datetime.now(timezone.utc)
        mid = str(uuid.uuid4())
        if status not in (STATUS_ACTIVE, STATUS_PENDING, STATUS_ARCHIVED):
            status = STATUS_PENDING
        mem = Memory(
            id=mid,
            category=category if category in VALID_CATEGORIES else CATEGORY_USER,
            guild_id=guild_id,
            user_id=user_id,
            content=content.strip(),
            created_at=now,
            confirmed_at=now,
            confidence=max(0.0, min(1.0, confidence)),
            status=status,
            chroma_id=mid if status == STATUS_ACTIVE else None,
            hits=1,
        )
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO memories
                (id, category, guild_id, user_id, content, created_at, confirmed_at,
                 confidence, status, chroma_id, hits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem.id, mem.category, mem.guild_id, mem.user_id, mem.content,
                    mem.created_at.isoformat(), mem.confirmed_at.isoformat(),
                    mem.confidence, mem.status, mem.chroma_id, mem.hits,
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
        """Renforce un souvenir (hits++) ; promeut pending → active si assez d'observations."""
        mem = self.get(memory_id)
        if mem is None or mem.status not in (STATUS_ACTIVE, STATUS_PENDING):
            return None
        now = datetime.now(timezone.utc)
        new_hits = mem.hits + 1
        new_conf = max(0.0, min(1.0, mem.confidence + confidence_delta))
        new_status = mem.status
        chroma_id = mem.chroma_id
        if mem.status == STATUS_PENDING and new_hits >= HITS_TO_PROMOTE:
            new_status = STATUS_ACTIVE
            chroma_id = mem.id
        with _db() as conn:
            conn.execute(
                """
                UPDATE memories
                SET content = ?, confirmed_at = ?, confidence = ?, hits = ?,
                    status = ?, chroma_id = ?
                WHERE id = ?
                """,
                (
                    content.strip(), now.isoformat(), new_conf, new_hits,
                    new_status, chroma_id, memory_id,
                ),
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

    def promote_stable(self, memory_id: str, content: Optional[str] = None) -> Optional[Memory]:
        """Force un souvenir en ACTIVE à confiance stable (faits immuables confirmés)."""
        return self._promote(memory_id, CONFIDENCE_STABLE, content=content)

    def promote_direct(self, memory_id: str, content: Optional[str] = None) -> Optional[Memory]:
        """Force ACTIVE à confiance « dit à MARIA » (sans descendre si déjà plus haut)."""
        mem = self.get(memory_id)
        if mem is None:
            return None
        conf = max(float(mem.confidence), CONFIDENCE_DIRECT)
        return self._promote(memory_id, conf, content=content)

    def _promote(
        self,
        memory_id: str,
        confidence: float,
        *,
        content: Optional[str] = None,
    ) -> Optional[Memory]:
        mem = self.get(memory_id)
        if mem is None or mem.status not in (STATUS_ACTIVE, STATUS_PENDING):
            return None
        now = datetime.now(timezone.utc)
        new_content = (content if content is not None else mem.content).strip()
        new_conf = max(0.0, min(1.0, confidence))
        with _db() as conn:
            conn.execute(
                """
                UPDATE memories
                SET content = ?, status = ?, confidence = ?, chroma_id = ?,
                    confirmed_at = ?, hits = CASE WHEN hits < 2 THEN 2 ELSE hits END
                WHERE id = ?
                """,
                (
                    new_content, STATUS_ACTIVE, new_conf, memory_id,
                    now.isoformat(), memory_id,
                ),
            )
        return self.get(memory_id)

    def contradict(self, memory_id: str) -> Optional[Memory]:
        mem = self.get(memory_id)
        if mem is None or mem.status not in (STATUS_ACTIVE, STATUS_PENDING):
            return None
        # Pending invalidé : on archive directement (jamais vu 2 fois = pas solide).
        if mem.status == STATUS_PENDING:
            self.archive(memory_id)
            return self.get(memory_id)
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

    def forget_user_memory(self, memory_id: str, user_id: int) -> tuple[bool, Optional[str]]:
        """Archive un souvenir perso. Renvoie (ok, chroma_id éventuel)."""
        mem = self.get(memory_id)
        if (
            mem is None
            or mem.category != CATEGORY_USER
            or mem.user_id != user_id
            or mem.status not in (STATUS_ACTIVE, STATUS_PENDING)
        ):
            return False, None
        chroma = mem.id if (mem.status == STATUS_ACTIVE or mem.chroma_id) else None
        self.archive(memory_id)
        return True, chroma

    def forget_server_memory(self, memory_id: str, guild_id: int) -> tuple[bool, Optional[str]]:
        """Archive un souvenir server/event. Renvoie (ok, chroma_id éventuel)."""
        mem = self.get(memory_id)
        if (
            mem is None
            or mem.guild_id != guild_id
            or mem.category not in (CATEGORY_SERVER, CATEGORY_EVENT)
            or mem.status not in (STATUS_ACTIVE, STATUS_PENDING)
        ):
            return False, None
        chroma = mem.id if (mem.status == STATUS_ACTIVE or mem.chroma_id) else None
        self.archive(memory_id)
        return True, chroma

    def clear_user(self, user_id: int) -> list[str]:
        """Archive toutes les mémoires perso d'un membre. Renvoie les ids à retirer de Chroma."""
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT id, status, chroma_id FROM memories
                WHERE category = ? AND user_id = ? AND status IN (?, ?)
                """,
                (CATEGORY_USER, user_id, STATUS_ACTIVE, STATUS_PENDING),
            ).fetchall()
            chroma_ids: list[str] = []
            for r in rows:
                if r["status"] == STATUS_ACTIVE or r["chroma_id"]:
                    chroma_ids.append(r["id"])
                conn.execute(
                    "UPDATE memories SET status = ? WHERE id = ?",
                    (STATUS_ARCHIVED, r["id"]),
                )
        return chroma_ids

    def clear_server(self, guild_id: int) -> list[str]:
        """Archive server + event d'un guild. Renvoie les ids à retirer de Chroma."""
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT id, status, chroma_id FROM memories
                WHERE guild_id = ? AND category IN (?, ?) AND status IN (?, ?)
                """,
                (
                    guild_id, CATEGORY_SERVER, CATEGORY_EVENT,
                    STATUS_ACTIVE, STATUS_PENDING,
                ),
            ).fetchall()
            chroma_ids: list[str] = []
            for r in rows:
                if r["status"] == STATUS_ACTIVE or r["chroma_id"]:
                    chroma_ids.append(r["id"])
                conn.execute(
                    "UPDATE memories SET status = ? WHERE id = ?",
                    (STATUS_ARCHIVED, r["id"]),
                )
        return chroma_ids

    def count_below_confidence(self, threshold: float) -> dict[str, int]:
        """Compte les souvenirs actifs/pending avec confidence < threshold (global)."""
        threshold = max(0.0, min(1.0, float(threshold)))
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT status, category, COUNT(*) AS n FROM memories
                WHERE status IN (?, ?) AND confidence < ?
                GROUP BY status, category
                """,
                (STATUS_ACTIVE, STATUS_PENDING, threshold),
            ).fetchall()
        out = {"total": 0, "pending": 0, "active": 0, "user": 0, "server": 0, "event": 0}
        for r in rows:
            n = int(r["n"])
            out["total"] += n
            st = r["status"]
            cat = r["category"]
            if st in out:
                out[st] += n
            if cat in out:
                out[cat] += n
        return out

    def clear_below_confidence(self, threshold: float) -> list[str]:
        """Archive tous les souvenirs (user+server+event, tous guilds) sous le seuil.

        Renvoie les ids à retirer de Chroma.
        """
        threshold = max(0.0, min(1.0, float(threshold)))
        with _db() as conn:
            rows = conn.execute(
                """
                SELECT id, status, chroma_id FROM memories
                WHERE status IN (?, ?) AND confidence < ?
                """,
                (STATUS_ACTIVE, STATUS_PENDING, threshold),
            ).fetchall()
            chroma_ids: list[str] = []
            for r in rows:
                if r["status"] == STATUS_ACTIVE or r["chroma_id"]:
                    chroma_ids.append(r["id"])
                conn.execute(
                    "UPDATE memories SET status = ? WHERE id = ?",
                    (STATUS_ARCHIVED, r["id"]),
                )
        return chroma_ids

    def apply_decay(self) -> list[str]:
        """Expire les pending trop vieux + decay des actifs non confirmés.

        Renvoie les ids à retirer de Chroma (actifs archivés).
        """
        now = datetime.now(timezone.utc)
        active_cutoff = now - timedelta(days=DECAY_AFTER_DAYS)
        pending_cutoff = now - timedelta(days=PENDING_EXPIRE_DAYS)
        archived: list[str] = []
        with _db() as conn:
            # Tampon : jamais reconfirmé → oubli silencieux.
            pending = conn.execute(
                """
                SELECT id FROM memories
                WHERE status = ? AND confirmed_at < ?
                """,
                (STATUS_PENDING, pending_cutoff.isoformat()),
            ).fetchall()
            for r in pending:
                conn.execute(
                    "UPDATE memories SET status = ? WHERE id = ?",
                    (STATUS_ARCHIVED, r["id"]),
                )

            rows = conn.execute(
                """
                SELECT id, confidence FROM memories
                WHERE status = ? AND confirmed_at < ?
                """,
                (STATUS_ACTIVE, active_cutoff.isoformat()),
            ).fetchall()
            for r in rows:
                if float(r["confidence"]) >= 0.99:
                    continue
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
        if archived or pending:
            logger.info(
                "Decay: %d actif(s) archivé(s), %d pending expiré(s)",
                len(archived), len(pending),
            )
        return archived
