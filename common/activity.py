"""Compteurs d'activité légers — salon le plus actif, membres les plus bavards,
membres qui sollicitent le plus MARIA. Alimente le widget `get_server_stats`.

Bufferisé en mémoire (aucune écriture disque dans le chemin d'un message individuel) ;
un flush périodique (cf. `cogs/chat/chat.py`) écrit en base et purge les jours trop
vieux. Granularité journalière : suffisant pour des classements sur quelques jours,
beaucoup plus léger qu'un log par message.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("MARIA.Activity")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "activity.db"
RETENTION_DAYS = 30  # purge au-delà — les classements portent sur quelques jours

KIND_MESSAGE = "message"
KIND_SUMMON = "summon"

_BufferKey = tuple[str, int, int, int, str]  # (day, guild_id, channel_id, user_id, kind)


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity (
                day        TEXT NOT NULL,
                guild_id   INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                user_id    INTEGER NOT NULL,
                kind       TEXT NOT NULL,
                count      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, guild_id, channel_id, user_id, kind)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_guild_day ON activity(guild_id, day)"
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


class ActivityTracker:
    """`bump_*` est synchrone et pur mémoire (safe à appeler depuis `on_message`).
    `flush()` et les requêtes `top_*`/`guild_message_count` font de l'I/O disque —
    à appeler via `asyncio.to_thread` depuis le code async."""

    def __init__(self):
        _init_db()
        self._buffer: Counter[_BufferKey] = Counter()

    def bump(self, guild_id: int, channel_id: int, user_id: int, kind: str = KIND_MESSAGE) -> None:
        today = date.today().isoformat()
        self._buffer[(today, guild_id, channel_id, user_id, kind)] += 1

    def bump_message(self, guild_id: int, channel_id: int, user_id: int) -> None:
        self.bump(guild_id, channel_id, user_id, KIND_MESSAGE)

    def bump_summon(self, guild_id: int, channel_id: int, user_id: int) -> None:
        self.bump(guild_id, channel_id, user_id, KIND_SUMMON)

    def flush(self) -> None:
        """Écrit le buffer en base (upsert additif) et le vide, puis purge les jours
        trop vieux. En cas d'échec d'écriture, les compteurs sont remis dans le buffer
        pour ne rien perdre au prochain passage."""
        if not self._buffer:
            return
        items = list(self._buffer.items())
        self._buffer.clear()
        cutoff = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
        try:
            with _db() as conn:
                for (day, guild_id, channel_id, user_id, kind), n in items:
                    conn.execute(
                        """
                        INSERT INTO activity (day, guild_id, channel_id, user_id, kind, count)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(day, guild_id, channel_id, user_id, kind)
                        DO UPDATE SET count = count + excluded.count
                        """,
                        (day, guild_id, channel_id, user_id, kind, n),
                    )
                conn.execute("DELETE FROM activity WHERE day < ?", (cutoff,))
        except sqlite3.Error as e:
            logger.error("Flush activité échoué : %s", e, exc_info=True)
            for key, n in items:
                self._buffer[key] += n

    def guild_message_count(self, guild_id: int, *, days: int = 7) -> int:
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        with _db() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(count), 0) AS n FROM activity "
                "WHERE guild_id = ? AND day >= ? AND kind = ?",
                (guild_id, cutoff, KIND_MESSAGE),
            ).fetchone()
        return int(row["n"]) if row else 0

    def top_channels(self, guild_id: int, *, days: int = 7, limit: int = 3) -> list[tuple[int, int]]:
        return self._top(guild_id, "channel_id", KIND_MESSAGE, days=days, limit=limit)

    def top_users(
        self, guild_id: int, *, days: int = 7, limit: int = 3, kind: str = KIND_MESSAGE,
    ) -> list[tuple[int, int]]:
        return self._top(guild_id, "user_id", kind, days=days, limit=limit)

    def _top(
        self, guild_id: int, group_col: str, kind: str, *, days: int, limit: int,
    ) -> list[tuple[int, int]]:
        # group_col est toujours une constante interne ("channel_id" / "user_id"),
        # jamais une entrée utilisateur — l'interpolation ci-dessous est sûre.
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        with _db() as conn:
            rows = conn.execute(
                f"""
                SELECT {group_col} AS grp, SUM(count) AS n
                FROM activity
                WHERE guild_id = ? AND day >= ? AND kind = ?
                GROUP BY {group_col}
                ORDER BY n DESC
                LIMIT ?
                """,
                (guild_id, cutoff, kind, limit),
            ).fetchall()
        return [(int(r["grp"]), int(r["n"])) for r in rows]
