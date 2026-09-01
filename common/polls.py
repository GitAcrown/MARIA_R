"""Sondages actifs créés par MARIA (`create_poll`) — persistance légère.

But : que le LLM reste conscient d'un sondage en cours dans un salon bien au-delà
de la fenêtre `[CONTEXTE RÉCENT]` (~20 min), sans devoir le rechercher. Discord ne
donne pas d'API pour lister les sondages actifs d'un salon — on garde donc notre
propre registre (question, options, échéance), sans dupliquer les votes eux-mêmes
(gérés nativement par Discord).

Limite connue : perdu au redémarrage du bot si un sondage était encore actif à ce
moment-là (rare vu la durée habituelle) — pas de résurrection depuis l'historique.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "polls.db"


@dataclass
class ActivePoll:
    message_id: int
    channel_id: int
    guild_id: int
    author_id: int
    question: str
    options: list[str]
    created_at: datetime
    expires_at: datetime


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS polls (
                message_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL,
                guild_id   INTEGER NOT NULL,
                author_id  INTEGER NOT NULL,
                question   TEXT NOT NULL,
                options    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_polls_channel ON polls(channel_id, expires_at)"
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


def _row_to_poll(r: sqlite3.Row) -> ActivePoll:
    return ActivePoll(
        message_id=r["message_id"],
        channel_id=r["channel_id"],
        guild_id=r["guild_id"],
        author_id=r["author_id"],
        question=r["question"],
        options=json.loads(r["options"]),
        created_at=datetime.fromisoformat(r["created_at"]),
        expires_at=datetime.fromisoformat(r["expires_at"]),
    )


class PollStore:
    """Toutes les méthodes font de l'I/O disque — à appeler via `asyncio.to_thread`."""

    def __init__(self):
        _init_db()

    def create(
        self, *, message_id: int, channel_id: int, guild_id: int, author_id: int,
        question: str, options: list[str], expires_at: datetime,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with _db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO polls
                    (message_id, channel_id, guild_id, author_id, question, options, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id, channel_id, guild_id, author_id, question,
                    json.dumps(options, ensure_ascii=False), now, expires_at.isoformat(),
                ),
            )

    def active_for_channel(self, channel_id: int) -> list[ActivePoll]:
        self._purge_expired()
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM polls WHERE channel_id = ? ORDER BY created_at DESC",
                (channel_id,),
            ).fetchall()
        return [_row_to_poll(r) for r in rows]

    def _purge_expired(self) -> None:
        with _db() as conn:
            conn.execute(
                "DELETE FROM polls WHERE expires_at < ?",
                (datetime.now(timezone.utc).isoformat(),),
            )
