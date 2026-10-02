"""Usage des emojis custom appris via les réactions Discord."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
import zlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from common import emojis as chrome_emojis

logger = logging.getLogger("MARIA.EmojiUsage")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "emoji_usage.db"
SAMPLES_PER_EMOJI = 8
EXCERPT_MAX = 120
SHORTLIST_K = 5
_PAD_TO = 8

_TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëïîôùûüçœæ_]{2,}", re.IGNORECASE)
_CHROME_ID_RE = re.compile(r":(\d+)>")


def _chrome_emoji_ids() -> frozenset[int]:
    ids: set[int] = set()
    for val in vars(chrome_emojis).values():
        if not isinstance(val, str):
            continue
        m = _CHROME_ID_RE.search(val)
        if m:
            ids.add(int(m.group(1)))
    return frozenset(ids)


CHROME_EMOJI_IDS = _chrome_emoji_ids()


def unicode_emoji_id(char: str) -> int:
    """Id synthétique négatif et stable pour un emoji Unicode (les ids custom sont > 0)."""
    return -(zlib.crc32(char.encode("utf8")) + 1)


# Repli quand le serveur n'a pas encore assez d'historique : (emoji, usage).
DEFAULT_UNICODE: tuple[tuple[str, str], ...] = (
    ("😂", "funny joke, something amusing"),
    ("💀", "absurd or so funny it is deadly, cringe"),
    ("👍", "agreement, ok, acknowledgement"),
    ("❤️", "affection, kind or wholesome message"),
    ("😭", "dramatic sadness or laughing so hard it hurts"),
    ("👀", "something intriguing, gossip, looking at it"),
    ("🔥", "impressive, great news, hype"),
    ("🤔", "doubt, odd statement, questioning"),
)


@dataclass(frozen=True)
class EmojiCandidate:
    emoji_id: int
    name: str
    animated: bool
    count: int
    samples: tuple[str, ...]
    hint: str = ""

    @property
    def is_unicode(self) -> bool:
        return self.emoji_id < 0


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emoji_stats (
                guild_id   INTEGER NOT NULL,
                emoji_id   INTEGER NOT NULL,
                emoji_name TEXT NOT NULL,
                animated   INTEGER NOT NULL DEFAULT 0,
                count      INTEGER NOT NULL DEFAULT 0,
                last_seen  REAL NOT NULL,
                PRIMARY KEY (guild_id, emoji_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emoji_sample (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id   INTEGER NOT NULL,
                emoji_id   INTEGER NOT NULL,
                emoji_name TEXT NOT NULL,
                animated   INTEGER NOT NULL DEFAULT 0,
                excerpt    TEXT NOT NULL,
                seen_at    REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emoji_sample_guild_emoji "
            "ON emoji_sample(guild_id, emoji_id, seen_at)"
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


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in _TOKEN_RE.findall(text or "")}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


class EmojiUsageTracker:
    def __init__(self) -> None:
        _init_db()
        self._buf_stats: dict[tuple[int, int], tuple[str, bool, int, float]] = {}
        self._buf_samples: list[tuple[int, int, str, bool, str, float]] = []
        self._lock = threading.Lock()

    def observe(
        self,
        guild_id: int,
        *,
        emoji_id: Optional[int],
        emoji_name: str,
        animated: bool = False,
        excerpt: str = "",
    ) -> None:
        if emoji_id is None or emoji_id in CHROME_EMOJI_IDS:
            return
        name = (emoji_name or "").strip()
        if not name:
            return
        text = re.sub(r"\s+", " ", (excerpt or "").strip())[:EXCERPT_MAX]
        now = time.time()
        key = (guild_id, emoji_id)
        with self._lock:
            prev = self._buf_stats.get(key)
            if prev:
                _, _, n, _ = prev
                self._buf_stats[key] = (name, animated, n + 1, now)
            else:
                self._buf_stats[key] = (name, animated, 1, now)
            if text:
                self._buf_samples.append((guild_id, emoji_id, name, animated, text, now))

    def flush(self) -> None:
        with self._lock:
            stats = list(self._buf_stats.items())
            samples = list(self._buf_samples)
            self._buf_stats.clear()
            self._buf_samples.clear()
        if not stats and not samples:
            return
        try:
            with _db() as conn:
                for (guild_id, emoji_id), (name, animated, n, last_seen) in stats:
                    conn.execute(
                        """
                        INSERT INTO emoji_stats
                            (guild_id, emoji_id, emoji_name, animated, count, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(guild_id, emoji_id) DO UPDATE SET
                            emoji_name = excluded.emoji_name,
                            animated = excluded.animated,
                            count = count + excluded.count,
                            last_seen = excluded.last_seen
                        """,
                        (guild_id, emoji_id, name, int(animated), n, last_seen),
                    )
                for guild_id, emoji_id, name, animated, excerpt, seen_at in samples:
                    conn.execute(
                        """
                        INSERT INTO emoji_sample
                            (guild_id, emoji_id, emoji_name, animated, excerpt, seen_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (guild_id, emoji_id, name, int(animated), excerpt, seen_at),
                    )
                    rows = conn.execute(
                        """
                        SELECT id FROM emoji_sample
                        WHERE guild_id = ? AND emoji_id = ?
                        ORDER BY seen_at DESC, id DESC
                        """,
                        (guild_id, emoji_id),
                    ).fetchall()
                    if len(rows) > SAMPLES_PER_EMOJI:
                        drop = [r["id"] for r in rows[SAMPLES_PER_EMOJI:]]
                        conn.executemany(
                            "DELETE FROM emoji_sample WHERE id = ?",
                            [(i,) for i in drop],
                        )
        except sqlite3.Error as e:
            logger.error("Flush emoji_usage échoué : %s", e, exc_info=True)
            with self._lock:
                for key, (name, animated, n, last_seen) in stats:
                    prev = self._buf_stats.get(key)
                    if prev:
                        pn, pa, pc, _ = prev
                        self._buf_stats[key] = (pn or name, pa or animated, pc + n, last_seen)
                    else:
                        self._buf_stats[key] = (name, animated, n, last_seen)
                self._buf_samples.extend(samples)

    def shortlist(
        self,
        guild_id: int,
        message_text: str,
        *,
        k: int = SHORTLIST_K,
        unicode: bool = False,
    ) -> list[EmojiCandidate]:
        """Top k candidats : fréquence × overlap lexical avec les extraits.

        `unicode=False` : emojis custom du serveur uniquement.
        `unicode=True` : emojis classiques appris, complétés par les défauts.
        """
        self.flush()
        sign = "<" if unicode else ">"
        with _db() as conn:
            stats = conn.execute(
                f"""
                SELECT emoji_id, emoji_name, animated, count
                FROM emoji_stats
                WHERE guild_id = ? AND count > 0 AND emoji_id {sign} 0
                ORDER BY count DESC
                LIMIT 40
                """,
                (guild_id,),
            ).fetchall()
            samples_by: dict[int, list[str]] = defaultdict(list)
            for row in conn.execute(
                """
                SELECT emoji_id, excerpt FROM emoji_sample
                WHERE guild_id = ?
                ORDER BY seen_at DESC
                """,
                (guild_id,),
            ).fetchall():
                eid = int(row["emoji_id"])
                if len(samples_by[eid]) < SAMPLES_PER_EMOJI:
                    samples_by[eid].append(row["excerpt"])

        msg_tok = _tokens(message_text)
        scored: list[tuple[float, EmojiCandidate]] = []
        for row in stats:
            eid = int(row["emoji_id"])
            samples = tuple(samples_by.get(eid, [])[:3])
            sample_tok: set[str] = set()
            for s in samples:
                sample_tok |= _tokens(s)
            overlap = _overlap(msg_tok, sample_tok) if msg_tok else 0.0
            count = int(row["count"])
            # Froid (pas d'extrait / pas d'overlap) : ranking par fréquence seule.
            score = count * (1.0 + 4.0 * overlap)
            scored.append(
                (
                    score,
                    EmojiCandidate(
                        emoji_id=eid,
                        name=str(row["emoji_name"]),
                        animated=bool(row["animated"]),
                        count=count,
                        samples=samples,
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [c for _, c in scored[: max(1, k)]]
        if unicode and len(picked) < k:
            have = {c.emoji_id for c in picked}
            for char, hint in DEFAULT_UNICODE:
                eid = unicode_emoji_id(char)
                if eid in have:
                    continue
                picked.append(EmojiCandidate(
                    emoji_id=eid, name=char, animated=False, count=0, samples=(), hint=hint,
                ))
                if len(picked) >= _PAD_TO:
                    break
        return picked
