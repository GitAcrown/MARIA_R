"""Suggestions intelligentes générées par l'IA passive.

Une suggestion est une proposition personnelle (rappel ou mise à jour de profil)
détectée passivement dans la conversation.
Elle reste en `pending` jusqu'à validation/refus via `/suggestions`.
"""

import hashlib
import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("MARIA.Suggestions")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "suggestions.db"

# Types de suggestions reconnus (strictement personnel)
KIND_PERSONAL_REMINDER = "personal_reminder"
KIND_PROFILE_UPDATE = "profile_update"

ALL_KINDS = (KIND_PERSONAL_REMINDER, KIND_PROFILE_UPDATE)

# Durée de vie d'une suggestion non traitée
DEFAULT_TTL_DAYS = 3  # 72 h
# Plafond de suggestions en attente par utilisateur.
# Quand le plafond est atteint, la plus ancienne est évincée pour faire place à la nouvelle.
MAX_PENDING_REMINDERS = 5   # rappels
MAX_PENDING_PROFILES = 5    # notes de profil


@dataclass
class Suggestion:
    id: int
    kind: str
    status: str
    guild_id: int
    channel_id: int
    target_user_id: Optional[int]
    payload: dict[str, Any]
    source_excerpt: str
    created_at: datetime
    expires_at: datetime
    signature: str = ""

    @property
    def description(self) -> str:
        """Texte court résumant la suggestion (selon le type)."""
        p = self.payload
        return (
            p.get("description")
            or p.get("title")
            or p.get("info")
            or "(sans description)"
        )


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS suggestions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                kind           TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                guild_id       INTEGER NOT NULL,
                channel_id     INTEGER NOT NULL,
                target_user_id INTEGER,
                payload        TEXT NOT NULL DEFAULT '{}',
                source_excerpt TEXT NOT NULL DEFAULT '',
                signature      TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                expires_at     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sugg_status_kind ON suggestions(status, kind)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sugg_target ON suggestions(status, target_user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sugg_guild ON suggestions(status, guild_id)"
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


def _row_to_suggestion(r: sqlite3.Row) -> Suggestion:
    try:
        payload = json.loads(r["payload"]) if r["payload"] else {}
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return Suggestion(
        id=r["id"],
        kind=r["kind"],
        status=r["status"],
        guild_id=r["guild_id"],
        channel_id=r["channel_id"],
        target_user_id=r["target_user_id"],
        payload=payload,
        source_excerpt=r["source_excerpt"] or "",
        created_at=datetime.fromisoformat(r["created_at"]),
        expires_at=datetime.fromisoformat(r["expires_at"]),
        signature=r["signature"],
    )


def make_signature(kind: str, target_user_id: Optional[int], text: str) -> str:
    """Empreinte de dedup : type + cible + texte normalisé."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    raw = f"{kind}|{target_user_id or 0}|{normalized}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SuggestionStore:
    def __init__(self) -> None:
        _init_db()

    def add(
        self,
        *,
        kind: str,
        guild_id: int,
        channel_id: int,
        target_user_id: int,
        payload: dict[str, Any],
        source_excerpt: str = "",
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> Optional[int]:
        """Insère une suggestion personnelle. Retourne l'id, ou None si doublon."""
        if kind not in ALL_KINDS:
            return None
        sig = make_signature(kind, target_user_id, payload.get("description") or payload.get("info") or "")
        cap = MAX_PENDING_REMINDERS if kind == KIND_PERSONAL_REMINDER else MAX_PENDING_PROFILES
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=ttl_days)
        with _db() as conn:
            dup = conn.execute(
                "SELECT 1 FROM suggestions WHERE status IN ('pending','rejected') AND signature=?",
                (sig,),
            ).fetchone()
            if dup:
                return None

            # Éviction FIFO si le plafond par type est atteint.
            count_row = conn.execute(
                "SELECT COUNT(*) FROM suggestions WHERE status='pending' AND target_user_id=? AND kind=?",
                (target_user_id, kind),
            ).fetchone()
            if (count_row[0] or 0) >= cap:
                conn.execute(
                    "DELETE FROM suggestions WHERE id = ("
                    "  SELECT id FROM suggestions WHERE status='pending'"
                    "  AND target_user_id=? AND kind=? ORDER BY created_at ASC LIMIT 1"
                    ")",
                    (target_user_id, kind),
                )

            cur = conn.execute(
                "INSERT INTO suggestions"
                " (kind, status, guild_id, channel_id, target_user_id, payload, source_excerpt, signature, created_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    kind, "pending", guild_id, channel_id, target_user_id,
                    json.dumps(payload, ensure_ascii=False), source_excerpt[:500], sig,
                    now.isoformat(), expires.isoformat(),
                ),
            )
            return cur.lastrowid

    def get(self, suggestion_id: int) -> Optional[Suggestion]:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM suggestions WHERE id=?", (suggestion_id,)
            ).fetchone()
        return _row_to_suggestion(row) if row else None

    def list_reminders(self, user_id: int) -> list[Suggestion]:
        """Suggestions de rappel en attente pour un utilisateur."""
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM suggestions WHERE status='pending'"
                " AND target_user_id=? AND kind=?"
                " ORDER BY created_at",
                (user_id, KIND_PERSONAL_REMINDER),
            ).fetchall()
        return [_row_to_suggestion(r) for r in rows]

    def list_profiles(self, user_id: int) -> list[Suggestion]:
        """Suggestions de mise à jour de profil en attente pour un utilisateur."""
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM suggestions WHERE status='pending'"
                " AND target_user_id=? AND kind=?"
                " ORDER BY created_at",
                (user_id, KIND_PROFILE_UPDATE),
            ).fetchall()
        return [_row_to_suggestion(r) for r in rows]

    def set_status(self, suggestion_id: int, status: str, *, user_id: int) -> bool:
        """Change le statut. Vérifie l'appartenance à user_id."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE suggestions SET status=? WHERE id=? AND target_user_id=? AND status='pending'",
                (status, suggestion_id, user_id),
            )
            return cur.rowcount > 0

    def pending_descriptions(self, guild_id: int, limit: int = 30) -> list[str]:
        """Descriptions des suggestions en attente ou refusées d'une guild (pour le prompt LLM).

        Inclut les refusées pour que le modèle évite de les reproposer.
        """
        with _db() as conn:
            rows = conn.execute(
                "SELECT kind, payload, status FROM suggestions"
                " WHERE status IN ('pending','rejected') AND guild_id=?"
                " ORDER BY created_at DESC LIMIT ?",
                (guild_id, limit),
            ).fetchall()
        out: list[str] = []
        for r in rows:
            try:
                p = json.loads(r["payload"]) if r["payload"] else {}
            except (json.JSONDecodeError, TypeError):
                p = {}
            desc = p.get("description") or p.get("info") or ""
            if desc:
                label = "refusé" if r["status"] == "rejected" else "en attente"
                out.append(f"[{r['kind']}|{label}] {desc}")
        return out

    def count_pending(self, user_id: int, kind: str) -> int:
        """Nombre de suggestions en attente d'un type donné pour un utilisateur."""
        with _db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM suggestions WHERE status='pending' AND target_user_id=? AND kind=?",
                (user_id, kind),
            ).fetchone()
        return row[0] if row else 0

    def expire_old(self) -> int:
        """Marque `expired` les suggestions en attente dont le TTL est dépassé."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE suggestions SET status='expired' WHERE status='pending' AND expires_at <= ?",
                (datetime.now(timezone.utc).isoformat(),),
            )
            return cur.rowcount
