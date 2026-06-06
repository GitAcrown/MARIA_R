"""Suggestions intelligentes générées par l'IA passive.

Une suggestion est une proposition (rappel personnel, événement serveur, mise à
jour de profil, activité de groupe) détectée passivement dans la conversation.
Elle reste en `pending` jusqu'à validation/refus via les commandes dédiées.
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

# Types de suggestions reconnus
KIND_PERSONAL_REMINDER = "personal_reminder"
KIND_SERVER_EVENT = "server_event"
KIND_PROFILE_UPDATE = "profile_update"
KIND_GROUP_ACTIVITY = "group_activity"

PERSONAL_KINDS = (KIND_PERSONAL_REMINDER, KIND_PROFILE_UPDATE)
EVENT_KINDS = (KIND_SERVER_EVENT, KIND_GROUP_ACTIVITY)

# Durée de vie d'une suggestion non traitée
DEFAULT_TTL_DAYS = 3  # 72 h
# Plafond de suggestions en attente par portée.
# Quand le plafond est atteint, la plus ancienne est évincée pour faire place à la nouvelle.
MAX_PENDING_PERSONAL = 10   # par utilisateur (toutes suggestions personnelles confondues)
MAX_PENDING_EVENTS = 8      # par guild (événements + activités de groupe)


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
        target_user_id: Optional[int],
        payload: dict[str, Any],
        source_excerpt: str = "",
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> Optional[int]:
        """Insère une suggestion. Retourne l'id, ou None si un doublon `pending` existe."""
        sig = make_signature(kind, target_user_id, payload.get("description") or payload.get("title") or payload.get("info") or "")
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=ttl_days)
        with _db() as conn:
            dup = conn.execute(
                "SELECT 1 FROM suggestions WHERE status IN ('pending','rejected') AND signature=?",
                (sig,),
            ).fetchone()
            if dup:
                return None

            # Éviction de la plus ancienne si le plafond est atteint.
            if kind in PERSONAL_KINDS and target_user_id is not None:
                count_row = conn.execute(
                    f"SELECT COUNT(*) FROM suggestions WHERE status='pending'"
                    f" AND target_user_id=? AND kind IN ({','.join('?'*len(PERSONAL_KINDS))})",
                    (target_user_id, *PERSONAL_KINDS),
                ).fetchone()
                if (count_row[0] or 0) >= MAX_PENDING_PERSONAL:
                    conn.execute(
                        f"DELETE FROM suggestions WHERE id = ("
                        f"  SELECT id FROM suggestions WHERE status='pending'"
                        f"  AND target_user_id=? AND kind IN ({','.join('?'*len(PERSONAL_KINDS))})"
                        f"  ORDER BY created_at ASC LIMIT 1"
                        f")",
                        (target_user_id, *PERSONAL_KINDS),
                    )
            elif kind in EVENT_KINDS:
                count_row = conn.execute(
                    f"SELECT COUNT(*) FROM suggestions WHERE status='pending'"
                    f" AND guild_id=? AND kind IN ({','.join('?'*len(EVENT_KINDS))})",
                    (guild_id, *EVENT_KINDS),
                ).fetchone()
                if (count_row[0] or 0) >= MAX_PENDING_EVENTS:
                    conn.execute(
                        f"DELETE FROM suggestions WHERE id = ("
                        f"  SELECT id FROM suggestions WHERE status='pending'"
                        f"  AND guild_id=? AND kind IN ({','.join('?'*len(EVENT_KINDS))})"
                        f"  ORDER BY created_at ASC LIMIT 1"
                        f")",
                        (guild_id, *EVENT_KINDS),
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

    def list_personal(self, user_id: int) -> list[Suggestion]:
        """Suggestions personnelles en attente pour un utilisateur (rappels, profil)."""
        placeholders = ",".join("?" * len(PERSONAL_KINDS))
        with _db() as conn:
            rows = conn.execute(
                f"SELECT * FROM suggestions WHERE status='pending'"
                f" AND target_user_id=? AND kind IN ({placeholders})"
                f" ORDER BY created_at",
                (user_id, *PERSONAL_KINDS),
            ).fetchall()
        return [_row_to_suggestion(r) for r in rows]

    def list_events(self, guild_id: int) -> list[Suggestion]:
        """Suggestions d'événements/activités en attente pour une guild."""
        placeholders = ",".join("?" * len(EVENT_KINDS))
        with _db() as conn:
            rows = conn.execute(
                f"SELECT * FROM suggestions WHERE status='pending'"
                f" AND guild_id=? AND kind IN ({placeholders})"
                f" ORDER BY created_at",
                (guild_id, *EVENT_KINDS),
            ).fetchall()
        return [_row_to_suggestion(r) for r in rows]

    def set_status(self, suggestion_id: int, status: str, *, user_id: Optional[int] = None) -> bool:
        """Change le statut. Si user_id est fourni, vérifie l'appartenance (suggestions perso)."""
        with _db() as conn:
            if user_id is not None:
                cur = conn.execute(
                    "UPDATE suggestions SET status=? WHERE id=? AND target_user_id=? AND status='pending'",
                    (status, suggestion_id, user_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE suggestions SET status=? WHERE id=? AND status='pending'",
                    (status, suggestion_id),
                )
            return cur.rowcount > 0

    def pending_signatures(self, *, target_user_id: Optional[int] = None, guild_id: Optional[int] = None) -> set[str]:
        """Empreintes des suggestions déjà en attente (pour informer le modèle)."""
        query = "SELECT signature FROM suggestions WHERE status='pending'"
        params: list[Any] = []
        if target_user_id is not None:
            query += " AND target_user_id=?"
            params.append(target_user_id)
        if guild_id is not None:
            query += " AND guild_id=?"
            params.append(guild_id)
        with _db() as conn:
            rows = conn.execute(query, params).fetchall()
        return {r[0] for r in rows}

    def pending_descriptions(self, guild_id: int, limit: int = 40) -> list[str]:
        """Descriptions courtes des suggestions en attente ou refusées d'une guild (pour le prompt).

        Les suggestions refusées sont également incluses pour que le modèle évite de les reproposer.
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
            desc = p.get("description") or p.get("title") or p.get("info") or ""
            if desc:
                label = "refusé" if r["status"] == "rejected" else "en attente"
                out.append(f"[{r['kind']}|{label}] {desc}")
        return out

    def count_pending(self, *, target_user_id: Optional[int] = None, guild_id: Optional[int] = None, kinds: tuple[str, ...] = ()) -> int:
        query = "SELECT COUNT(*) FROM suggestions WHERE status='pending'"
        params: list[Any] = []
        if target_user_id is not None:
            query += " AND target_user_id=?"
            params.append(target_user_id)
        if guild_id is not None:
            query += " AND guild_id=?"
            params.append(guild_id)
        if kinds:
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        with _db() as conn:
            row = conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def expire_old(self) -> int:
        """Marque `expired` les suggestions en attente dont le TTL est dépassé."""
        with _db() as conn:
            cur = conn.execute(
                "UPDATE suggestions SET status='expired' WHERE status='pending' AND expires_at <= ?",
                (datetime.now(timezone.utc).isoformat(),),
            )
            return cur.rowcount
