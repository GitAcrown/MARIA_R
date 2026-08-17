"""Favoris de layouts (`render_widget`) — bouton 10 min + stockage persistant."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import discord

from common.emojis import BOOKMARK
from common.widget_catalog import render_free_widget

logger = logging.getLogger("MARIA.Bookmarks")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "bookmarks.db"
TTL = timedelta(minutes=10)
PURGE_AFTER = timedelta(hours=1)
VIEW_ATTR = "_maria_bm_id"
BOOKMARK_MAX = 30
_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_EMOJI = discord.PartialEmoji.from_str(BOOKMARK)


@dataclass
class Bookmark:
    id: str
    user_id: int
    title: str
    spec: dict
    created_at: datetime


@dataclass
class _Pending:
    id: str
    spec: dict
    commentary: str
    title: str
    channel_id: int
    message_id: int
    expires_at: datetime
    stripped: bool


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def _init_db() -> None:
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending (
                id          TEXT PRIMARY KEY,
                spec        TEXT NOT NULL,
                commentary  TEXT DEFAULT '',
                title       TEXT DEFAULT '',
                channel_id  INTEGER DEFAULT 0,
                message_id  INTEGER DEFAULT 0,
                expires_at  TEXT NOT NULL,
                stripped    INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm_pending_exp ON pending(stripped, expires_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookmarks (
                id          TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                title       TEXT NOT NULL,
                spec        TEXT NOT NULL,
                spec_hash   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE(user_id, spec_hash)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm_user ON bookmarks(user_id, created_at)"
        )


def spec_title(spec: dict) -> str:
    title = str(spec.get("title") or "").strip()
    if title:
        return title[:80]
    for raw in spec.get("blocks") or []:
        if not isinstance(raw, dict) or (raw.get("type") or "") != "text":
            continue
        content = str(raw.get("content") or "").strip().replace("\n", " ")
        if content:
            return content[:80]
    return "Layout"


def spec_hash(spec: dict) -> str:
    blob = json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _loads(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _pending_from_row(r: sqlite3.Row) -> _Pending:
    return _Pending(
        id=r["id"],
        spec=_loads(r["spec"]),
        commentary=r["commentary"] or "",
        title=r["title"] or "",
        channel_id=int(r["channel_id"] or 0),
        message_id=int(r["message_id"] or 0),
        expires_at=_as_utc(datetime.fromisoformat(r["expires_at"])),
        stripped=bool(r["stripped"]),
    )


def _bookmark_from_row(r: sqlite3.Row) -> Bookmark:
    return Bookmark(
        id=r["id"],
        user_id=int(r["user_id"]),
        title=r["title"] or "Layout",
        spec=_loads(r["spec"]),
        created_at=_as_utc(datetime.fromisoformat(r["created_at"])),
    )


def _insert_pending(spec: dict, commentary: str, title: str) -> str:
    wid = uuid.uuid4().hex[:8]
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO pending (id, spec, commentary, title, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                wid,
                json.dumps(spec, ensure_ascii=False),
                commentary or "",
                title,
                (_now() + TTL).isoformat(),
            ),
        )
    return wid


def _get_pending(wid: str) -> Optional[_Pending]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM pending WHERE id = ?", (wid,)).fetchone()
    return _pending_from_row(row) if row else None


def _bind_pending(wid: str, channel_id: int, message_id: int) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE pending SET channel_id = ?, message_id = ? WHERE id = ?",
            (channel_id, message_id, wid),
        )


def _mark_stripped(wid: str) -> None:
    with _db() as conn:
        conn.execute("UPDATE pending SET stripped = 1 WHERE id = ?", (wid,))


def _due_unstripped(now: datetime) -> list[_Pending]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM pending WHERE stripped = 0 AND expires_at <= ?",
            (now.isoformat(),),
        ).fetchall()
    return [_pending_from_row(r) for r in rows]


def _purge(now: datetime) -> None:
    cut = (now - PURGE_AFTER).isoformat()
    with _db() as conn:
        conn.execute(
            "DELETE FROM pending WHERE stripped = 1 AND expires_at <= ?",
            (cut,),
        )
        conn.execute(
            "DELETE FROM pending WHERE message_id = 0 AND expires_at <= ?",
            (now.isoformat(),),
        )


def count_for_user(user_id: int) -> int:
    with _db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM bookmarks WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["n"] if row else 0)


def list_for_user(user_id: int) -> list[Bookmark]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM bookmarks WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [_bookmark_from_row(r) for r in rows]


def get_bookmark(bid: str, user_id: int) -> Optional[Bookmark]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM bookmarks WHERE id = ? AND user_id = ?",
            (bid, user_id),
        ).fetchone()
    return _bookmark_from_row(row) if row else None


def search_for_user(user_id: int, query: str) -> list[Bookmark]:
    q = (query or "").strip().casefold()
    if not q:
        return list_for_user(user_id)
    out: list[Bookmark] = []
    for bm in list_for_user(user_id):
        blob = json.dumps(bm.spec, ensure_ascii=False).casefold()
        if q in bm.title.casefold() or q in blob:
            out.append(bm)
    return out


def save_for_user(user_id: int, spec: dict, *, title: str = "") -> tuple[str, Optional[Bookmark]]:
    """Enregistre. Retourne (ok|dup|full|bad, bookmark ou None)."""
    if not isinstance(spec, dict) or not spec.get("blocks"):
        return "bad", None
    if count_for_user(user_id) >= BOOKMARK_MAX:
        return "full", None
    digest = spec_hash(spec)
    label = (title or spec_title(spec)).strip()[:80] or "Layout"
    bid = uuid.uuid4().hex[:8]
    now = _now().isoformat()
    try:
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO bookmarks (id, user_id, title, spec, spec_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    bid,
                    user_id,
                    label,
                    json.dumps(spec, ensure_ascii=False),
                    digest,
                    now,
                ),
            )
    except sqlite3.IntegrityError:
        return "dup", None
    return "ok", Bookmark(
        id=bid, user_id=user_id, title=label, spec=spec, created_at=_as_utc(datetime.fromisoformat(now)),
    )


def delete_bookmark(bid: str, user_id: int) -> bool:
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE id = ? AND user_id = ?",
            (bid, user_id),
        )
        return cur.rowcount > 0


def _layout_without_button(pending: _Pending) -> Optional[discord.ui.LayoutView]:
    return render_free_widget(pending.spec, commentary=pending.commentary)


def attach_bookmark_button(
    view: discord.ui.LayoutView,
    spec: Optional[dict],
    *,
    commentary: str = "",
) -> None:
    """Ajoute le bouton *sous* le layout (hors Container). No-op si spec invalide."""
    if not isinstance(spec, dict) or not spec.get("blocks"):
        return
    try:
        json.dumps(spec)
    except (TypeError, ValueError):
        logger.warning("spec bookmark non sérialisable")
        return
    title = spec_title(spec)
    wid = _insert_pending(spec, commentary, title)
    view.add_item(discord.ui.ActionRow(BookmarkButton(wid)))
    setattr(view, VIEW_ATTR, wid)


async def bind(view: discord.ui.LayoutView, message: discord.Message) -> None:
    wid = getattr(view, VIEW_ATTR, None)
    if not isinstance(wid, str) or not _ID_RE.match(wid):
        return
    _bind_pending(wid, message.channel.id, message.id)


async def _publish(bot: discord.Client, rec: _Pending) -> bool:
    view = _layout_without_button(rec)
    if view is None or not rec.channel_id or not rec.message_id:
        return False
    try:
        channel = bot.get_channel(rec.channel_id)
        if channel is None:
            channel = await bot.fetch_channel(rec.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return False
        msg = await channel.fetch_message(rec.message_id)
        await msg.edit(view=view)
        return True
    except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
        logger.info("bookmark strip %s : %s", rec.id, e)
        return False


async def sweep_expired(bot: discord.Client) -> None:
    now = _now()
    for rec in _due_unstripped(now):
        if rec.message_id:
            await _publish(bot, rec)
        _mark_stripped(rec.id)
    _purge(now)


class BookmarkButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"maria:bm:(?P<wid>[0-9a-f]{8})",
):
    def __init__(self, wid: str, *, emoji: Optional[discord.PartialEmoji] = None) -> None:
        super().__init__(
            discord.ui.Button(
                style=discord.ButtonStyle.secondary,
                emoji=emoji or _EMOJI,
                custom_id=f"maria:bm:{wid}",
            )
        )
        self.wid = wid

    @classmethod
    async def from_custom_id(
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Button,
        match: re.Match[str],
        /,
    ):
        return cls(match["wid"], emoji=item.emoji or _EMOJI)

    async def callback(self, interaction: discord.Interaction) -> None:
        rec = _get_pending(self.wid)
        if rec is None or rec.stripped or rec.expires_at <= _now():
            if rec is not None and not rec.stripped:
                view = _layout_without_button(rec)
                if view is not None:
                    try:
                        await interaction.response.edit_message(view=view)
                    except discord.HTTPException:
                        pass
                _mark_stripped(rec.id)
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "Le bouton a expiré.", ephemeral=True,
                )
            return
        status, _ = save_for_user(interaction.user.id, rec.spec, title=rec.title)
        if status == "ok":
            msg = f"{BOOKMARK} Enregistré — `/signets` pour le retrouver."
        elif status == "dup":
            msg = "Déjà dans tes signets."
        elif status == "full":
            msg = f"Limite atteinte ({BOOKMARK_MAX}). Supprime-en un dans `/signets`."
        else:
            msg = "Impossible d'enregistrer ce layout."
        await interaction.response.send_message(msg, ephemeral=True)


_init_db()
