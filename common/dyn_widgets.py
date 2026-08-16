"""Widgets à onglets — boutons persistants (DynamicItem) qui disparaissent après 10 min.

Survit au redémarrage : custom_id `maria:tab:{id}:{index}` + SQLite.
À l'échéance (ou au clic trop tard), le message est réécrit sans boutons.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

import discord

logger = logging.getLogger("MARIA.DynWidgets")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "dyn_widgets.db"
TTL = timedelta(minutes=10)
PURGE_AFTER = timedelta(hours=1)
VIEW_ATTR = "_maria_dyn_id"
_ROW = 5
_MAX_TABS = 25
_BTN_LABEL_MAX = 16
_ID_RE = re.compile(r"^[0-9a-f]{8}$")

LabelsFn = Callable[[dict], list[str]]
BodyFn = Callable[[dict, int], discord.ui.Item]

_RENDERERS: dict[str, tuple[LabelsFn, BodyFn]] = {}
_FORCE_SELECT: set[str] = set()
_PLACEHOLDERS: dict[str, str] = {}


@dataclass
class _Record:
    id: str
    kind: str
    payload: dict
    commentary: str
    selected: int
    expires_at: datetime
    channel_id: int
    message_id: int
    stripped: bool


def register_tabs(
    kind: str,
    labels: LabelsFn,
    body: BodyFn,
    *,
    force_select: bool = False,
    placeholder: str = "",
) -> None:
    _RENDERERS[kind] = (labels, body)
    if force_select:
        _FORCE_SELECT.add(kind)
    else:
        _FORCE_SELECT.discard(kind)
    if placeholder:
        _PLACEHOLDERS[kind] = placeholder
    else:
        _PLACEHOLDERS.pop(kind, None)


def unregister_tabs(kind: str) -> None:
    _RENDERERS.pop(kind, None)
    _FORCE_SELECT.discard(kind)
    _PLACEHOLDERS.pop(kind, None)


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
            CREATE TABLE IF NOT EXISTS widgets (
                id          TEXT PRIMARY KEY,
                kind        TEXT NOT NULL,
                payload     TEXT NOT NULL,
                commentary  TEXT DEFAULT '',
                selected    INTEGER DEFAULT 0,
                expires_at  TEXT NOT NULL,
                channel_id  INTEGER DEFAULT 0,
                message_id  INTEGER DEFAULT 0,
                stripped    INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dyn_exp ON widgets(stripped, expires_at)"
        )


def _row_to_rec(r: sqlite3.Row) -> _Record:
    try:
        payload = json.loads(r["payload"] or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return _Record(
        id=r["id"],
        kind=r["kind"],
        payload=payload,
        commentary=r["commentary"] or "",
        selected=int(r["selected"] or 0),
        expires_at=_as_utc(datetime.fromisoformat(r["expires_at"])),
        channel_id=int(r["channel_id"] or 0),
        message_id=int(r["message_id"] or 0),
        stripped=bool(r["stripped"]),
    )


def _get(wid: str) -> Optional[_Record]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM widgets WHERE id = ?", (wid,)).fetchone()
    return _row_to_rec(row) if row else None


def _insert(kind: str, payload: dict, commentary: str, selected: int) -> str:
    wid = uuid.uuid4().hex[:8]
    expires = (_now() + TTL).isoformat()
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO widgets
                (id, kind, payload, commentary, selected, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (wid, kind, json.dumps(payload, ensure_ascii=False), commentary, selected, expires),
        )
    return wid


def _set_selected(wid: str, selected: int) -> None:
    with _db() as conn:
        conn.execute("UPDATE widgets SET selected = ? WHERE id = ?", (selected, wid))


def _bind(wid: str, channel_id: int, message_id: int) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE widgets SET channel_id = ?, message_id = ? WHERE id = ?",
            (channel_id, message_id, wid),
        )


def _mark_stripped(wid: str) -> None:
    with _db() as conn:
        conn.execute("UPDATE widgets SET stripped = 1 WHERE id = ?", (wid,))


def _delete(wid: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM widgets WHERE id = ?", (wid,))


def _due_unstripped(now: datetime) -> list[_Record]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM widgets WHERE stripped = 0 AND expires_at <= ?",
            (now.isoformat(),),
        ).fetchall()
    return [_row_to_rec(r) for r in rows]


def _purge(now: datetime) -> None:
    cut = (now - PURGE_AFTER).isoformat()
    with _db() as conn:
        conn.execute(
            "DELETE FROM widgets WHERE stripped = 1 AND expires_at <= ?",
            (cut,),
        )
        conn.execute(
            "DELETE FROM widgets WHERE message_id = 0 AND expires_at <= ?",
            (now.isoformat(),),
        )


def _labels(kind: str, payload: dict) -> list[str]:
    pair = _RENDERERS.get(kind)
    if not pair:
        return []
    try:
        labels = pair[0](payload) or []
    except Exception:
        logger.exception("labels %s", kind)
        return []
    out: list[str] = []
    for lab in labels[:_MAX_TABS]:
        text = " ".join(str(lab).split())[:100]
        out.append(text or "·")
    return out


def _body(kind: str, payload: dict, index: int) -> Optional[discord.ui.Item]:
    pair = _RENDERERS.get(kind)
    if not pair:
        return None
    try:
        return pair[1](payload, index)
    except Exception:
        logger.exception("body %s[%s]", kind, index)
        return None


def _use_select(labels: list[str]) -> bool:
    """Select dès qu'un libellé ne tient pas dans un bouton (dates courtes → boutons)."""
    return any(len(lab) > _BTN_LABEL_MAX for lab in labels)


def _tab_rows(wid: str, labels: list[str], selected: int) -> list[discord.ui.ActionRow]:
    rows: list[discord.ui.ActionRow] = []
    for start in range(0, len(labels), _ROW):
        chunk = labels[start:start + _ROW]
        buttons = [
            TabButton(wid, start + i, label=lab, selected=(start + i) == selected)
            for i, lab in enumerate(chunk)
        ]
        rows.append(discord.ui.ActionRow(*buttons))
    return rows


def _tab_controls(
    wid: str, labels: list[str], selected: int, *, kind: str = "",
) -> list[discord.ui.ActionRow]:
    if kind in _FORCE_SELECT or _use_select(labels):
        ph = _PLACEHOLDERS.get(kind) or "Choisir…"
        return [discord.ui.ActionRow(TabSelect(wid, labels, selected, placeholder=ph))]
    return _tab_rows(wid, labels, selected)


def render_record(rec: _Record, *, live: bool) -> Optional[discord.ui.LayoutView]:
    labels = _labels(rec.kind, rec.payload)
    if not labels:
        body = _body(rec.kind, rec.payload, rec.selected)
        if body is None:
            return None
        view = discord.ui.LayoutView(timeout=None)
        if rec.commentary:
            view.add_item(discord.ui.TextDisplay(rec.commentary))
            view.add_item(discord.ui.Separator())
        view.add_item(body)
        return view
    index = rec.selected if 0 <= rec.selected < len(labels) else 0
    body = _body(rec.kind, rec.payload, index)
    if body is None:
        return None
    tabs_in_card = False
    if live and len(labels) >= 2 and isinstance(body, discord.ui.Container):
        old = list(body.children)
        body.clear_items()
        for row in _tab_controls(rec.id, labels, index, kind=rec.kind):
            body.add_item(row)
        body.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.small))
        for item in old:
            body.add_item(item)
        tabs_in_card = True
    view = discord.ui.LayoutView(timeout=None)
    if rec.commentary:
        view.add_item(discord.ui.TextDisplay(rec.commentary))
        view.add_item(discord.ui.Separator())
    if live and len(labels) >= 2 and not tabs_in_card:
        for row in _tab_controls(rec.id, labels, index, kind=rec.kind):
            view.add_item(row)
        view.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.small))
    view.add_item(body)
    if live:
        setattr(view, VIEW_ATTR, rec.id)
    return view


def make_tabbed_view(
    kind: str,
    payload: dict,
    commentary: str = "",
    selected: int = 0,
) -> Optional[discord.ui.LayoutView]:
    """Crée un widget à onglets et l'enregistre (TTL 10 min). None si pas assez d'onglets."""
    labels = _labels(kind, payload)
    if len(labels) < 2:
        return None
    index = selected if 0 <= selected < len(labels) else 0
    try:
        json.dumps(payload)
    except (TypeError, ValueError):
        logger.exception("payload onglets non sérialisable (%s)", kind)
        return None
    wid = _insert(kind, payload, commentary, index)
    rec = _Record(
        id=wid,
        kind=kind,
        payload=payload,
        commentary=commentary,
        selected=index,
        expires_at=_now() + TTL,
        channel_id=0,
        message_id=0,
        stripped=False,
    )
    return render_record(rec, live=True)


async def bind(view: discord.ui.LayoutView, message: discord.Message) -> None:
    wid = getattr(view, VIEW_ATTR, None)
    if not isinstance(wid, str) or not _ID_RE.match(wid):
        return
    _bind(wid, message.channel.id, message.id)


async def _publish(bot: discord.Client, rec: _Record, *, live: bool) -> bool:
    view = render_record(rec, live=live)
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
        logger.info("dyn strip %s : %s", rec.id, e)
        return False


async def sweep_expired(bot: discord.Client) -> None:
    now = _now()
    for rec in _due_unstripped(now):
        if rec.message_id:
            await _publish(bot, rec, live=False)
        _mark_stripped(rec.id)
    _purge(now)


async def _pick_tab(interaction: discord.Interaction, wid: str, idx: int) -> None:
    rec = _get(wid)
    if rec is None or rec.stripped or rec.expires_at <= _now():
        if rec is not None:
            view = render_record(rec, live=False)
            if view is not None:
                try:
                    await interaction.response.edit_message(view=view)
                except discord.HTTPException:
                    pass
                _mark_stripped(rec.id)
                return
            _mark_stripped(rec.id)
        if not interaction.response.is_done():
            await interaction.response.send_message(
                "Les onglets ont expiré.", ephemeral=True,
            )
        return
    labels = _labels(rec.kind, rec.payload)
    if not labels:
        return
    rec.selected = idx if 0 <= idx < len(labels) else 0
    _set_selected(rec.id, rec.selected)
    view = render_record(rec, live=True)
    if view is None:
        return
    await interaction.response.edit_message(view=view)


class TabButton(discord.ui.DynamicItem[discord.ui.Button], template=r"maria:tab:(?P<wid>[0-9a-f]{8}):(?P<idx>[0-9]+)"):
    def __init__(self, wid: str, idx: int, *, label: str = "·", selected: bool = False) -> None:
        super().__init__(
            discord.ui.Button(
                style=discord.ButtonStyle.primary if selected else discord.ButtonStyle.secondary,
                label=(label or "·")[:80],
                custom_id=f"maria:tab:{wid}:{idx}",
            )
        )
        self.wid = wid
        self.idx = idx

    @classmethod
    async def from_custom_id(
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Button,
        match: re.Match[str],
        /,
    ):
        return cls(match["wid"], int(match["idx"]), label=item.label or "·")

    async def callback(self, interaction: discord.Interaction) -> None:
        await _pick_tab(interaction, self.wid, self.idx)


class TabSelect(discord.ui.DynamicItem[discord.ui.Select], template=r"maria:pick:(?P<wid>[0-9a-f]{8})"):
    def __init__(
        self,
        wid: str,
        labels: list[str],
        selected: int = 0,
        *,
        placeholder: str = "Choisir…",
    ) -> None:
        options = [
            discord.SelectOption(
                label=(lab or "·")[:100],
                value=str(i),
                default=(i == selected),
            )
            for i, lab in enumerate(labels[:25])
        ]
        if not options:
            options = [discord.SelectOption(label="·", value="0")]
        super().__init__(
            discord.ui.Select(
                placeholder=(placeholder or "Choisir…")[:150],
                min_values=1,
                max_values=1,
                options=options,
                custom_id=f"maria:pick:{wid}",
            )
        )
        self.wid = wid

    @classmethod
    async def from_custom_id(
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Select,
        match: re.Match[str],
        /,
    ):
        labels = [opt.label for opt in (item.options or [])]
        selected = 0
        for i, opt in enumerate(item.options or []):
            if opt.default:
                selected = i
                break
        return cls(match["wid"], labels, selected, placeholder=item.placeholder or "Choisir…")

    async def callback(self, interaction: discord.Interaction) -> None:
        raw = (self.item.values or ["0"])[0]
        try:
            idx = int(raw)
        except ValueError:
            idx = 0
        await _pick_tab(interaction, self.wid, idx)


_init_db()
