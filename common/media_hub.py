"""Dual channel média : widget public + menu éphémère (Plus / DynamicItem)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

import discord

from common.dyn_widgets import _now, get_payload, store_payload
from common.menu_layout import (
    MariaLayout,
    apply_view,
    send_ephemeral_menu,
    sep_tight,
)

logger = logging.getLogger("MARIA.MediaHub")

_MAX_HITS = 5
_PAYLOAD_CHARS = 12000


def _shrink(obj):
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return {}
    if len(raw) <= _PAYLOAD_CHARS:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(obj, dict):
        keep = {}
        for key in (
            "id", "steam_appid", "media_type", "title", "name", "overview",
            "short_description", "release_date", "first_air_date", "vote_average",
            "vote_count", "poster_path", "genres", "runtime", "number_of_seasons",
            "original_language", "artists", "album", "duration_ms", "popularity",
            "explicit", "external_urls", "price_overview", "price", "is_free",
            "developers", "review_score_desc", "header_image", "website",
            "fixture", "teams", "goals", "league", "_events", "_statistics",
            "mode", "result", "results",
        ):
            if key in obj:
                keep[key] = obj[key]
        try:
            json.dumps(keep, ensure_ascii=False, default=str)
            return json.loads(json.dumps(keep, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return {"name": obj.get("name") or obj.get("title") or "?"}
    return {}


def _hit_label(kind: str, item: dict, index: int) -> str:
    if kind == "tmdb":
        title = item.get("title") or item.get("name") or f"#{index + 1}"
        year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
        return f"{title} ({year})" if year else str(title)
    if kind == "steam":
        return str(item.get("name") or f"Jeu {index + 1}")
    if kind == "spotify":
        name = item.get("name") or f"Titre {index + 1}"
        artists = ", ".join(
            a.get("name", "") for a in (item.get("artists") or []) if isinstance(a, dict)
        )
        return f"{name} — {artists}" if artists else str(name)
    if kind == "football":
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if home and away:
            return f"{home} – {away}"
        return str(item.get("title") or f"Match {index + 1}")
    return item.get("name") or item.get("title") or f"#{index + 1}"


def _extra_text(kind: str, result: dict, summary: str) -> str:
    if summary:
        body = summary.strip()
        if len(body) > 900:
            body = body[:900].rstrip() + "…"
        return body
    if kind == "tmdb":
        overview = (result.get("overview") or "").strip()
        mid = result.get("id")
        mtype = result.get("media_type") or "movie"
        link = f"https://www.themoviedb.org/{mtype}/{mid}" if mid else ""
        return "\n".join(x for x in (overview[:800], link) if x)
    if kind == "steam":
        desc = (result.get("short_description") or result.get("about_the_game") or "").strip()
        appid = result.get("steam_appid") or result.get("id")
        link = f"https://store.steampowered.com/app/{appid}" if appid else ""
        return "\n".join(x for x in (desc[:800], link) if x)
    if kind == "spotify":
        url = ((result.get("external_urls") or {}).get("spotify") or "").strip()
        album = ((result.get("album") or {}).get("name") or "").strip()
        return "\n".join(x for x in (album, url) if x)
    return ""


def attach_media_actions(
    view: Optional[discord.ui.LayoutView],
    *,
    kind: str,
    result: dict,
    hits: Optional[list] = None,
    summary: str = "",
    extra: str = "",
) -> Optional[discord.ui.LayoutView]:
    """Ajoute le bouton Plus (DynamicItem) sur le widget public."""
    if view is None:
        return None
    slim_hits = []
    for item in (hits or [])[:_MAX_HITS]:
        if isinstance(item, dict):
            slim_hits.append(_shrink(item))
    payload = {
        "kind": kind,
        "result": _shrink(result),
        "hits": slim_hits,
        "selected": 0,
        "summary": (summary or "")[:1500],
        "extra": (extra or "")[:1500],
    }
    try:
        wid = store_payload("media_hub", payload)
    except Exception:
        logger.exception("store media hub")
        return view
    view.add_item(discord.ui.ActionRow(MediaActionsButton(wid)))
    return view


def _inject_artifact(interaction: discord.Interaction, text: str, kind: str = "widget") -> None:
    chat = interaction.client.get_cog("Chat")
    if chat is None or not hasattr(chat, "gpt_api"):
        return
    channel = interaction.channel
    if channel is None:
        return
    try:
        session = chat.gpt_api.session_manager.get_or_create(channel)
        session.record_artifact(kind, text)
    except Exception:
        logger.debug("artifact média ignoré", exc_info=True)


class MediaActionsButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"maria:media:(?P<wid>[0-9a-f]{8})",
):
    def __init__(self, wid: str) -> None:
        super().__init__(
            discord.ui.Button(
                style=discord.ButtonStyle.secondary,
                label="Plus",
                custom_id=f"maria:media:{wid}",
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
        return cls(match["wid"])

    async def callback(self, interaction: discord.Interaction) -> None:
        rec = get_payload(self.wid)
        if rec is None or rec.stripped or rec.expires_at <= _now():
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "Ce menu a expiré.", ephemeral=True,
                )
            return
        try:
            view = MediaSessionView(
                rec.payload,
                viewer_id=interaction.user.id,
            )
            view._build()
            await send_ephemeral_menu(interaction, view)
        except Exception:
            logger.exception("menu média")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "Impossible d'ouvrir le menu.", ephemeral=True,
                )
            return
        summary = (rec.payload.get("summary") or "").strip()
        if summary:
            _inject_artifact(interaction, summary, "widget")


class _HitSelect(discord.ui.Select):
    def __init__(self, parent: "MediaSessionView", labels: list[str], selected: int):
        options = [
            discord.SelectOption(
                label=(lab or f"#{i + 1}")[:100],
                value=str(i),
                default=(i == selected),
            )
            for i, lab in enumerate(labels[:25])
        ]
        super().__init__(placeholder="Autre résultat…", min_values=1, max_values=1, options=options)
        self._hub = parent

    async def callback(self, interaction: discord.Interaction) -> None:
        try:
            idx = int((self.values or ["0"])[0])
        except ValueError:
            idx = 0
        await self._hub.select_hit(interaction, idx)


class MediaSessionView(MariaLayout):
    """Hub éphémère : autres résultats + texte en plus (pas de 2e Container)."""

    def __init__(self, payload: dict, *, viewer_id: int):
        super().__init__(viewer_id=viewer_id)
        self.kind = (payload.get("kind") or "tmdb").strip()
        self.result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        self.hits = [h for h in (payload.get("hits") or []) if isinstance(h, dict)]
        self.selected = int(payload.get("selected") or 0)
        self.summary = (payload.get("summary") or "").strip()
        self.extra = (payload.get("extra") or "").strip()
        if self.hits and 0 <= self.selected < len(self.hits):
            current = self.hits[self.selected]
            if current.get("id") and current.get("id") == self.result.get("id"):
                pass
            elif not self.result:
                self.result = current

    def _current(self) -> dict:
        if self.hits and 0 <= self.selected < len(self.hits):
            hit = self.hits[self.selected]
            rid = self.result.get("id") or self.result.get("steam_appid")
            hid = hit.get("id") or hit.get("steam_appid")
            if rid is not None and hid is not None and str(rid) == str(hid):
                return self.result
            return hit
        return self.result

    def _build(self) -> None:
        current = self._current()
        labels = [_hit_label(self.kind, h, i) for i, h in enumerate(self.hits)]
        title = _hit_label(self.kind, current, self.selected)
        extra = self.extra or _extra_text(self.kind, current, self.summary)
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {title}"),
            sep_tight(),
            discord.ui.TextDisplay(extra or "-# Rien de plus sur cette fiche."),
        ]
        rows: list[discord.ui.Item] = []
        if len(labels) >= 2:
            rows.append(discord.ui.ActionRow(_HitSelect(self, labels, self.selected)))
        self.set_layout(body, *rows)

    async def select_hit(self, interaction: discord.Interaction, idx: int) -> None:
        if not self.hits:
            return
        self.selected = max(0, min(len(self.hits) - 1, idx))
        hit = self.hits[self.selected]
        fetched = await _fetch_details(interaction, self.kind, hit)
        self.result = fetched or hit
        self.extra = ""
        self.summary = ""
        self._build()
        await apply_view(interaction, self)
        _inject_artifact(
            interaction,
            f"Résultat média : {_hit_label(self.kind, self.result, self.selected)}",
            "widget",
        )


async def _fetch_details(interaction: discord.Interaction, kind: str, hit: dict) -> Optional[dict]:
    client = interaction.client
    try:
        if kind == "tmdb":
            cog = client.get_cog("TMDB")
            mid = hit.get("id")
            mtype = hit.get("media_type") or "movie"
            if cog is None or not mid:
                return None
            details = await asyncio.to_thread(cog._get_details, int(mid), mtype)
            return {**hit, **(details or {}), "media_type": mtype}
        if kind == "steam":
            cog = client.get_cog("Steam")
            appid = hit.get("id") or hit.get("steam_appid")
            if cog is None or not appid:
                return None
            details = await asyncio.to_thread(cog._get_details, int(appid))
            return {**hit, **(details or {})}
        if kind == "spotify":
            return hit
    except Exception:
        logger.debug("détails média", exc_info=True)
    return None
