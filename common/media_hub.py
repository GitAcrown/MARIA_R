"""Fiches média : select sur le widget public s'il y a plusieurs résultats (comme la météo)."""

from __future__ import annotations

import json
import logging
from typing import Optional

import discord

from common.discord_ui import layout_with_commentary
from common.dyn_widgets import make_tabbed_view, register_tabs, unregister_tabs

logger = logging.getLogger("MARIA.MediaHub")

_MAX_HITS = 5
_PAYLOAD_CHARS = 12000
_KIND = "media_hub"


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
            "developers", "review_score_desc", "header_image",
        ):
            if key in obj:
                keep[key] = obj[key]
        try:
            return json.loads(json.dumps(keep, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return {"name": obj.get("name") or obj.get("title") or "?"}
    return {}


def _item_id(item: dict) -> str:
    val = item.get("id") or item.get("steam_appid")
    return str(val) if val is not None else ""


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
    return item.get("name") or item.get("title") or f"#{index + 1}"


def _container_for(kind: str, result: dict):
    if not result:
        return None
    try:
        if kind == "tmdb":
            from cogs.tmdb.tmdb import _media_container
            return _media_container(result)
        if kind == "steam":
            from cogs.steam.steam import _game_container
            return _game_container(result)
        if kind == "spotify":
            from cogs.spotify.spotify import _track_container
            return _track_container(result)
    except Exception:
        logger.exception("container média %s", kind)
    return None


def _align_hits(result: dict, hits: list) -> list[dict]:
    cleaned = [h for h in (hits or []) if isinstance(h, dict)]
    if result and not cleaned:
        return [result]
    rid = _item_id(result) if result else ""
    out: list[dict] = []
    if result:
        out.append(result)
    seen = {rid} if rid else set()
    for h in cleaned:
        hid = _item_id(h)
        if hid and hid in seen:
            continue
        if hid:
            seen.add(hid)
        out.append(h)
    return out[:_MAX_HITS]


def media_tab_labels(payload: dict) -> list[str]:
    kind = payload.get("kind") or ""
    hits = payload.get("hits") or []
    return [_hit_label(kind, h, i) for i, h in enumerate(hits) if isinstance(h, dict)]


def media_tab_body(payload: dict, index: int) -> discord.ui.Item:
    kind = payload.get("kind") or "tmdb"
    hits = [h for h in (payload.get("hits") or []) if isinstance(h, dict)]
    hit = hits[index] if 0 <= index < len(hits) else (hits[0] if hits else {})
    return _container_for(kind, hit) or discord.ui.TextDisplay("-# Fiche illisible.")


def build_media_layout(
    *,
    kind: str,
    result: dict,
    hits: Optional[list] = None,
    commentary: str = "",
) -> Optional[discord.ui.LayoutView]:
    """Une fiche seule, ou un select sur le widget s'il y a plusieurs hits."""
    aligned = _align_hits(result, hits or [])
    if len(aligned) < 2:
        container = _container_for(kind, aligned[0] if aligned else result)
        if container is None:
            return None
        return layout_with_commentary(container, commentary)
    payload = {
        "kind": kind,
        "hits": [_shrink(h) for h in aligned],
    }
    view = make_tabbed_view(
        _KIND, payload, commentary, 0,
    )
    if view is not None:
        return view
    container = _container_for(kind, aligned[0])
    return layout_with_commentary(container, commentary) if container else None


def register_media_tabs() -> None:
    register_tabs(
        _KIND, media_tab_labels, media_tab_body,
        force_select=True, placeholder="Choisir un résultat",
    )


def unregister_media_tabs() -> None:
    unregister_tabs(_KIND)


register_media_tabs()
