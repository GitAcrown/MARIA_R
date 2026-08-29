"""Cog YouTube — lecture des sous-titres (gratuit, pas d'API key)."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from common.discord_ui import layout_with_commentary, section_with_thumbnail
from common.emojis import YOUTUBE
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.widgets import register_widget, unregister_widget

logger = logging.getLogger("MARIA.YouTube")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
OEMBED_URL = "https://www.youtube.com/oembed"
CACHE_SEC = 3600
TRANSCRIPT_MAX = 8000
_LANGS = ("fr", "fr-FR", "en", "en-US", "en-GB")

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:[^#]*&)?v=|shorts/|live/|embed/|v/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.I,
)
_YT_ID_ONLY = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_video_id(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    if _YT_ID_ONLY.fullmatch(text):
        return text
    m = _YT_ID_RE.search(text)
    return m.group(1) if m else None


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _fmt_ts(seconds: float) -> str:
    t = max(0, int(seconds))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_snippets(snippets, *, max_chars: int) -> tuple[str, bool, float]:
    lines: list[str] = []
    last_mark = -999.0
    end = 0.0
    for sn in snippets:
        text = html.unescape((getattr(sn, "text", None) or "")).replace("\n", " ").strip()
        if not text:
            continue
        start = float(getattr(sn, "start", 0) or 0)
        dur = float(getattr(sn, "duration", 0) or 0)
        end = max(end, start + dur)
        if start - last_mark >= 30:
            lines.append(f"[{_fmt_ts(start)}] {text}")
            last_mark = start
        elif lines:
            lines[-1] += f" {text}"
        else:
            lines.append(text)
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full, False, end
    cut = full[:max_chars]
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary < max_chars * 0.6:
        boundary = max_chars
    return cut[:boundary].rstrip() + "…", True, end


def _oembed(video_id: str) -> dict:
    try:
        r = requests.get(
            OEMBED_URL,
            params={"url": _watch_url(video_id), "format": "json"},
            headers=HEADERS,
            timeout=(4, 8),
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fetch_captions(video_id: str) -> dict:
    api = YouTubeTranscriptApi()
    try:
        listing = api.list(video_id)
    except TranscriptsDisabled:
        return {"error": "Sous-titres désactivés sur cette vidéo."}
    except VideoUnavailable:
        return {"error": "Vidéo introuvable, privée ou indisponible."}
    except AgeRestricted:
        return {"error": "Vidéo limitée par l'âge, sous-titres inaccessibles."}
    except (IpBlocked, RequestBlocked):
        return {"error": "YouTube a bloqué la lecture des sous-titres. Réessaie plus tard."}
    except Exception as e:
        logger.warning("list(%s): %s", video_id, e)
        return {"error": "Impossible de lister les sous-titres."}

    transcript = None
    try:
        transcript = listing.find_transcript(_LANGS)
    except NoTranscriptFound:
        transcript = next(iter(listing), None)
    except Exception:
        transcript = next(iter(listing), None)
    if transcript is None:
        return {"error": "Aucun sous-titre disponible pour cette vidéo."}

    try:
        fetched = transcript.fetch()
    except NoTranscriptFound:
        return {"error": "Aucun sous-titre dans une langue lisible (fr/en)."}
    except (IpBlocked, RequestBlocked):
        return {"error": "YouTube a bloqué la lecture des sous-titres. Réessaie plus tard."}
    except Exception as e:
        logger.warning("fetch(%s): %s", video_id, e)
        return {"error": "Lecture des sous-titres échouée."}

    body, truncated, duration_s = _format_snippets(fetched.snippets, max_chars=TRANSCRIPT_MAX)
    if not body.strip():
        return {"error": "Sous-titres vides."}

    lang = fetched.language_code or getattr(transcript, "language_code", "") or "?"
    generated = bool(fetched.is_generated)
    kind = "auto" if generated else "manuel"
    meta = (
        f"Vidéo YouTube {video_id} · sous-titres {lang} ({kind})"
        f" · durée ~{_fmt_ts(duration_s)}"
        + (" · TRONQUÉ (début seulement)" if truncated else "")
    )
    llm = f"{meta}\n\nSOUS-TITRES :\n{body}"
    return {
        "video_id": video_id,
        "language": lang,
        "generated": generated,
        "truncated": truncated,
        "duration_s": int(duration_s),
        "transcript": body,
        "_llm_summary": llm,
    }


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

def build_youtube_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if not isinstance(data, dict) or "error" in data or not data.get("video_id"):
        return None
    container = _youtube_container(data)
    if container is None:
        return None
    return layout_with_commentary(container, commentary)


def _youtube_container(data: dict) -> Optional[discord.ui.Container]:
    video_id = data.get("video_id") or ""
    title = (data.get("title") or "").strip() or f"Vidéo {video_id}"
    author = (data.get("author") or "").strip()
    thumb = (data.get("thumbnail") or "").strip() or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    url = (data.get("url") or _watch_url(video_id)).strip()
    lang = (data.get("language") or "").strip()
    generated = bool(data.get("generated"))
    duration_s = data.get("duration_s")

    body_lines = [f"## {YOUTUBE} {title}"]
    if author:
        body_lines.append(f"**{author}**")
    meta: list[str] = []
    if isinstance(duration_s, int) and duration_s > 0:
        meta.append(_fmt_ts(duration_s))
    cap = "Sous-titres auto" if generated else "Sous-titres"
    if lang:
        cap += f" · {lang}"
    meta.append(cap)
    if data.get("truncated"):
        meta.append("début seulement")
    body_lines.append(f"-# {' · '.join(meta)}")
    main = section_with_thumbnail(discord.ui.TextDisplay("\n".join(body_lines)), thumb)

    children: list = [main]
    if url.startswith("http"):
        children.append(discord.ui.TextDisplay(f"-# [Ouvrir sur YouTube]({url})"))
    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Youtube(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._cache: dict[str, tuple[dict, float]] = {}

    def _load(self, video_id: str) -> dict:
        now = time.time()
        hit = self._cache.get(video_id)
        if hit and now - hit[1] < CACHE_SEC:
            return dict(hit[0])

        caps = _fetch_captions(video_id)
        if "error" in caps:
            return caps

        meta = _oembed(video_id)
        title = (meta.get("title") or "").strip()
        author = (meta.get("author_name") or "").strip()
        thumb = (meta.get("thumbnail_url") or "").strip()
        url = _watch_url(video_id)

        header_bits = [title or f"Vidéo {video_id}"]
        if author:
            header_bits.append(f"par {author}")
        llm = caps.get("_llm_summary") or ""
        if title:
            llm = f"{' — '.join(header_bits)}\n{llm}"

        payload = {
            "_tool": "read_youtube",
            "_llm_summary": llm,
            "video_id": video_id,
            "url": url,
            "title": title,
            "author": author,
            "thumbnail": thumb,
            "language": caps.get("language") or "",
            "generated": bool(caps.get("generated")),
            "truncated": bool(caps.get("truncated")),
            "duration_s": caps.get("duration_s") or 0,
        }
        self._cache[video_id] = (payload, now)
        return dict(payload)

    async def _tool_read(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        raw = (tc.arguments.get("url") or "").strip()
        if not raw:
            return ToolResponseRecord(
                tc.id, {"error": "URL YouTube manquante"}, datetime.now(timezone.utc),
            )
        video_id = parse_video_id(raw)
        if not video_id:
            return ToolResponseRecord(
                tc.id, {"error": "Lien YouTube invalide (watch / youtu.be / shorts)."},
                datetime.now(timezone.utc),
            )
        data = await asyncio.to_thread(self._load, video_id)
        return ToolResponseRecord(tc.id, data, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="read_youtube",
                description=(
                    "Lit les sous-titres d'une vidéo YouTube (piste auteur ou auto-générée). "
                    "Pour résumer, citer, ou répondre à une question sur la vidéo. "
                    "Ce n'est PAS regarder l'image ni écouter le son. "
                    "Pas de sous-titres → dis-le, n'invente pas. "
                    "Pas pour un fichier vidéo Discord (ça, tu ne peux pas)."
                ),
                properties={
                    "url": {
                        "type": "string",
                        "description": "URL YouTube (watch, youtu.be, shorts) ou id 11 caractères",
                    },
                },
                function=self._tool_read,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Youtube(bot))
    register_widget("read_youtube", build_youtube_view)


async def teardown(bot: commands.Bot) -> None:
    unregister_widget("read_youtube")
