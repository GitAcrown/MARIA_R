"""Cog Spotify — Recherche de morceaux via l'API Spotify (Client Credentials)."""

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.discord_ui import layout_with_commentary, section_with_thumbnail
from common.emojis import MUSIC
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.widgets import register_widget, unregister_widget

logger = logging.getLogger("MARIA.Spotify")

TOKEN_URL  = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

# Marge de sécurité avant expiration réelle du token (évite un 401 en plein appel).
_TOKEN_SAFETY_MARGIN = 60


def _fmt_duration(ms: int) -> str:
    total_seconds = ms // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _best_image(images: list[dict]) -> Optional[str]:
    if not images:
        return None
    # Les images Spotify sont déjà triées par taille décroissante ; on prend la plus petite
    # qui reste correcte pour une vignette (évite de télécharger inutilement du 640x640).
    sorted_imgs = sorted(images, key=lambda i: i.get("width") or 0)
    for img in sorted_imgs:
        if (img.get("width") or 0) >= 300:
            return img.get("url")
    return images[0].get("url")


# ---------------------------------------------------------------------------
# Résumé LLM
# ---------------------------------------------------------------------------

def _track_llm_summary(t: dict) -> str:
    name    = t.get("name", "?")
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    album   = (t.get("album") or {}).get("name", "")
    year    = ((t.get("album") or {}).get("release_date") or "")[:4]
    dur     = t.get("duration_ms")
    pop     = t.get("popularity")
    url     = (t.get("external_urls") or {}).get("spotify", "")

    parts = [f"Morceau Spotify : {name}"]
    if artists:
        parts.append(f"par {artists}")
    if album:
        parts.append(f"Album : {album}" + (f" ({year})" if year else ""))
    if isinstance(dur, int):
        parts.append(f"Durée : {_fmt_duration(dur)}")
    if isinstance(pop, int):
        parts.append(f"Popularité : {pop}/100")
    if t.get("explicit"):
        parts.append("Explicite")
    if url:
        parts.append(f"Lien : {url}")
    parts.append("Widget affiché.")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Builder LayoutView
# ---------------------------------------------------------------------------

def build_track_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView pour un morceau Spotify."""
    if "error" in data or "result" not in data:
        return None
    container = _track_container(data["result"])
    if container is None:
        return None
    return layout_with_commentary(container, commentary)


def _track_container(t: dict) -> Optional[discord.ui.Container]:
    name    = t.get("name", "?")
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    album   = (t.get("album") or {}).get("name", "")
    year    = ((t.get("album") or {}).get("release_date") or "")[:4]
    dur     = t.get("duration_ms")
    cover   = _best_image((t.get("album") or {}).get("images") or [])
    url     = (t.get("external_urls") or {}).get("spotify", "")
    explicit = t.get("explicit", False)

    # Titre + artiste + méta dans le même bloc (à côté de la pochette) pour éviter
    # une section clairsemée quand il y a peu de texte à afficher.
    body_lines = [f"## {MUSIC} {name}"]
    if artists:
        body_lines.append(f"**{artists}**")
    meta = []
    if album:
        meta.append(album)
    if year:
        meta.append(year)
    if isinstance(dur, int):
        meta.append(_fmt_duration(dur))
    if explicit:
        meta.append("🅴")
    if meta:
        body_lines.append(f"-# {' · '.join(meta)}")
    body_block = discord.ui.TextDisplay("\n".join(body_lines))
    main_section = section_with_thumbnail(body_block, cover)

    children: list = [main_section]
    if url:
        children += [discord.ui.TextDisplay(f"-# [Écouter sur Spotify]({url})")]

    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Spotify(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._client_id: str = getattr(bot, "config", {}).get("SPOTIFY_CLIENT_ID", "") or ""
        self._client_secret: str = getattr(bot, "config", {}).get("SPOTIFY_CLIENT_SECRET", "") or ""
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def _fetch_token(self) -> Optional[str]:
        """Récupère (et met en cache) un token via le flow Client Credentials.

        Ce flow n'expose que des données publiques (recherche, fiches) — pas de
        données utilisateur — donc pas besoin d'OAuth ni de refresh token.
        """
        if self._token and time.time() < self._token_expires_at:
            return self._token

        creds = f"{self._client_id}:{self._client_secret}".encode()
        b64_creds = base64.b64encode(creds).decode()
        try:
            r = requests.post(
                TOKEN_URL,
                headers={
                    "Authorization": f"Basic {b64_creds}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
                timeout=8,
            )
            if not r.ok:
                logger.warning("Auth Spotify échouée : HTTP %s", r.status_code)
                return None
            payload = r.json()
            token = payload.get("access_token")
            expires_in = payload.get("expires_in", 3600)
            if not token:
                return None
            self._token = token
            self._token_expires_at = time.time() + expires_in - _TOKEN_SAFETY_MARGIN
            return token
        except requests.RequestException as e:
            logger.warning("Auth Spotify échouée : %s", e)
            return None

    def _search_track(self, query: str) -> dict:
        if not self._client_id or not self._client_secret:
            return {"error": "Clés Spotify manquantes (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET dans .env)"}

        token = self._fetch_token()
        if not token:
            return {"error": "Authentification Spotify impossible"}

        try:
            r = requests.get(
                SEARCH_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={"q": query, "type": "track", "market": "FR", "limit": 1},
                timeout=8,
            )
            if r.status_code == 401:
                # Token invalidé côté Spotify avant l'expiration attendue : on force un refresh.
                self._token = None
                return {"error": "Session Spotify expirée, réessaie."}
            if not r.ok:
                return {"error": f"Erreur Spotify {r.status_code}"}
            items = (r.json().get("tracks") or {}).get("items") or []
            if not items:
                return {"error": f"Morceau introuvable sur Spotify : {query!r}"}
            return {"first": items[0]}
        except requests.RequestException as e:
            return {"error": str(e)}

    async def _tool_search_track(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        query = (tc.arguments.get("query") or "").strip()
        if not query:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))

        search = await asyncio.to_thread(self._search_track, query)
        if "error" in search:
            return ToolResponseRecord(tc.id, search, datetime.now(timezone.utc))

        result      = search["first"]
        llm_summary = _track_llm_summary(result)

        return ToolResponseRecord(tc.id, {
            "_tool":        "search_track",
            "_llm_summary": llm_summary,
            "result":       result,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="search_track",
                description=(
                    "Recherche un morceau sur Spotify et affiche sa fiche "
                    "(artiste, album, année, durée, pochette, lien Spotify). "
                    "Pour identifier une chanson (« c'est qui qui chante… », « le son avec tel refrain »), "
                    "ou afficher la fiche d'un titre précis. "
                    "Inclure artiste + titre si connus pour un meilleur match."
                ),
                properties={
                    "query": {
                        "type":        "string",
                        "description": "Titre du morceau, idéalement avec l'artiste (ex: 'Blinding Lights The Weeknd')",
                    },
                },
                function=self._tool_search_track,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Spotify(bot))
    register_widget("search_track", build_track_view)


async def teardown(bot: commands.Bot) -> None:
    unregister_widget("search_track")
