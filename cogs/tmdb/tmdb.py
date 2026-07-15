"""Cog TMDB — Recherche de films et séries via The Movie Database."""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.discord_ui import layout_with_commentary, section_with_thumbnail
from common.emojis import MOVIE, TV
from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.TMDB")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG  = "https://image.tmdb.org/t/p/w500{}"

_TYPE_EMOJI = {"movie": MOVIE, "tv": TV}
_TYPE_LABEL = {"movie": "Film", "tv": "Série"}


def _poster_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return TMDB_IMG.format(path)


_YEAR_RE = re.compile(r"\s*[\(\[]?(19\d{2}|20\d{2})[\)\]]?\s*$")


def _parse_query_year(query: str) -> tuple[str, Optional[int]]:
    """Sépare un éventuel millésime en fin de requête : 'Backrooms (2026)' → ('Backrooms', 2026).

    /search/multi (contrairement à /search/movie ou /search/tv) n'a pas de paramètre
    année : le laisser dans le texte de recherche pollue le matching plutôt que de l'aider.
    """
    m = _YEAR_RE.search(query)
    if not m:
        return query.strip(), None
    year = int(m.group(1))
    cleaned = query[: m.start()].strip()
    return cleaned or query.strip(), year


def _pick_best_result(results: list[dict], query: str, year: Optional[int] = None) -> dict:
    """Choisit le meilleur résultat movie/tv (match exact + année → match exact → année → popularité).

    /search/multi renvoie un ordre de « pertinence » qui change avec language=
    et inclut des personnes. Après filtre, le 1er n'est pas toujours le bon —
    notamment pour un titre repris par plusieurs œuvres (ex: plusieurs "Backrooms").
    """
    q = query.strip().casefold()

    def title_of(r: dict) -> str:
        return (r.get("title") or r.get("name") or "").strip()

    def year_of(r: dict) -> Optional[int]:
        date_str = (r.get("release_date") or r.get("first_air_date") or "")[:4]
        return int(date_str) if date_str.isdigit() else None

    def is_exact(r: dict) -> bool:
        return (
            title_of(r).casefold() == q
            or (r.get("original_title") or r.get("original_name") or "").strip().casefold() == q
        )

    def score(r: dict) -> tuple[int, int, float]:
        exact = is_exact(r)
        year_match = year is not None and year_of(r) == year
        return (int(exact and year_match), int(exact or year_match), float(r.get("popularity") or 0.0))

    return max(results, key=score)


# ---------------------------------------------------------------------------
# Résumé LLM
# ---------------------------------------------------------------------------

def _media_llm_summary(r: dict) -> str:
    media_type = r.get("media_type", "movie")
    title      = r.get("title") or r.get("name", "?")
    date_str   = r.get("release_date") or r.get("first_air_date") or ""
    year       = date_str[:4] if date_str else ""
    rating     = r.get("vote_average", 0.0)
    overview   = (r.get("overview") or "").strip()
    genres     = [g["name"] for g in r.get("genres", [])]

    parts = [f"{_TYPE_LABEL.get(media_type, 'Média')} : {title}"]
    if year:
        parts.append(f"({year})")
    if genres:
        parts.append(f"Genres : {', '.join(genres[:3])}")
    if rating:
        parts.append(f"Note TMDB : {rating:.1f}/10")
    if r.get("runtime"):
        h, m = divmod(r["runtime"], 60)
        parts.append(f"Durée : {h}h{m:02d}" if h else f"Durée : {m} min")
    if r.get("number_of_seasons"):
        s = r["number_of_seasons"]
        parts.append(f"{s} saison{'s' if s > 1 else ''}")
    if overview:
        short = overview[:350] + "…" if len(overview) > 350 else overview
        parts.append(f"Synopsis : {short}")
    parts.append("Widget affiché.")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Builder LayoutView
# ---------------------------------------------------------------------------

def build_media_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView pour un film ou une série."""
    if "error" in data or "result" not in data:
        return None
    container = _media_container(data["result"])
    if container is None:
        return None
    return layout_with_commentary(container, commentary)


def _media_container(r: dict) -> Optional[discord.ui.Container]:
    media_type = r.get("media_type", "movie")
    title      = r.get("title") or r.get("name", "?")
    overview   = (r.get("overview") or "").strip()
    rating     = r.get("vote_average", 0.0)
    vote_count = r.get("vote_count", 0)
    date_str   = r.get("release_date") or r.get("first_air_date") or ""
    year       = date_str[:4] if date_str else ""
    genres     = [g["name"] for g in r.get("genres", [])]
    poster_url = _poster_url(r.get("poster_path"))

    emoji      = _TYPE_EMOJI.get(media_type, MOVIE)
    title_line = f"## {emoji} {title}"
    if year:
        title_line += f"  ·  {year}"

    meta_parts = [_TYPE_LABEL.get(media_type, "Média")] + genres[:3]
    header = discord.ui.TextDisplay(f"{title_line}\n-# {'  ·  '.join(meta_parts)}")
    sep1   = discord.ui.Separator()

    # Note + synopsis
    if rating and vote_count:
        stars     = "★" * round(rating / 2) + "☆" * (5 - round(rating / 2))
        rating_ln = f"{stars}  **{rating:.1f}/10**  ·  {vote_count:,} votes"
    elif rating:
        rating_ln = f"**{rating:.1f}/10**"
    else:
        rating_ln = ""

    overview_short = overview[:380] + "…" if len(overview) > 380 else overview
    body_text      = f"{rating_ln}\n{overview_short}" if rating_ln else overview_short
    body_block     = discord.ui.TextDisplay(body_text or "-# Aucune description disponible.")

    # Poster en MediaGallery (plus visible qu'une Thumbnail Section),
    # sinon fallback vignette si la galerie échoue.
    poster_item: Optional[discord.ui.Item] = None
    if poster_url:
        try:
            gallery = discord.ui.MediaGallery()
            gallery.add_item(media=poster_url, description=title[:256])
            poster_item = gallery
        except Exception:
            poster_item = section_with_thumbnail(body_block, poster_url)

    # Infos supplémentaires
    extra = []
    if r.get("runtime"):
        h, m = divmod(r["runtime"], 60)
        extra.append(f"{h}h{m:02d}" if h else f"{m} min")
    if r.get("number_of_seasons"):
        s = r["number_of_seasons"]
        extra.append(f"{s} saison{'s' if s > 1 else ''}")
    lang = r.get("original_language", "")
    if lang and lang != "fr":
        extra.append(lang.upper())

    tmdb_id = r.get("id")
    footer_parts = extra[:]
    if tmdb_id:
        footer_parts.append(f"[TMDB](https://www.themoviedb.org/{media_type}/{tmdb_id})")

    children: list = [header, sep1]
    if poster_item is not None and isinstance(poster_item, discord.ui.MediaGallery):
        children += [poster_item, discord.ui.Separator(), body_block]
    elif poster_item is not None:
        children.append(poster_item)  # Section avec thumbnail + body
    else:
        children.append(body_block)

    if footer_parts:
        children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# {'  ·  '.join(footer_parts)}")]
    else:
        children += [discord.ui.Separator(), discord.ui.TextDisplay("-# Source : The Movie Database (TMDB)")]

    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class TMDB(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot     = bot
        self._api_key: str = getattr(bot, "config", {}).get("TMDB_API_KEY", "") or ""

    def _search_multi(self, query: str) -> dict:
        if not self._api_key:
            return {"error": "Clé API TMDB manquante (TMDB_API_KEY dans .env)"}
        clean_query, year = _parse_query_year(query)
        try:
            r = requests.get(
                f"{TMDB_BASE}/search/multi",
                params={
                    "query":          clean_query,
                    "api_key":        self._api_key,
                    "language":       "fr-FR",
                    "include_adult":  False,
                },
                timeout=8,
            )
            if r.status_code == 401:
                return {"error": "Clé API TMDB invalide"}
            if not r.ok:
                return {"error": f"Erreur TMDB {r.status_code}"}
            results = [x for x in r.json().get("results", []) if x.get("media_type") in ("movie", "tv")]
            if not results:
                return {"error": f"Aucun résultat pour : {query!r}"}
            return {"first": _pick_best_result(results, clean_query, year)}
        except requests.RequestException as e:
            return {"error": str(e)}

    def _get_details(self, media_id: int, media_type: str) -> dict:
        # Échec non bloquant : on retourne {} et le résultat reste affichable
        # avec les seules données de recherche (fiche partielle).
        try:
            r = requests.get(
                f"{TMDB_BASE}/{media_type}/{media_id}",
                params={"api_key": self._api_key, "language": "fr-FR"},
                timeout=8,
            )
            if not r.ok:
                logger.warning("Détails TMDB indisponibles (%s/%s): HTTP %s", media_type, media_id, r.status_code)
                return {}
            return r.json()
        except requests.RequestException as e:
            logger.warning("Détails TMDB indisponibles (%s/%s): %s", media_type, media_id, e)
            return {}

    async def _tool_search_media(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        query = (tc.arguments.get("query") or "").strip()
        if not query:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))

        search = await asyncio.to_thread(self._search_multi, query)
        if "error" in search:
            return ToolResponseRecord(tc.id, search, datetime.now(timezone.utc))

        first      = search["first"]
        media_type = first.get("media_type", "movie")
        media_id   = first.get("id")

        details = {}
        if media_id:
            details = await asyncio.to_thread(self._get_details, media_id, media_type)

        result      = {**first, **details, "media_type": media_type}
        llm_summary = _media_llm_summary(result)

        return ToolResponseRecord(tc.id, {
            "_tool":        "search_media",
            "_llm_summary": llm_summary,
            "result":       result,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="search_media",
                description=(
                    "Recherche un film ou une série sur TMDB et affiche sa fiche "
                    "(titre, note, synopsis, genres, durée/saisons). "
                    "Utilise le titre exact ou le plus précis possible. "
                    "Si tu n'es pas sûr du titre (description vague, «le film avec X», titre approximatif), "
                    "utilise d'abord search_web pour identifier le bon titre, puis appelle search_media."
                ),
                properties={
                    "query": {
                        "type":        "string",
                        "description": (
                            "Titre du film ou de la série. "
                            "Ajouter l'année si ambigu (ex: 'Dune 2021')."
                        ),
                    },
                },
                function=self._tool_search_media,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(TMDB(bot))
