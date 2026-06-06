"""Cog Steam — Recherche de jeux via le Steam Store (sans clé API)."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.discord_ui import layout_with_commentary, section_with_thumbnail
from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Steam")

STEAM_SEARCH  = "https://store.steampowered.com/api/storesearch/"
STEAM_DETAILS = "https://store.steampowered.com/api/appdetails"
STEAM_HEADER  = "https://cdn.akamai.steamstatic.com/steam/apps/{}/header.jpg"
STEAM_STORE   = "https://store.steampowered.com/app/{}"

_REVIEW_MAP = {
    "Overwhelmingly Positive": ("🥇", "Acclamé"),
    "Very Positive":           ("👍", "Très positif"),
    "Positive":                ("👍", "Positif"),
    "Mostly Positive":         ("🙂", "Plutôt positif"),
    "Mixed":                   ("😐", "Mitigé"),
    "Mostly Negative":         ("👎", "Plutôt négatif"),
    "Negative":                ("👎", "Négatif"),
    "Very Negative":           ("💀", "Très négatif"),
    "Overwhelmingly Negative": ("💀", "Descendu en flammes"),
}


def _fmt_price(cents: int) -> str:
    return f"{cents / 100:.2f} €"


# ---------------------------------------------------------------------------
# Résumé LLM
# ---------------------------------------------------------------------------

def _game_llm_summary(r: dict) -> str:
    name        = r.get("name", "?")
    appid       = r.get("steam_appid") or r.get("id")
    price_data  = r.get("price_overview") or r.get("price") or {}
    is_free     = r.get("is_free", False)
    short_desc  = (r.get("short_description") or "").strip()
    genres      = [g["description"] for g in r.get("genres", [])]
    devs        = r.get("developers", [])
    review_desc = r.get("review_score_desc", "")

    parts = [f"Jeu Steam : {name}"]

    if is_free:
        parts.append("Gratuit")
    elif price_data:
        final    = price_data.get("final")
        discount = price_data.get("discount_percent", 0)
        if isinstance(final, int):
            parts.append(f"Prix : {_fmt_price(final)}")
        if discount:
            initial = price_data.get("initial")
            if isinstance(initial, int):
                parts.append(f"En solde -{ discount }% (était {_fmt_price(initial)})")

    if review_desc:
        _, label = _REVIEW_MAP.get(review_desc, ("", review_desc))
        if label:
            parts.append(f"Avis : {label}")

    if genres:
        parts.append(f"Genres : {', '.join(genres[:3])}")
    if devs:
        parts.append(f"Développeur : {devs[0]}")
    if short_desc:
        short = short_desc[:280] + "…" if len(short_desc) > 280 else short_desc
        parts.append(f"Description : {short}")
    if appid:
        parts.append(f"Lien : {STEAM_STORE.format(appid)}")

    parts.append("Widget affiché.")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Builder LayoutView
# ---------------------------------------------------------------------------

def build_game_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView pour un jeu Steam."""
    if "error" in data or "result" not in data:
        return None
    container = _game_container(data["result"])
    if container is None:
        return None
    return layout_with_commentary(container, commentary)


def _game_container(r: dict) -> Optional[discord.ui.Container]:
    appid      = r.get("steam_appid") or r.get("id")
    name       = r.get("name", "?")
    is_free    = r.get("is_free", False)
    price_data = r.get("price_overview") or r.get("price") or {}
    short_desc = (r.get("short_description") or "").strip()
    genres     = [g["description"] for g in r.get("genres", [])]
    devs       = r.get("developers", [])
    review_desc = r.get("review_score_desc", "")

    # Prix
    if is_free:
        price_str = "**Gratuit**"
    elif price_data:
        final    = price_data.get("final")
        discount = price_data.get("discount_percent", 0)
        if isinstance(final, int):
            if discount and discount > 0:
                initial     = price_data.get("initial")
                initial_str = f" ~~{_fmt_price(initial)}~~" if isinstance(initial, int) else ""
                price_str   = f"**{_fmt_price(final)}**{initial_str}  🏷️ **-{discount}%**"
            else:
                price_str = f"**{_fmt_price(final)}**"
        else:
            price_str = str(final) if final else "Prix inconnu"
    else:
        price_str = "Prix inconnu"

    # Avis
    review_emoji, review_label = _REVIEW_MAP.get(review_desc, ("", review_desc or ""))
    review_str = f"{review_emoji} {review_label}".strip() if review_label else ""

    # Corps central
    short_desc_short = short_desc[:260] + "…" if len(short_desc) > 260 else short_desc
    body_lines = [price_str]
    if review_str:
        body_lines.append(review_str)
    if short_desc_short:
        body_lines.append(f"-# {short_desc_short}")
    body_block = discord.ui.TextDisplay("\n".join(body_lines))

    # Thumbnail header Steam
    main_section = section_with_thumbnail(body_block, STEAM_HEADER.format(appid) if appid else None)

    # Header + séparateur
    header = discord.ui.TextDisplay(f"## 🎮 {name}")
    sep1   = discord.ui.Separator()

    # Footer méta
    meta_parts = []
    if genres:
        meta_parts.append(", ".join(genres[:3]))
    if devs:
        meta_parts.append(f"par {devs[0]}")
    if appid:
        meta_parts.append(f"[Steam]({STEAM_STORE.format(appid)})")

    children: list = [header, sep1, main_section]
    if meta_parts:
        children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# {'  ·  '.join(meta_parts)}")]

    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Steam(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _search(self, query: str) -> dict:
        try:
            r = requests.get(
                STEAM_SEARCH,
                params={"term": query, "cc": "fr", "l": "french", "num": 5},
                timeout=8,
            )
            if not r.ok:
                return {"error": f"Erreur Steam Store {r.status_code}"}
            items = r.json().get("items", [])
            if not items:
                return {"error": f"Jeu introuvable sur Steam : {query!r}"}
            return {"first": items[0]}
        except requests.RequestException as e:
            return {"error": str(e)}

    def _get_details(self, appid: int) -> dict:
        # Échec non bloquant : on retourne {} et le jeu reste affichable
        # avec les seules données de recherche (fiche partielle).
        try:
            r = requests.get(
                STEAM_DETAILS,
                params={"appids": appid, "cc": "fr", "l": "french"},
                timeout=8,
            )
            if not r.ok:
                logger.warning("Détails Steam indisponibles (appid %s): HTTP %s", appid, r.status_code)
                return {}
            payload = r.json().get(str(appid), {})
            if not payload.get("success"):
                logger.warning("Détails Steam indisponibles (appid %s): réponse non valide", appid)
                return {}
            return payload.get("data", {})
        except requests.RequestException as e:
            logger.warning("Détails Steam indisponibles (appid %s): %s", appid, e)
            return {}

    async def _tool_search_game(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        query = (tc.arguments.get("query") or "").strip()
        if not query:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))

        search = await asyncio.to_thread(self._search, query)
        if "error" in search:
            return ToolResponseRecord(tc.id, search, datetime.now(timezone.utc))

        first = search["first"]
        appid = first.get("id")

        details = {}
        if appid:
            details = await asyncio.to_thread(self._get_details, appid)

        result      = {**first, **details}
        llm_summary = _game_llm_summary(result)

        return ToolResponseRecord(tc.id, {
            "_tool":        "search_game",
            "_llm_summary": llm_summary,
            "result":       result,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="search_game",
                description=(
                    "Recherche un jeu sur le Steam Store et affiche sa fiche "
                    "(prix, soldes, avis, description, genres, développeur). "
                    "Utilise le nom exact ou le plus précis possible. "
                    "Si le nom est flou ou incertain, utilise d'abord search_web pour identifier "
                    "le bon titre Steam, puis appelle search_game."
                ),
                properties={
                    "query": {
                        "type":        "string",
                        "description": "Nom du jeu à rechercher sur Steam",
                    },
                },
                function=self._tool_search_game,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Steam(bot))
