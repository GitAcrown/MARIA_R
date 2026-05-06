"""Cog Sport — football-data.org avec rendu LayoutView (scores live, résultats, matchs à venir)."""

import asyncio
import logging
import zoneinfo
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Sport")

FD_BASE   = "https://api.football-data.org/v4"
PARIS_TZ  = zoneinfo.ZoneInfo("Europe/Paris")

# Compétitions disponibles sur le free tier football-data.org
COMPETITIONS: dict[str, str] = {
    "ligue 1":            "FL1",
    "ligue1":             "FL1",
    "l1":                 "FL1",
    "premier league":     "PL",
    "pl":                 "PL",
    "la liga":            "PD",
    "liga":               "PD",
    "bundesliga":         "BL1",
    "serie a":            "SA",
    "seriea":             "SA",
    "champions league":   "CL",
    "ldc":                "CL",
    "ucl":                "CL",
    "ligue des champions":"CL",
    "world cup":          "WC",
    "coupe du monde":     "WC",
    "euro":               "EC",
    "eredivisie":         "DED",
    "primeira liga":      "PPL",
    "championship":       "ELC",
}

# Emojis de statut
_STATUS_EMOJI = {
    "SCHEDULED":   "🕐",
    "TIMED":       "🕐",
    "IN_PLAY":     "🔴",
    "PAUSED":      "⏸",
    "FINISHED":    "✅",
    "SUSPENDED":   "⚠️",
    "POSTPONED":   "📅",
    "CANCELLED":   "❌",
    "AWARDED":     "🏆",
}

_COMP_NAMES = {
    "FL1": "Ligue 1", "PL": "Premier League", "PD": "La Liga",
    "BL1": "Bundesliga", "SA": "Serie A", "CL": "Champions League",
    "WC": "Coupe du Monde", "EC": "Euro", "DED": "Eredivisie",
    "PPL": "Primeira Liga", "ELC": "Championship",
}



def _status_label(match: dict) -> str:
    status = match.get("status", "")
    score  = match.get("score", {})
    ft     = score.get("fullTime", {})
    ht     = score.get("halfTime", {})
    minute = match.get("minute")

    if status in ("IN_PLAY", "PAUSED"):
        m = f" **{minute}'**" if minute else ""
        home = ft.get("home", "?")
        away = ft.get("away", "?")
        return f"🔴{m}  **{home} – {away}**"
    if status == "FINISHED":
        home = ft.get("home", "?")
        away = ft.get("away", "?")
        ht_h = ht.get("home")
        ht_a = ht.get("away")
        ht_str = f"\n-# mi-temps {ht_h}-{ht_a}" if ht_h is not None else ""
        return f"✅  **{home} – {away}**{ht_str}"
    if status in ("SCHEDULED", "TIMED"):
        utc_str = match.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
            local = dt.astimezone(PARIS_TZ)
            return f"🕐  **{local.strftime('%H:%M')}**"
        except Exception:
            return "🕐"
    return _STATUS_EMOJI.get(status, "❓")


def _match_line(match: dict) -> str:
    home = match.get("homeTeam", {}).get("shortName") or match.get("homeTeam", {}).get("name", "?")
    away = match.get("awayTeam", {}).get("shortName") or match.get("awayTeam", {}).get("name", "?")
    label = _status_label(match)
    return f"*{home}* vs *{away}*  ·  {label}"


# ---------------------------------------------------------------------------
# Builders de vue
# ---------------------------------------------------------------------------

def build_sport_view(data: dict) -> Optional[discord.ui.LayoutView]:
    if "error" in data or "matches" not in data:
        return None
    try:
        return _matches_view(data)
    except Exception as e:
        logger.error(f"Erreur build_sport_view: {e}", exc_info=True)
        return None


def _matches_view(data: dict) -> discord.ui.LayoutView:
    view      = discord.ui.LayoutView(timeout=None)
    matches   = data.get("matches", [])
    comp_code = data.get("competition", "")
    query_type = data.get("query_type", "today")
    comp_name = _COMP_NAMES.get(comp_code, comp_code)
    team_name = data.get("team_name", "")
    updated   = datetime.now(PARIS_TZ).strftime("%H:%M")

    # Titre
    _type_labels = {"live": "Matchs live", "today": "Matchs du jour", "results": "Résultats récents", "upcoming": "Prochains matchs"}
    type_label = _type_labels.get(query_type, "Matchs")
    subject = team_name if team_name else comp_name
    title = f"## ⚽ *{subject}* — {type_label}"

    children: list = [discord.ui.TextDisplay(title), discord.ui.Separator()]

    if not matches:
        children.append(discord.ui.TextDisplay("-# Aucun match trouvé pour cette recherche."))
    else:
        for i, match in enumerate(matches[:8]):
            main_line = _match_line(match)
            # Sous-ligne : date + compétition (chacun doit être en début de ligne pour -#)
            sub_parts: list[str] = []
            if query_type in ("results", "upcoming"):
                utc_str = match.get("utcDate", "")
                try:
                    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                    sub_parts.append(dt.astimezone(PARIS_TZ).strftime("%d/%m"))
                except Exception:
                    pass
            if team_name:
                match_comp = (match.get("competition") or {}).get("name", "")
                if match_comp:
                    sub_parts.append(match_comp)
            sub_line = f"-# {' · '.join(sub_parts)}" if sub_parts else ""
            block = f"{main_line}\n{sub_line}" if sub_line else main_line

            children.append(discord.ui.TextDisplay(block))
            if i < min(len(matches), 8) - 1:
                children.append(discord.ui.Separator())

    children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# Mis à jour à {updated}")]
    view.add_item(discord.ui.Container(*children))
    return view


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Sport(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._api_key: str = getattr(bot, "config", {}).get("FOOTBALL_DATA_KEY", "") or ""

    def _headers(self) -> dict:
        return {"X-Auth-Token": self._api_key}

    def _fetch(self, path: str, params: Optional[dict] = None) -> dict:
        if not self._api_key:
            return {"error": "Clé API football-data.org manquante (FOOTBALL_DATA_KEY dans .env)"}
        try:
            r = requests.get(f"{FD_BASE}{path}", headers=self._headers(), params=params, timeout=8)
            if r.status_code == 400:
                return {"error": "Paramètres invalides"}
            if r.status_code == 401:
                return {"error": "Clé API football-data.org invalide"}
            if r.status_code == 403:
                return {"error": "Compétition non disponible sur le free tier"}
            if r.status_code == 404:
                return {"error": "Ressource introuvable"}
            if r.status_code == 429:
                return {"error": "Limite de requêtes atteinte (10/min)"}
            if not r.ok:
                return {"error": f"Erreur API {r.status_code}"}
            return r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _resolve_competition(self, query: str) -> Optional[str]:
        """Retourne le code de compétition depuis un nom free-form."""
        q = query.strip().lower()
        return COMPETITIONS.get(q)

    def _search_team(self, name: str) -> Optional[dict]:
        """Cherche une équipe par nom, retourne le premier résultat."""
        data = self._fetch("/teams", {"name": name})
        teams = data.get("teams", [])
        return teams[0] if teams else None

    def _get_team_matches(self, team_id: int, query_type: str) -> dict:
        status_map = {
            "live":     "IN_PLAY,PAUSED",
            "results":  "FINISHED",
            "upcoming": "SCHEDULED,TIMED",
            "today":    "IN_PLAY,PAUSED,SCHEDULED,TIMED,FINISHED",
        }
        status = status_map.get(query_type, "IN_PLAY,PAUSED,SCHEDULED,TIMED,FINISHED")
        params: dict = {"status": status, "limit": 8}
        if query_type == "today":
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            params["dateFrom"] = today
            params["dateTo"]   = today
        data = self._fetch(f"/teams/{team_id}/matches", params)
        return data

    def _get_competition_matches(self, comp_code: str, query_type: str) -> dict:
        status_map = {
            "live":     "IN_PLAY,PAUSED",
            "results":  "FINISHED",
            "upcoming": "SCHEDULED,TIMED",
            "today":    "IN_PLAY,PAUSED,SCHEDULED,TIMED,FINISHED",
        }
        status = status_map.get(query_type, "IN_PLAY,PAUSED,SCHEDULED,TIMED,FINISHED")
        params: dict = {"status": status, "limit": 8}
        if query_type == "today":
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            params["dateFrom"] = today
            params["dateTo"]   = today
        data = self._fetch(f"/competitions/{comp_code}/matches", params)
        return data

    async def _tool_sport(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        query      = (tc.arguments.get("query") or "").strip()
        query_type = (tc.arguments.get("type") or "today").strip().lower()

        if not query:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))

        loop = asyncio.get_event_loop()

        # Résolution : compétition connue ou recherche d'équipe
        search_query = query
        comp_code = self._resolve_competition(search_query)

        if comp_code:
            raw = await loop.run_in_executor(None, self._get_competition_matches, comp_code, query_type)
            if "error" in raw:
                return ToolResponseRecord(tc.id, raw, datetime.now(timezone.utc))
            matches = raw.get("matches", [])
            llm_summary = (
                f"{len(matches)} match(s) {query_type} pour {_COMP_NAMES.get(comp_code, comp_code)}. "
                "LayoutView envoyé."
            )
            return ToolResponseRecord(tc.id, {
                "_tool":        "get_sport_scores",
                "_llm_summary": llm_summary,
                "competition":  comp_code,
                "query_type":   query_type,
                "matches":      matches,
            }, datetime.now(timezone.utc))
        else:
            # Recherche par équipe (avec alias résolu)
            team = await loop.run_in_executor(None, self._search_team, search_query)
            if not team:
                return ToolResponseRecord(tc.id, {"error": f"Équipe introuvable : {query!r}"}, datetime.now(timezone.utc))

            team_id   = team["id"]
            team_name = team.get("shortName") or team.get("name", query)
            raw = await loop.run_in_executor(None, self._get_team_matches, team_id, query_type)
            if "error" in raw:
                return ToolResponseRecord(tc.id, raw, datetime.now(timezone.utc))

            matches = raw.get("matches", [])
            llm_summary = (
                f"{len(matches)} match(s) {query_type} pour {team_name}. "
                "LayoutView envoyé."
            )
            return ToolResponseRecord(tc.id, {
                "_tool":        "get_sport_scores",
                "_llm_summary": llm_summary,
                "competition":  "",
                "team_name":    team_name,
                "query_type":   query_type,
                "matches":      matches,
            }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_sport_scores",
                description=(
                    "Récupère les scores et matchs de foot. "
                    "query = nom officiel complet d'une compétition (ex: 'Ligue 1', 'Champions League', 'Premier League') "
                    "ou d'une équipe — toujours le nom officiel complet, jamais un sigle ou surnom "
                    "(ex: 'Paris Saint-Germain' et non 'PSG', 'Olympique de Marseille' et non 'OM', "
                    "'FC Bayern München' et non 'Bayern', 'Borussia Dortmund' et non 'BVB'). "
                    "type : 'live' (matchs en cours), 'today' (matchs du jour), "
                    "'results' (derniers résultats), 'upcoming' (prochains matchs). "
                    "Réponse : un mot max après le LayoutView ('tiens', 'voilà')."
                ),
                properties={
                    "query": {
                        "type": "string",
                        "description": "Nom officiel complet de la compétition ou de l'équipe",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["live", "today", "results", "upcoming"],
                        "description": "Type de recherche",
                    },
                },
                function=self._tool_sport,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Sport(bot))
