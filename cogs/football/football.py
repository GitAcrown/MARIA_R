"""Cog Football — Scores et matchs de foot.

Deux sources pour préserver le quota :
- API-Football (api-sports.io) : uniquement le LIVE (score minute par minute,
  buteurs, statistiques). Tier gratuit limité à 100 requêtes/jour.
- TheSportsDB (clé publique gratuite, généreuse) : tout le NON-live, c.-à-d.
  dernier résultat et prochain match. Pas de live ni de stats.

Mode snapshot : score figé au moment de la requête (rappeler l'outil pour rafraîchir).
"""

import asyncio
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Football")

API_BASE = "https://v3.football.api-sports.io"

# Source gratuite pour le non-live (clé publique "123"), sans live ni stats
TSDB_BASE = "https://www.thesportsdb.com/api/v1/json"

# Statuts API-Football → (emoji, libellé, est_en_direct, est_termine)
_LIVE_STATUSES     = {"1H", "2H", "ET", "BT", "P", "LIVE", "INT"}
_FINISHED_STATUSES = {"FT", "AET", "PEN"}
_HALFTIME_STATUS   = "HT"

# Statistiques à afficher (clé API → libellé court)
_STAT_LABELS = {
    "Ball Possession": "Possession",
    "Total Shots":     "Tirs",
    "Shots on Goal":   "Tirs cadrés",
    "Corner Kicks":    "Corners",
    "Yellow Cards":    "Cartons jaunes",
    "Red Cards":       "Cartons rouges",
}


# ---------------------------------------------------------------------------
# Helpers de présentation
# ---------------------------------------------------------------------------

def _status_label(fixture: dict) -> str:
    """Libellé lisible du statut d'un match."""
    status  = fixture.get("status", {})
    short   = status.get("short", "")
    elapsed = status.get("elapsed")

    if short in _LIVE_STATUSES:
        return f"En direct · {elapsed}'" if elapsed else "En direct"
    if short == _HALFTIME_STATUS:
        return "Mi-temps"
    if short in _FINISHED_STATUSES:
        suffix = {"AET": " (a.p.)", "PEN": " (t.a.b.)"}.get(short, "")
        return f"Terminé{suffix}"
    if short in ("NS", "TBD"):
        return "À venir"
    return {
        "PST": "Reporté", "CANC": "Annulé", "ABD": "Abandonné",
        "SUSP": "Suspendu", "AWD": "Forfait", "WO": "Forfait",
    }.get(short, short or "?")


def _is_live(fixture: dict) -> bool:
    short = fixture.get("status", {}).get("short", "")
    return short in _LIVE_STATUSES or short == _HALFTIME_STATUS


def _is_started(fixture: dict) -> bool:
    short = fixture.get("status", {}).get("short", "")
    return short in _LIVE_STATUSES or short == _HALFTIME_STATUS or short in _FINISHED_STATUSES


def _parse_kickoff(date_str: Optional[str]) -> Optional[datetime]:
    """Parse une date ISO (API-Football ou TheSportsDB), UTC si pas de fuseau."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _kickoff_str(fixture: dict) -> str:
    """Renvoie un timestamp Discord pour l'heure de coup d'envoi."""
    dt = _parse_kickoff(fixture.get("date"))
    return f"<t:{int(dt.timestamp())}:F>" if dt else ""


def _scorers_by_team(events: list) -> dict[str, list[str]]:
    """Regroupe les buteurs par nom d'équipe : {équipe: ['Mbappé 23'', ...]}."""
    out: dict[str, list[str]] = {}
    for ev in events:
        if ev.get("type") != "Goal":
            continue
        detail = ev.get("detail", "")
        if detail == "Missed Penalty":
            continue
        team   = (ev.get("team") or {}).get("name", "")
        player = (ev.get("player") or {}).get("name") or "?"
        time   = ev.get("time") or {}
        minute = time.get("elapsed")
        extra  = time.get("extra")
        min_str = f"{minute}'" if minute else ""
        if extra:
            min_str += f"+{extra}"
        tag = " (csc)" if detail == "Own Goal" else (" (pen)" if detail == "Penalty" else "")
        out.setdefault(team, []).append(f"{player} {min_str}{tag}".strip())
    return out


def _stat_value(stats_block: list, team_name: str, stat_key: str) -> Optional[str]:
    for entry in stats_block:
        if (entry.get("team") or {}).get("name") == team_name:
            for s in entry.get("statistics", []):
                if s.get("type") == stat_key:
                    val = s.get("value")
                    return str(val) if val is not None else None
    return None


def _to_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _norm_name(name: str) -> str:
    """Normalise un nom d'équipe pour comparaison : minuscules, sans accents ni séparateurs."""
    if not name:
        return ""
    txt = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(c for c in txt.lower() if c.isalnum())


def _name_matches(target: str, candidate: str) -> bool:
    t, c = _norm_name(target), _norm_name(candidate)
    if not t or not c:
        return False
    return t == c or t in c or c in t


_TSDB_FINISHED = {"Match Finished", "FT", "AET", "After Extra Time", "Pen", "PEN"}


def _normalize_tsdb(ev: dict, team: dict) -> dict:
    """Convertit un événement TheSportsDB vers la structure interne (sans live/stats)."""
    status_raw = (ev.get("strStatus") or "").strip()
    gh = _to_int(ev.get("intHomeScore"))
    ga = _to_int(ev.get("intAwayScore"))
    finished = status_raw in _TSDB_FINISHED or (gh is not None and ga is not None)
    short = "FT" if finished else "NS"

    date_iso = ev.get("strTimestamp") or ""
    if not date_iso and ev.get("dateEvent"):
        date_iso = f"{ev['dateEvent']}T{ev.get('strTime') or '00:00:00'}"

    rnd = ev.get("intRound")
    league_round = f"J{rnd}" if rnd and str(rnd).isdigit() and int(rnd) > 0 else None
    fx = {"status": {"short": short, "elapsed": None}, "date": date_iso}

    return {
        "fixture": fx,
        "teams":   {"home": {"name": ev.get("strHomeTeam", "?")},
                    "away": {"name": ev.get("strAwayTeam", "?")}},
        "goals":   {"home": gh, "away": ga},
        "league":  {"name": ev.get("strLeague", ""), "round": league_round,
                    "logo": ev.get("strLeagueBadge") or team.get("strBadge")},
        "score":   {},
        "_events":     [],
        "_statistics": [],
        "_source":     "thesportsdb",
        "_kickoff_human": _kickoff_str_human(fx),
    }


# ---------------------------------------------------------------------------
# Résumé LLM
# ---------------------------------------------------------------------------

def _match_llm_summary(m: dict) -> str:
    fixture = m["fixture"]
    teams   = m["teams"]
    goals   = m["goals"]
    league  = m.get("league", {})

    home = teams.get("home", {}).get("name", "?")
    away = teams.get("away", {}).get("name", "?")
    gh   = goals.get("home")
    ga   = goals.get("away")

    parts = [f"Match : {home} vs {away}"]
    if league.get("name"):
        rnd = f" ({league['round']})" if league.get("round") else ""
        parts.append(f"{league['name']}{rnd}")

    if _is_started(fixture):
        parts.append(f"Score : {home} {gh}-{ga} {away}")
    parts.append(f"Statut : {_status_label(fixture)}")

    scorers = _scorers_by_team(m.get("_events", []))
    for team, names in scorers.items():
        parts.append(f"Buts {team} : {', '.join(names)}")

    stats = m.get("_statistics", [])
    if stats:
        poss_h = _stat_value(stats, home, "Ball Possession")
        poss_a = _stat_value(stats, away, "Ball Possession")
        if poss_h and poss_a:
            parts.append(f"Possession : {home} {poss_h} / {away} {poss_a}")
        sh_h = _stat_value(stats, home, "Total Shots")
        sh_a = _stat_value(stats, away, "Total Shots")
        if sh_h and sh_a:
            parts.append(f"Tirs : {home} {sh_h} / {away} {sh_a}")

    if not _is_started(fixture):
        ko = m.get("_kickoff_human", "")
        if ko:
            parts.append(f"Coup d'envoi : {ko}")

    parts.append("Widget affiché.")
    return " | ".join(parts)


def _live_list_llm_summary(matches: list) -> str:
    if not matches:
        return "Aucun match en direct actuellement. Widget affiché."
    parts = [f"{len(matches)} match(s) en direct :"]
    for m in matches[:8]:
        teams = m["teams"]
        goals = m["goals"]
        home  = teams.get("home", {}).get("name", "?")
        away  = teams.get("away", {}).get("name", "?")
        elapsed = m["fixture"].get("status", {}).get("elapsed")
        min_str = f" {elapsed}'" if elapsed else ""
        parts.append(f"{home} {goals.get('home')}-{goals.get('away')} {away}{min_str}")
    parts.append("Widget affiché.")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Builder LayoutView
# ---------------------------------------------------------------------------

def build_football_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView pour un match ou une liste de matchs en direct."""
    if "error" in data:
        return None

    mode = data.get("mode")
    if mode == "match" and data.get("result"):
        container = _match_container(data["result"])
    elif mode == "live_list":
        container = _live_list_container(data.get("results", []))
    else:
        return None

    if container is None:
        return None

    view = discord.ui.LayoutView(timeout=None)
    if commentary:
        view.add_item(discord.ui.TextDisplay(commentary))
        view.add_item(discord.ui.Separator())
    view.add_item(container)
    return view


def _match_container(m: dict) -> Optional[discord.ui.Container]:
    fixture = m["fixture"]
    teams   = m["teams"]
    goals   = m["goals"]
    league  = m.get("league", {})

    home = teams.get("home", {}).get("name", "?")
    away = teams.get("away", {}).get("name", "?")
    gh   = goals.get("home")
    ga   = goals.get("away")
    started = _is_started(fixture)

    # En-tête : ligue + statut
    league_name = league.get("name", "Football")
    rnd = f"  ·  {league['round']}" if league.get("round") else ""
    header = discord.ui.TextDisplay(f"## ⚽ {league_name}\n-# {_status_label(fixture)}{rnd}")

    # Ligne de score
    if started:
        score_line = f"### {home}  **{gh} - {ga}**  {away}"
    else:
        ko = _kickoff_str(fixture)
        score_line = f"### {home}  vs  {away}\n-# {ko}" if ko else f"### {home}  vs  {away}"

    # Thumbnail logo de ligue
    logo = league.get("logo")
    score_block = discord.ui.TextDisplay(score_line)
    try:
        if logo:
            thumb        = discord.ui.Thumbnail(discord.ui.UnfurledMediaItem(url=logo))
            score_section = discord.ui.Section(score_block, accessory=thumb)
        else:
            score_section = score_block
    except Exception:
        score_section = score_block

    children: list = [header, discord.ui.Separator(), score_section]

    # Buteurs
    scorers = _scorers_by_team(m.get("_events", []))
    if scorers:
        lines = []
        for team_name in (home, away):
            if team_name in scorers:
                lines.append(f"**{team_name}**  ·  {', '.join(scorers[team_name])}")
        if lines:
            children += [discord.ui.Separator(), discord.ui.TextDisplay("⚽ " + "\n⚽ ".join(lines))]

    # Statistiques (uniquement en direct)
    stats = m.get("_statistics", [])
    if stats and _is_live(fixture):
        stat_lines = []
        for key, label in _STAT_LABELS.items():
            vh = _stat_value(stats, home, key)
            va = _stat_value(stats, away, key)
            if vh is None and va is None:
                continue
            stat_lines.append(f"-# {label} · {vh or '0'} — {va or '0'}")
        if stat_lines:
            children += [discord.ui.Separator(), discord.ui.TextDisplay("\n".join(stat_lines))]

    source = "TheSportsDB" if m.get("_source") == "thesportsdb" else "API-Football"
    children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# Source : {source}")]
    return discord.ui.Container(*children)


def _live_list_container(matches: list) -> Optional[discord.ui.Container]:
    header = discord.ui.TextDisplay("## ⚽ Matchs en direct")
    children: list = [header, discord.ui.Separator()]

    if not matches:
        children.append(discord.ui.TextDisplay("-# Aucun match en direct pour le moment."))
        return discord.ui.Container(*children)

    lines = []
    for m in matches[:10]:
        teams = m["teams"]
        goals = m["goals"]
        home  = teams.get("home", {}).get("name", "?")
        away  = teams.get("away", {}).get("name", "?")
        elapsed = m["fixture"].get("status", {}).get("elapsed")
        min_str = f"`{elapsed}'`" if elapsed else "`live`"
        lines.append(f"{min_str}  {home} **{goals.get('home')}-{goals.get('away')}** {away}")
    children.append(discord.ui.TextDisplay("\n".join(lines)))
    children += [discord.ui.Separator(), discord.ui.TextDisplay("-# Source : API-Football")]
    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Football(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        cfg = getattr(bot, "config", {})
        self._api_key: str = cfg.get("API_FOOTBALL_KEY", "") or ""
        self._tsdb_key: str = cfg.get("THESPORTSDB_KEY", "") or "123"

    # -- Requêtes API (synchrones, exécutées dans un thread) ----------------

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        try:
            r = requests.get(
                f"{API_BASE}/{endpoint}",
                params=params,
                headers={"x-apisports-key": self._api_key},
                timeout=8,
            )
            if r.status_code == 401 or r.status_code == 403:
                return {"_auth_error": True}
            if not r.ok:
                logger.warning(f"API-Football {endpoint} → {r.status_code}")
                return None
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"API-Football {endpoint} : {e}")
            return None

    # -- TheSportsDB (gratuit) : tout ce qui n'est pas live -----------------

    def _tsdb_get(self, endpoint: str, params: dict) -> Optional[dict]:
        try:
            r = requests.get(f"{TSDB_BASE}/{self._tsdb_key}/{endpoint}", params=params, timeout=8)
            if not r.ok:
                logger.warning(f"TheSportsDB {endpoint} → {r.status_code}")
                return None
            return r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"TheSportsDB {endpoint} : {e}")
            return None

    def _tsdb_resolve(self, name: str) -> Optional[dict]:
        payload = self._tsdb_get("searchteams.php", {"t": name})
        teams = (payload or {}).get("teams") or []
        # Filtre football (Soccer) si plusieurs sports portent le même nom
        for t in teams:
            if (t.get("strSport") or "").lower() == "soccer":
                return t
        return teams[0] if teams else None

    def _tsdb_event(self, team_id: str, kind: str) -> Optional[dict]:
        endpoint = "eventslast.php" if kind == "last" else "eventsnext.php"
        payload = self._tsdb_get(endpoint, {"id": team_id})
        if not payload:
            return None
        items = payload.get("results") if kind == "last" else payload.get("events")
        return items[0] if items else None

    def _tsdb_card_from(self, team: Optional[dict], when: str) -> Optional[dict]:
        """Fiche légère (dernier résultat / prochain match) via l'API gratuite."""
        if not team:
            return None
        team_id = team.get("idTeam")
        if not team_id:
            return None

        if when == "next":
            ev = self._tsdb_event(team_id, "next")
        elif when == "last":
            ev = self._tsdb_event(team_id, "last")
        else:  # auto : dernier résultat en priorité, sinon prochain
            ev = self._tsdb_event(team_id, "last") or self._tsdb_event(team_id, "next")

        if not ev:
            return None
        return {"mode": "match", "result": _normalize_tsdb(ev, team)}

    # -- API-Football (payant, réservé au live) ------------------------------

    def _af_live_match(self, *names: str) -> Optional[dict]:
        """Cherche un match en direct via API-Football pour une des graphies données.

        Un seul appel (live=all) : la réponse contient déjà noms et ids, on filtre
        côté code par nom (le nom canonique TheSportsDB gère les alias type "PSG").
        """
        if not self._api_key:
            return None
        names = tuple(n for n in names if n)
        if not names:
            return None

        payload = self._get("fixtures", {"live": "all"})
        if not payload or payload.get("_auth_error") or not payload.get("response"):
            return None

        for fx in payload["response"]:
            teams = fx.get("teams", {})
            home  = (teams.get("home") or {}).get("name", "")
            away  = (teams.get("away") or {}).get("name", "")
            if any(_name_matches(n, home) or _name_matches(n, away) for n in names):
                return self._enrich_fixture(fx)

        return None

    def _enrich_fixture(self, fixture: dict) -> dict:
        """Ajoute events (buteurs) et statistiques pour un match en direct."""
        fx = fixture.get("fixture", {})
        fixture_id = fx.get("id")
        result = {
            "fixture": fx,
            "teams":   fixture.get("teams", {}),
            "goals":   fixture.get("goals", {}),
            "league":  fixture.get("league", {}),
            "score":   fixture.get("score", {}),
            "_events":     [],
            "_statistics": [],
            "_source":     "apifootball",
        }
        result["_kickoff_human"] = _kickoff_str_human(fx)

        if fixture_id and _is_started(fx):
            events = self._get("fixtures/events", {"fixture": fixture_id})
            if events and events.get("response"):
                result["_events"] = events["response"]
            if _is_live(fx):
                stats = self._get("fixtures/statistics", {"fixture": fixture_id})
                if stats and stats.get("response"):
                    result["_statistics"] = stats["response"]

        return {"mode": "match", "result": result}

    # -- Orchestration -------------------------------------------------------

    def _fetch_team_match(self, team_name: str, when: str) -> dict:
        """Live → API-Football ; reste → TheSportsDB (gratuit)."""
        # Résolution sur l'API gratuite (gère bien les alias type "PSG")
        tsdb_team = self._tsdb_resolve(team_name)
        canonical = (tsdb_team or {}).get("strTeam") or ""

        # Cas explicitement non-live : API gratuite uniquement
        if when in ("next", "last"):
            card = self._tsdb_card_from(tsdb_team, when)
            return card or {"error": f"Aucun match trouvé pour {team_name}"}

        # when == "live" ou "auto" : on tente le live (API-Football) d'abord,
        # en cherchant avec le nom saisi ET le nom canonique.
        live = self._af_live_match(team_name, canonical)
        if live:
            return live

        # Pas de live → fiche gratuite (dernier résultat / prochain)
        card = self._tsdb_card_from(tsdb_team, "auto")
        if card:
            return card

        if when == "live":
            return {"error": f"{team_name} ne joue pas en ce moment."}
        return {"error": f"Aucun match trouvé pour {team_name}"}

    def _fetch_live_list(self) -> dict:
        if not self._api_key:
            return {"error": "Clé API-Football manquante (API_FOOTBALL_KEY dans .env)"}
        payload = self._get("fixtures", {"live": "all"})
        if payload and payload.get("_auth_error"):
            return {"error": "Clé API-Football invalide ou quota dépassé"}
        if not payload:
            return {"error": "Erreur API-Football"}
        matches = payload.get("response", [])
        return {"mode": "live_list", "results": matches}

    # -- Outil ---------------------------------------------------------------

    async def _tool_get_football(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        team = (tc.arguments.get("team") or "").strip()
        when = (tc.arguments.get("when") or "auto").strip().lower()
        if when not in ("auto", "live", "next", "last"):
            when = "auto"
        loop = asyncio.get_event_loop()

        if team:
            data = await loop.run_in_executor(None, self._fetch_team_match, team, when)
            summary = _match_llm_summary(data["result"]) if data.get("mode") == "match" else None
        else:
            data = await loop.run_in_executor(None, self._fetch_live_list)
            summary = _live_list_llm_summary(data.get("results", [])) if data.get("mode") == "live_list" else None

        if "error" in data:
            return ToolResponseRecord(tc.id, data, datetime.now(timezone.utc))

        return ToolResponseRecord(tc.id, {
            "_tool":        "get_football",
            "_llm_summary": summary or "Résultat football affiché.",
            **data,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_football",
                description=(
                    "Affiche le score d'un match de foot avec buteurs (et statistiques si en direct). "
                    "Renseigne 'team' avec le nom d'un club ou d'une sélection (ex: 'PSG', 'Real Madrid', "
                    "'France'). Laisse 'team' vide pour lister tous les matchs en direct du moment. "
                    "Snapshot : rappelle l'outil pour rafraîchir un score en direct."
                ),
                properties={
                    "team": {
                        "type":        "string",
                        "description": "Nom du club ou de la sélection. Vide = liste des matchs en direct.",
                    },
                    "when": {
                        "type":        "string",
                        "enum":        ["auto", "live", "next", "last"],
                        "description": (
                            "Quel match : 'auto' (en direct sinon dernier résultat), 'live' (uniquement "
                            "si l'équipe joue maintenant), 'next' (prochain match à venir), "
                            "'last' (dernier résultat). Utilise 'next'/'last' quand la question est "
                            "explicitement sur le futur/passé."
                        ),
                    },
                },
                function=self._tool_get_football,
            ),
        ]


def _kickoff_str_human(fixture: dict) -> str:
    """Version texte brut (pour le résumé LLM) du coup d'envoi."""
    dt = _parse_kickoff(fixture.get("date"))
    return dt.strftime("%d/%m %H:%M UTC") if dt else ""


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Football(bot))
