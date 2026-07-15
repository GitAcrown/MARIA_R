"""Cog Football — Scores et matchs de foot.

Deux sources :
- API-Football (api-sports.io) : source PRIORITAIRE (plus précise) pour un match
  donné — live (score minute par minute, buteurs, stats), dernier résultat,
  prochain match. Tier gratuit limité à 100 requêtes/jour.
- TheSportsDB (clé publique gratuite, généreuse) : SECOURS quand le quota
  API-Football est épuisé/indisponible, et source des listes "infos générales"
  (ex: les 5 derniers matchs d'une équipe) pour économiser le quota.

Mode snapshot : score figé au moment de la requête (rappeler l'outil pour rafraîchir).
"""

import asyncio
import logging
import random
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.discord_ui import layout_with_commentary
from common.emojis import DIRECT, FOOTBALL, FOOTBALL_PLAYER
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
        live_word = "EN DIRET" if random.random() < 0.2 else "EN DIRECT"
        label = f"{DIRECT} {live_word}"
        return f"{label} · {elapsed}'" if elapsed else label
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


def _teams_match_query(home: str, away: str, *names: str) -> bool:
    """True si chaque nom cité correspond à l'une des deux équipes du match."""
    names = [n for n in names if n]
    if not names:
        return False
    if len(names) == 1:
        return _name_matches(names[0], home) or _name_matches(names[0], away)
    return all(_name_matches(n, home) or _name_matches(n, away) for n in names)


def _af_down(payload: Optional[dict]) -> bool:
    """True si l'appel API-Football a échoué ou est indisponible (→ repli TheSportsDB)."""
    return payload is None or bool(payload.get("_unavailable"))


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
        stat_bits = []
        for key, label in _STAT_LABELS.items():
            vh = _stat_value(stats, home, key)
            va = _stat_value(stats, away, key)
            if vh is None and va is None:
                continue
            stat_bits.append(f"{label} : {home} {vh or '0'} / {away} {va or '0'}")
        if stat_bits:
            parts.append("Stats : " + " · ".join(stat_bits))

    if not _is_started(fixture):
        ko = m.get("_kickoff_human", "")
        if ko:
            parts.append(f"Coup d'envoi : {ko}")

    parts.append("Widget match détaillé affiché (score, buteurs, stats si en direct).")
    return " | ".join(parts)


def _match_list_llm_summary(matches: list, title: str) -> str:
    if not matches:
        return f"Aucun match récent pour {title}. Widget affiché."
    parts = [f"Derniers matchs de {title} :"]
    for m in matches[:5]:
        teams = m["teams"]
        goals = m["goals"]
        home  = teams.get("home", {}).get("name", "?")
        away  = teams.get("away", {}).get("name", "?")
        gh, ga = goals.get("home"), goals.get("away")
        score  = f"{gh}-{ga}" if gh is not None and ga is not None else "vs"
        parts.append(f"{home} {score} {away}")
    parts.append("Widget liste affiché.")
    return " | ".join(parts)


def _live_list_llm_summary(matches: list) -> str:
    if not matches:
        return "Aucun match en direct actuellement. Widget liste affiché."
    parts = [f"LISTE SEULEMENT — {len(matches)} match(s) en direct (scores sans stats détaillées) :"]
    for m in matches[:8]:
        teams = m["teams"]
        goals = m["goals"]
        home  = teams.get("home", {}).get("name", "?")
        away  = teams.get("away", {}).get("name", "?")
        elapsed = m["fixture"].get("status", {}).get("elapsed")
        min_str = f" {elapsed}'" if elapsed else ""
        parts.append(f"{home} {goals.get('home')}-{goals.get('away')} {away}{min_str}")
    parts.append(
        "Pour score/buteurs/stats d'un match précis, rappeler get_football avec team=<équipe> "
        "(et opponent=<autre équipe> si besoin). Widget liste affiché."
    )
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
    elif mode == "match_list":
        container = _match_list_container(data.get("results", []), data.get("title", "Derniers matchs"))
    else:
        return None

    if container is None:
        return None

    return layout_with_commentary(container, commentary)


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
    header = discord.ui.TextDisplay(f"## {FOOTBALL} {league_name}\n-# {_status_label(fixture)}{rnd}")

    # Ligne de score
    if started:
        score_line = f"### {home}  **{gh} - {ga}**  {away}"
    else:
        ko = _kickoff_str(fixture)
        score_line = f"### {home}  vs  {away}\n-# {ko}" if ko else f"### {home}  vs  {away}"

    score_block = discord.ui.TextDisplay(score_line)

    children: list = [header, discord.ui.Separator(), score_block]

    # Buteurs
    scorers = _scorers_by_team(m.get("_events", []))
    if scorers:
        lines = []
        for team_name in (home, away):
            if team_name in scorers:
                lines.append(f"**{team_name}**  ·  {', '.join(scorers[team_name])}")
        if lines:
            children += [discord.ui.Separator(), discord.ui.TextDisplay(f"{FOOTBALL_PLAYER} " + f"\n{FOOTBALL_PLAYER} ".join(lines))]

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


def _match_list_container(matches: list, title: str) -> Optional[discord.ui.Container]:
    header = discord.ui.TextDisplay(f"## {FOOTBALL} Derniers matchs · {title}")
    children: list = [header, discord.ui.Separator()]

    if not matches:
        children.append(discord.ui.TextDisplay("-# Aucun match récent."))
        return discord.ui.Container(*children)

    lines = []
    for m in matches[:5]:
        teams = m["teams"]
        goals = m["goals"]
        home  = teams.get("home", {}).get("name", "?")
        away  = teams.get("away", {}).get("name", "?")
        gh, ga = goals.get("home"), goals.get("away")
        score  = f"{gh}-{ga}" if gh is not None and ga is not None else "vs"
        date   = m.get("_kickoff_human", "")
        date_str = f"  `{date[:5]}`" if date else ""
        lines.append(f"{home} **{score}** {away}{date_str}")
    children.append(discord.ui.TextDisplay("\n".join(lines)))

    source = "API-Football" if any(m.get("_source") == "apifootball" for m in matches) else "TheSportsDB"
    children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# Source : {source}")]
    return discord.ui.Container(*children)


def _live_list_container(matches: list) -> Optional[discord.ui.Container]:
    header = discord.ui.TextDisplay(f"## {FOOTBALL} Matchs en direct")
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
        # "123" est la clé de test publique officielle de TheSportsDB (fallback assumé
        # quand THESPORTSDB_KEY n'est pas défini dans .env) — voir thesportsdb.com/api.php.
        self._tsdb_key: str = cfg.get("THESPORTSDB_KEY", "") or "123"
        self._id_cache: dict[str, Optional[int]] = {}

    # -- Requêtes API (synchrones, exécutées dans un thread) ----------------

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Appel API-Football. Renvoie {"_unavailable": True} si quota/clé/erreur API."""
        try:
            r = requests.get(
                f"{API_BASE}/{endpoint}",
                params=params,
                headers={"x-apisports-key": self._api_key},
                timeout=8,
            )
            if r.status_code in (401, 403, 429):
                logger.warning(f"API-Football {endpoint} indisponible (HTTP {r.status_code})")
                return {"_unavailable": True}
            if not r.ok:
                logger.warning(f"API-Football {endpoint} → {r.status_code}")
                return None
            data = r.json()
            # Quota journalier / limite de plan : HTTP 200 mais champ errors non vide
            if data.get("errors"):
                logger.warning(f"API-Football {endpoint} errors : {data['errors']}")
                return {"_unavailable": True}
            return data
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"API-Football {endpoint} : {e}")
            return None

    # -- TheSportsDB (gratuit) : secours quota + listes générales -----------

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

    # -- TheSportsDB : listes "infos générales" (derniers matchs) ------------

    def _tsdb_recent_list(self, team: Optional[dict], team_name: str) -> dict:
        """Liste des derniers matchs d'une équipe via TheSportsDB (secours quota)."""
        if not team or not team.get("idTeam"):
            return {"error": f"Équipe introuvable : {team_name!r}"}
        payload = self._tsdb_get("eventslast.php", {"id": team["idTeam"]})
        results = (payload or {}).get("results") or []
        if not results:
            return {"error": f"Aucun match récent pour {team_name}"}
        matches = [_normalize_tsdb(ev, team) for ev in results[:5]]
        return {"mode": "match_list", "results": matches,
                "title": team.get("strTeam") or team_name}

    # -- API-Football : liste des derniers matchs (plus précise) ------------

    def _af_recent_list(self, team_id: int, team_name: str, n: int = 5) -> Optional[dict]:
        """Derniers N matchs terminés via API-Football. None si API indisponible."""
        payload = self._get("fixtures", {"team": team_id, "last": n})
        if _af_down(payload) or not payload.get("response"):
            return None
        matches = []
        for fx in payload["response"]:
            r = fx.get("fixture", {})
            matches.append({
                "fixture": r,
                "teams":   fx.get("teams", {}),
                "goals":   fx.get("goals", {}),
                "league":  fx.get("league", {}),
                "score":   fx.get("score", {}),
                "_events":     [],
                "_statistics": [],
                "_source":     "apifootball",
                "_kickoff_human": _kickoff_str_human(r),
            })
        if not matches:
            return None
        return {"mode": "match_list", "results": matches, "title": team_name}

    # -- API-Football (source prioritaire) -----------------------------------

    def _af_resolve_id(self, *names: str) -> Optional[int]:
        """Résout l'id d'équipe API-Football (avec cache). None si introuvable/indispo."""
        for name in names:
            if not name:
                continue
            key = _norm_name(name)
            if key in self._id_cache:
                if self._id_cache[key] is not None:
                    return self._id_cache[key]
                continue
            payload = self._get("teams", {"search": name})
            if _af_down(payload):
                return None
            resp = payload.get("response") or []
            tid = resp[0].get("team", {}).get("id") if resp else None
            self._id_cache[key] = tid
            if tid is not None:
                return tid
        return None

    def _af_find_live(self, *names: str) -> Optional[dict]:
        """Match en direct via API-Football (live=all filtré par nom). None si pas live/indispo."""
        names = tuple(n for n in names if n)
        if not names:
            return None
        payload = self._get("fixtures", {"live": "all"})
        if _af_down(payload) or not payload.get("response"):
            return None
        for fx in payload["response"]:
            teams = fx.get("teams", {})
            home  = (teams.get("home") or {}).get("name", "")
            away  = (teams.get("away") or {}).get("name", "")
            if any(_name_matches(n, home) or _name_matches(n, away) for n in names):
                return self._enrich_fixture(fx)
        return None

    def _af_find_live_between(self, team_a: str, team_b: str) -> Optional[dict]:
        """Match en direct où les deux équipes citées s'affrontent."""
        if not self._api_key or not team_a or not team_b:
            return None
        tsdb_a = self._tsdb_resolve(team_a)
        tsdb_b = self._tsdb_resolve(team_b)
        names_a = [team_a]
        names_b = [team_b]
        if tsdb_a and tsdb_a.get("strTeam"):
            names_a.append(tsdb_a["strTeam"])
        if tsdb_b and tsdb_b.get("strTeam"):
            names_b.append(tsdb_b["strTeam"])

        payload = self._get("fixtures", {"live": "all"})
        if _af_down(payload) or not payload.get("response"):
            return None
        for fx in payload["response"]:
            teams = fx.get("teams", {})
            home  = (teams.get("home") or {}).get("name", "")
            away  = (teams.get("away") or {}).get("name", "")
            match_a = any(_name_matches(n, home) or _name_matches(n, away) for n in names_a)
            match_b = any(_name_matches(n, home) or _name_matches(n, away) for n in names_b)
            if match_a and match_b:
                return self._enrich_fixture(fx)
        return None

    def _af_team_match(self, name: str, canonical: str, when: str) -> Optional[dict]:
        """Match d'une équipe via API-Football. None = pas trouvé OU API indisponible."""
        if not self._api_key:
            return None

        if when in ("auto", "live"):
            live = self._af_find_live(name, canonical)
            if live:
                return live
            if when == "live":
                return None  # pas en direct → le caller bascule sur TheSportsDB

        team_id = self._af_resolve_id(name, canonical)
        if team_id is None:
            return None
        kind = "next" if when == "next" else "last"
        payload = self._get("fixtures", {"team": team_id, kind: 1})
        if _af_down(payload) or not payload.get("response"):
            return None
        return self._enrich_fixture(payload["response"][0])

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
        """API-Football en priorité pour tous les modes ; TheSportsDB en secours quota.

        Le nom canonique TheSportsDB sert à résoudre les alias type "PSG" côté API-Football.
        """
        tsdb_team = self._tsdb_resolve(team_name)
        canonical = (tsdb_team or {}).get("strTeam") or ""

        if when == "recent":
            # Essaie API-Football d'abord (résultats plus précis avec buteurs/scores corrects)
            if self._api_key:
                team_id = self._af_resolve_id(team_name, canonical)
                if team_id is not None:
                    af_list = self._af_recent_list(team_id, canonical or team_name)
                    if af_list is not None:
                        return af_list
            # Secours : TheSportsDB (gratuit, sans quota)
            return self._tsdb_recent_list(tsdb_team, team_name)

        # Source prioritaire : API-Football
        af = self._af_team_match(team_name, canonical, when)
        if af is not None:
            return af

        # Secours gratuit (quota épuisé, clé absente, ou rien trouvé côté API-Football)
        card = self._tsdb_card_from(tsdb_team, when if when in ("next", "last") else "auto")
        if card:
            return card

        if when == "live":
            return {"error": f"{team_name} ne joue pas en ce moment."}
        return {"error": f"Aucun match trouvé pour {team_name}"}

    def _fetch_match_between(self, team_a: str, team_b: str, when: str) -> dict:
        """Match entre deux équipes.

        En live : filtre strictement sur les deux noms (+ canoniques TheSportsDB).
        Non-live : récupère simplement le dernier/prochain match de team_a puis team_b.
        On n'essaie pas de filtrer par nom d'adversaire hors live — les noms dans l'API
        sont en anglais et ne correspondent pas aux noms localisés (ex. Allemagne ≠ Germany).
        """
        if when in ("auto", "live"):
            live = self._af_find_live_between(team_a, team_b)
            if live:
                return live
            if when == "live":
                return {"error": f"Pas de match en direct entre {team_a} et {team_b}."}

        # Non-live : on cherche le match de team_a, puis team_b en fallback.
        result = self._fetch_team_match(team_a, when)
        if result.get("mode") == "match":
            return result
        result = self._fetch_team_match(team_b, when)
        if result.get("mode") == "match":
            return result

        return {"error": f"Aucun match trouvé pour {team_a} ou {team_b}"}

    def _fetch_live_list(self) -> dict:
        if not self._api_key:
            return {"error": "Clé API-Football manquante (API_FOOTBALL_KEY dans .env)"}
        payload = self._get("fixtures", {"live": "all"})
        if _af_down(payload):
            return {"error": "API-Football indisponible (clé invalide ou quota dépassé)"}
        return {"mode": "live_list", "results": payload.get("response", [])}

    # -- Outil ---------------------------------------------------------------

    async def _tool_get_football(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        team = (tc.arguments.get("team") or "").strip()
        opponent = (tc.arguments.get("opponent") or "").strip()
        when = (tc.arguments.get("when") or "auto").strip().lower()
        if when not in ("auto", "live", "next", "last", "recent"):
            when = "auto"

        if team and opponent:
            data = await asyncio.to_thread(self._fetch_match_between, team, opponent, when)
            summary = _match_llm_summary(data["result"]) if data.get("mode") == "match" else None
        elif team:
            data = await asyncio.to_thread(self._fetch_team_match, team, when)
            mode = data.get("mode")
            if mode == "match":
                summary = _match_llm_summary(data["result"])
            elif mode == "match_list":
                summary = _match_list_llm_summary(data.get("results", []), data.get("title", team))
            else:
                summary = None
        elif opponent:
            data = await asyncio.to_thread(self._fetch_team_match, opponent, when)
            mode = data.get("mode")
            if mode == "match":
                summary = _match_llm_summary(data["result"])
            elif mode == "match_list":
                summary = _match_list_llm_summary(data.get("results", []), data.get("title", opponent))
            else:
                summary = None
        else:
            data = await asyncio.to_thread(self._fetch_live_list)
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
                    "Score, buteurs et statistiques d'un match de foot (possession, tirs, cartons…). "
                    "Renseigne 'team' avec un club ou une sélection (ex: 'PSG', 'USA', 'France'). "
                    "Si deux équipes sont citées, renseigne aussi 'opponent'. "
                    "Laisse 'team' ET 'opponent' à null uniquement pour lister tous les matchs en direct "
                    "(sans stats détaillées). Snapshot : rappelle l'outil pour rafraîchir."
                ),
                properties={
                    "team": {
                        "type":        "string",
                        "description": (
                            "Nom d'une équipe. Obligatoire pour score/buteurs/stats d'un match précis. "
                            "null seulement pour la liste générale des matchs en direct."
                        ),
                    },
                    "opponent": {
                        "type":        "string",
                        "description": (
                            "Autre équipe quand les deux adversaires sont cités (ex: team='USA', "
                            "opponent='Australie'). null si une seule équipe suffit."
                        ),
                    },
                    "when": {
                        "type":        "string",
                        "enum":        ["auto", "live", "next", "last", "recent"],
                        "description": (
                            "Quel match : 'auto' (en direct sinon dernier résultat), 'live' (uniquement "
                            "si l'équipe joue maintenant), 'next' (prochain match), 'last' (dernier "
                            "résultat), 'recent' (liste des 5 derniers matchs de l'équipe). "
                            "null = 'auto'."
                        ),
                    },
                },
                optional_props=["team", "opponent", "when"],
                function=self._tool_get_football,
            ),
        ]


def _kickoff_str_human(fixture: dict) -> str:
    """Version texte brut (pour le résumé LLM) du coup d'envoi."""
    dt = _parse_kickoff(fixture.get("date"))
    return dt.strftime("%d/%m %H:%M UTC") if dt else ""


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Football(bot))
