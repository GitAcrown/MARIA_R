"""Cog Météo — OpenWeatherMap avec rendu LayoutView."""

import asyncio
import logging
import zoneinfo
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Meteo")

OWM_BASE = "https://api.openweathermap.org/data/2.5"
OWM_ICON  = "https://openweathermap.org/img/wn/{}@2x.png"
PARIS_TZ  = zoneinfo.ZoneInfo("Europe/Paris")

_ICON_EMOJI: dict[str, str] = {
    "01": "☀️", "02": "🌤️", "03": "⛅", "04": "☁️",
    "09": "🌦️", "10": "🌧️", "11": "⛈️", "13": "🌨️", "50": "🌫️",
}
_WEEKDAYS_FULL  = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
_WEEKDAYS_SHORT = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
_WIND_DIRS      = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]

# Périodes du jour : (label, heure_début_incluse, heure_fin_exclue)
_PERIODS = [
    ("🌅 Matin",      6,  12),
    ("☀️ Après-midi", 12, 18),
    ("🌆 Soir",       18, 22),
    ("🌙 Nuit",        0,  6),
]


def _emoji(icon: str) -> str:
    return _ICON_EMOJI.get(icon[:2], "🌡️")


def _wind_dir(deg: float) -> str:
    return _WIND_DIRS[round(deg / 45) % 8]


def _weekday_short(dt: datetime) -> str:
    return _WEEKDAYS_SHORT[dt.weekday()]


def _parse_target_date(target_date: str) -> Optional[date]:
    """Convertit une chaîne (demain, lundi, 2026-05-03…) en objet date."""
    t = target_date.strip().lower()
    today = datetime.now(PARIS_TZ).date()

    if t in ("today", "aujourd'hui", "auj"):
        return today
    if t in ("tomorrow", "demain"):
        return today + timedelta(days=1)
    if t in ("après-demain", "apres-demain"):
        return today + timedelta(days=2)

    # Nom de jour en français
    fr_days = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    if t in fr_days:
        target_wd = fr_days.index(t)
        current_wd = today.weekday()
        delta = (target_wd - current_wd) % 7 or 7  # prochain occurence (jamais aujourd'hui)
        return today + timedelta(days=delta)

    # Format ISO YYYY-MM-DD
    try:
        return date.fromisoformat(t)
    except ValueError:
        pass

    return None


# ---------------------------------------------------------------------------
# Builders de vue
# ---------------------------------------------------------------------------

def build_weather_view(data: dict) -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView à partir des données retournées par l'outil."""
    if "error" in data or "data" not in data:
        return None
    weather_type = data.get("type", "current")
    city = data.get("city", "?")
    raw  = data["data"]
    target_date_str = data.get("target_date")

    try:
        if weather_type == "forecast":
            if target_date_str:
                target = _parse_target_date(target_date_str)
                if target:
                    return _day_view(city, raw, target)
            return _forecast_view(city, raw)
        return _current_view(city, raw)
    except Exception as e:
        logger.error(f"Erreur build_weather_view: {e}", exc_info=True)
        return None


def _current_view(city: str, d: dict) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=None)

    weather   = d["weather"][0]
    icon_code = weather.get("icon", "01d")
    description = weather.get("description", "").capitalize()
    main      = d["main"]
    temp      = round(main["temp"])
    feels     = round(main["feels_like"])
    temp_min  = round(main["temp_min"])
    temp_max  = round(main["temp_max"])
    humidity  = main["humidity"]
    pressure  = main.get("pressure", 0)
    wind      = d.get("wind", {})
    wind_kmh  = round(wind.get("speed", 0) * 3.6)
    wind_dir  = _wind_dir(wind.get("deg", 0))
    vis_km    = round(d.get("visibility", 10000) / 1000, 1)
    country   = d.get("sys", {}).get("country", "")
    city_full = f"{city}, {country}" if country else city
    updated   = datetime.now(timezone.utc).strftime("%H:%M")

    header     = discord.ui.TextDisplay(f"## {_emoji(icon_code)} {city_full}")
    sep1       = discord.ui.Separator()
    temp_block = discord.ui.TextDisplay(
        f"# {temp}°C\n"
        f"-# {description}  ·  ressenti **{feels}°C**  ·  {temp_min}° / {temp_max}°"
    )
    try:
        thumbnail    = discord.ui.Thumbnail(discord.ui.UnfurledMediaItem(url=OWM_ICON.format(icon_code)))
        main_section = discord.ui.Section(temp_block, accessory=thumbnail)
    except Exception:
        main_section = temp_block

    sep2    = discord.ui.Separator()
    details = discord.ui.TextDisplay(
        f"💨 **{wind_kmh} km/h** {wind_dir}"
        f"  ·  💧 **{humidity}%**"
        f"  ·  👁 {vis_km} km"
        f"  ·  🌡 {pressure} hPa"
    )
    sep3   = discord.ui.Separator()
    footer = discord.ui.TextDisplay(f"-# Mis à jour à {updated} UTC")

    view.add_item(discord.ui.Container(header, sep1, main_section, sep2, details, sep3, footer))
    return view


def _forecast_view(city: str, d: dict) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=None)

    country   = d.get("city", {}).get("country", "")
    city_full = f"{city}, {country}" if country else city
    updated   = datetime.now(timezone.utc).strftime("%H:%M")

    days: dict[str, dict] = {}
    for item in d.get("list", []):
        dt      = datetime.fromtimestamp(item["dt"], tz=PARIS_TZ)
        day_key = dt.strftime("%Y-%m-%d")
        if day_key not in days:
            days[day_key] = {"dt": dt, "temps": [], "icons": [], "descs": []}
        days[day_key]["temps"].append(item["main"]["temp"])
        days[day_key]["icons"].append(item["weather"][0]["icon"])
        days[day_key]["descs"].append(item["weather"][0]["description"])

    header   = discord.ui.TextDisplay(f"## 📅 {city_full} — Prévisions 5 jours")
    children: list = [header, discord.ui.Separator()]

    for i, (_, info) in enumerate(list(days.items())[:5]):
        dt       = info["dt"]
        date_str = f"**{_weekday_short(dt)} {dt.strftime('%d/%m')}**"
        t_max    = round(max(info["temps"]))
        t_min    = round(min(info["temps"]))
        day_icons = [ic for ic in info["icons"] if ic.endswith("d")]
        icon_pool = day_icons if day_icons else info["icons"]
        icon      = Counter(icon_pool).most_common(1)[0][0]
        desc      = Counter(info["descs"]).most_common(1)[0][0].capitalize()

        children.append(discord.ui.TextDisplay(
            f"{date_str}  {_emoji(icon)}  **{t_max}°** / {t_min}°  ·  {desc}"
        ))
        if i < 4:
            children.append(discord.ui.Separator())

    children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# Mis à jour à {updated} UTC")]
    view.add_item(discord.ui.Container(*children))
    return view


def _day_view(city: str, d: dict, target: date) -> discord.ui.LayoutView:
    """Vue détaillée pour un jour spécifique : matin / après-midi / soir / nuit."""
    view = discord.ui.LayoutView(timeout=None)

    country   = d.get("city", {}).get("country", "")
    city_full = f"{city}, {country}" if country else city
    updated   = datetime.now(timezone.utc).strftime("%H:%M")

    today = datetime.now(PARIS_TZ).date()
    if target == today + timedelta(days=1):
        day_label = "Demain"
    elif target == today:
        day_label = "Aujourd'hui"
    else:
        day_label = f"{_WEEKDAYS_FULL[target.weekday()]} {target.strftime('%d/%m')}"

    # Filtrer les créneaux du jour cible
    slots = [
        item for item in d.get("list", [])
        if datetime.fromtimestamp(item["dt"], tz=PARIS_TZ).date() == target
    ]

    if not slots:
        # Jour hors des 5 jours → fallback vue 5j
        return _forecast_view(city, d)

    all_temps = [s["main"]["temp"] for s in slots]
    t_max     = round(max(all_temps))
    t_min     = round(min(all_temps))
    humidity  = round(sum(s["main"]["humidity"] for s in slots) / len(slots))
    wind_kmh  = round(sum(s["wind"]["speed"] for s in slots) / len(slots) * 3.6)
    wind_deg  = sum(s["wind"].get("deg", 0) for s in slots) / len(slots)

    # Icône globale du jour
    day_icons = [s["weather"][0]["icon"] for s in slots if s["weather"][0]["icon"].endswith("d")]
    icon_pool = day_icons if day_icons else [s["weather"][0]["icon"] for s in slots]
    main_icon = Counter(icon_pool).most_common(1)[0][0]

    header = discord.ui.TextDisplay(
        f"## {_emoji(main_icon)} {day_label} · {city_full}"
    )
    children: list = [header, discord.ui.Separator()]

    # Tranches horaires
    period_lines = []
    for label, h_start, h_end in _PERIODS:
        if h_start < h_end:
            period_slots = [
                s for s in slots
                if h_start <= datetime.fromtimestamp(s["dt"], tz=PARIS_TZ).hour < h_end
            ]
        else:  # nuit : 0-6 ET >=22
            period_slots = [
                s for s in slots
                if datetime.fromtimestamp(s["dt"], tz=PARIS_TZ).hour < h_end
                or datetime.fromtimestamp(s["dt"], tz=PARIS_TZ).hour >= 22
            ]
        if not period_slots:
            continue
        p_temps = [s["main"]["temp"] for s in period_slots]
        p_icons = [s["weather"][0]["icon"] for s in period_slots]
        p_icon  = Counter(p_icons).most_common(1)[0][0]
        p_temp  = round(sum(p_temps) / len(p_temps))
        period_lines.append(f"{label}  {_emoji(p_icon)}  **{p_temp}°C**")

    if period_lines:
        children.append(discord.ui.TextDisplay("\n".join(period_lines)))
        children.append(discord.ui.Separator())

    children.append(discord.ui.TextDisplay(
        f"🌡 **{t_max}°** / {t_min}°"
        f"  ·  💧 {humidity}%"
        f"  ·  💨 {wind_kmh} km/h {_wind_dir(wind_deg)}"
    ))
    children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# Mis à jour à {updated} UTC")]
    view.add_item(discord.ui.Container(*children))
    return view


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Meteo(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._api_key: str = getattr(bot, "config", {}).get("OPENWEATHERMAP_API_KEY", "") or ""

    def _check_status(self, r: requests.Response, city: str) -> Optional[dict]:
        if r.status_code == 401:
            return {"error": "Clé API OWM invalide ou pas encore activée (délai jusqu'à 2h après création)"}
        if r.status_code == 404:
            return {"error": f"Ville introuvable : {city!r}"}
        if r.status_code == 429:
            return {"error": "Limite de requêtes OWM atteinte"}
        if not r.ok:
            return {"error": f"Erreur OWM {r.status_code}"}
        return None

    def _fetch_current(self, city: str) -> dict:
        if not self._api_key:
            return {"error": "Clé API OWM manquante (OPENWEATHERMAP_API_KEY)"}
        try:
            r = requests.get(
                f"{OWM_BASE}/weather",
                params={"q": city, "appid": self._api_key, "units": "metric", "lang": "fr"},
                timeout=8,
            )
            err = self._check_status(r, city)
            return err if err else r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def _fetch_forecast(self, city: str) -> dict:
        if not self._api_key:
            return {"error": "Clé API OWM manquante (OPENWEATHERMAP_API_KEY)"}
        try:
            r = requests.get(
                f"{OWM_BASE}/forecast",
                params={"q": city, "appid": self._api_key, "units": "metric", "lang": "fr", "cnt": 40},
                timeout=8,
            )
            err = self._check_status(r, city)
            return err if err else r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    async def _tool_weather(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        city         = (tc.arguments.get("city") or "").strip()
        weather_type = (tc.arguments.get("type") or "current").strip()
        target_date  = (tc.arguments.get("target_date") or "").strip()

        if not city:
            return ToolResponseRecord(tc.id, {"error": "Ville manquante"}, datetime.now(timezone.utc))

        loop = asyncio.get_event_loop()
        if weather_type == "forecast":
            raw = await loop.run_in_executor(None, self._fetch_forecast, city)
        else:
            raw = await loop.run_in_executor(None, self._fetch_current, city)

        if "error" in raw:
            return ToolResponseRecord(tc.id, {"error": raw["error"]}, datetime.now(timezone.utc))

        city_name = raw.get("name") or raw.get("city", {}).get("name") or city

        if weather_type == "forecast":
            day_label = f" pour {target_date}" if target_date else " 5 jours"
            llm_summary = (
                f"Prévisions météo{day_label} affichées pour {city_name}. LayoutView envoyé. "
                f"Pour d'autres questions sur la météo de {city_name}, rappelle get_weather."
            )
        else:
            main = raw.get("main", {})
            temp = round(main.get("temp", 0))
            desc = (raw.get("weather") or [{}])[0].get("description", "")
            llm_summary = (
                f"Météo actuelle affichée pour {city_name} : {temp}°C, {desc}. LayoutView envoyé. "
                f"Pour d'autres questions sur la météo de {city_name} (prévisions, demain, semaine…), rappelle get_weather."
            )

        return ToolResponseRecord(tc.id, {
            "_tool":        "get_weather",
            "_llm_summary": llm_summary,
            "type":         weather_type,
            "city":         city_name,
            "target_date":  target_date,
            "data":         raw,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_weather",
                description=(
                    "Récupère et affiche la météo d'une ville. "
                    "type='current' → météo maintenant. type='forecast' → prévisions. "
                    "Pour un jour précis ('demain', 'jeudi'…) : type='forecast' + target_date='demain'/'lundi'/etc. "
                    "Questions de suivi SANS ville explicite ('et demain ?', 'et la semaine ?', 'pour lundi ?', "
                    "'il fait quoi demain ?') → réutiliser la ville du dernier appel get_weather visible en contexte."
                ),
                properties={
                    "city": {
                        "type": "string",
                        "description": "Ville (ex: Paris, Lyon, Tokyo)",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["current", "forecast"],
                        "description": "'current' pour maintenant, 'forecast' pour les prévisions",
                    },
                    "target_date": {
                        "type": "string",
                        "description": (
                            "Jour visé, uniquement avec type='forecast'. "
                            "Valeurs : 'demain', 'lundi', 'mardi', … ou format YYYY-MM-DD. "
                            "Laisser vide pour le panorama 5 jours."
                        ),
                    },
                },
                function=self._tool_weather,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Meteo(bot))
