"""Cog Météo — OpenWeatherMap avec rendu LayoutView."""

import asyncio
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import requests
import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Meteo")

OWM_BASE = "https://api.openweathermap.org/data/2.5"
OWM_ICON = "https://openweathermap.org/img/wn/{}@2x.png"

_ICON_EMOJI: dict[str, str] = {
    "01": "☀️", "02": "🌤️", "03": "⛅", "04": "☁️",
    "09": "🌦️", "10": "🌧️", "11": "⛈️", "13": "🌨️", "50": "🌫️",
}

_WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
_WIND_DIRS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


def _emoji(icon: str) -> str:
    return _ICON_EMOJI.get(icon[:2], "🌡️")


def _wind_dir(deg: float) -> str:
    return _WIND_DIRS[round(deg / 45) % 8]


def _weekday(dt: datetime) -> str:
    return _WEEKDAYS[dt.weekday()]


# ---------------------------------------------------------------------------
# Builders de vue
# ---------------------------------------------------------------------------

def build_weather_view(data: dict) -> Optional[discord.ui.LayoutView]:
    """Construit le LayoutView à partir des données retournées par l'outil."""
    if "error" in data or "data" not in data:
        return None
    weather_type = data.get("type", "current")
    city = data.get("city", "?")
    raw = data["data"]
    try:
        if weather_type == "forecast":
            return _forecast_view(city, raw)
        return _current_view(city, raw)
    except Exception as e:
        logger.error(f"Erreur build_weather_view: {e}", exc_info=True)
        return None


def _current_view(city: str, d: dict) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=None)

    weather = d["weather"][0]
    icon_code = weather.get("icon", "01d")
    description = weather.get("description", "").capitalize()
    main = d["main"]
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

    # --- Header ---
    header = discord.ui.TextDisplay(f"## {_emoji(icon_code)} {city_full}")
    sep1 = discord.ui.Separator()

    # --- Température + icône OWM en thumbnail ---
    temp_block = discord.ui.TextDisplay(
        f"# {temp}°C\n"
        f"-# {description}  ·  ressenti **{feels}°C**  ·  {temp_min}° / {temp_max}°"
    )
    try:
        thumbnail = discord.ui.Thumbnail(
            discord.ui.UnfurledMediaItem(url=OWM_ICON.format(icon_code))
        )
        main_section = discord.ui.Section(temp_block, accessory=thumbnail)
    except Exception:
        main_section = temp_block

    sep2 = discord.ui.Separator()

    # --- Détails ---
    details = discord.ui.TextDisplay(
        f"💨 **{wind_kmh} km/h** {wind_dir}"
        f"  ·  💧 **{humidity}%**"
        f"  ·  👁 {vis_km} km"
        f"  ·  🌡 {pressure} hPa"
    )

    sep3 = discord.ui.Separator()
    footer = discord.ui.TextDisplay(f"-# Mis à jour à {updated} UTC")

    view.add_item(discord.ui.Container(
        header, sep1, main_section, sep2, details, sep3, footer
    ))
    return view


def _forecast_view(city: str, d: dict) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=None)

    city_obj  = d.get("city", {})
    country   = city_obj.get("country", "")
    city_full = f"{city}, {country}" if country else city
    updated   = datetime.now(timezone.utc).strftime("%H:%M")

    # Grouper les intervalles 3h par jour calendaire
    days: dict[str, dict] = {}
    for item in d.get("list", []):
        dt      = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        day_key = dt.strftime("%Y-%m-%d")
        if day_key not in days:
            days[day_key] = {"dt": dt, "temps": [], "icons": [], "descs": []}
        days[day_key]["temps"].append(item["main"]["temp"])
        days[day_key]["icons"].append(item["weather"][0]["icon"])
        days[day_key]["descs"].append(item["weather"][0]["description"])

    header = discord.ui.TextDisplay(f"## 📅 {city_full} — Prévisions 5 jours")
    children: list = [header, discord.ui.Separator()]

    for i, (_, info) in enumerate(list(days.items())[:5]):
        dt       = info["dt"]
        date_str = f"**{_weekday(dt)} {dt.strftime('%d/%m')}**"
        t_max    = round(max(info["temps"]))
        t_min    = round(min(info["temps"]))

        # Icône dominante sur les heures de jour
        day_icons = [ic for ic in info["icons"] if ic.endswith("d")]
        icon_pool = day_icons if day_icons else info["icons"]
        icon      = Counter(icon_pool).most_common(1)[0][0]

        desc = Counter(info["descs"]).most_common(1)[0][0].capitalize()

        children.append(discord.ui.TextDisplay(
            f"{date_str}  {_emoji(icon)}  **{t_max}°** / {t_min}°  ·  {desc}"
        ))
        if i < 4:
            children.append(discord.ui.Separator())

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
        """Retourne un dict d'erreur si le statut HTTP n'est pas OK, sinon None."""
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
            if err:
                return err
            return r.json()
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
            if err:
                return err
            return r.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    async def _tool_weather(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        city = (tc.arguments.get("city") or "").strip()
        weather_type = (tc.arguments.get("type") or "current").strip()
        if not city:
            return ToolResponseRecord(tc.id, {"error": "Ville manquante"}, datetime.now(timezone.utc))

        loop = asyncio.get_event_loop()
        if weather_type == "forecast":
            raw = await loop.run_in_executor(None, self._fetch_forecast, city)
        else:
            raw = await loop.run_in_executor(None, self._fetch_current, city)

        if "error" in raw:
            return ToolResponseRecord(tc.id, {"error": raw["error"]}, datetime.now(timezone.utc))

        # Récupérer le nom normalisé par OWM
        city_name = raw.get("name") or raw.get("city", {}).get("name") or city

        if weather_type == "forecast":
            llm_summary = f"Prévisions 5 jours affichées pour {city_name}. LayoutView envoyé dans le salon."
        else:
            main = raw.get("main", {})
            temp = round(main.get("temp", 0))
            desc = (raw.get("weather") or [{}])[0].get("description", "")
            llm_summary = f"Météo actuelle affichée pour {city_name} : {temp}°C, {desc}. LayoutView envoyé dans le salon."

        return ToolResponseRecord(tc.id, {
            "_tool": "get_weather",
            "_llm_summary": llm_summary,
            "type": weather_type,
            "city": city_name,
            "data": raw,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_weather",
                description=(
                    "Affiche la météo d'une ville via OpenWeatherMap. "
                    "type='current' pour maintenant, 'forecast' pour les 5 prochains jours. "
                    "Utilise dès qu'on demande la météo, le temps qu'il fait, les prévisions, "
                    "ou une question de suivi sur une météo déjà affichée (ex: 'et la semaine ?', "
                    "'les prochains jours ?', 'etr la suite ?') — dans ce cas, réutilise la ville "
                    "du dernier appel get_weather visible dans le contexte."
                ),
                properties={
                    "city": {"type": "string", "description": "Ville (ex: Paris, Lyon, Tokyo)"},
                    "type": {"type": "string", "enum": ["current", "forecast"], "description": "'current' ou 'forecast'"},
                },
                function=self._tool_weather,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Meteo(bot))
