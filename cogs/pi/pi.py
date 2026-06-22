"""Cog Pi — données du capteur DHT22 du Raspberry Pi exposées comme outil LLM.

Le capteur est câblé sur la broche GPIO 4 (board.D4).
Ce cog se charge silencieusement si les libs adafruit sont absentes (autre OS).

Formules :
- Point de rosée  : Magnus-Tetens (Heinrich Gustav Magnus-Tetens)
- Humidex         : formule de Maurice Richard
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

import discord
from discord.ext import commands

from common.discord_ui import layout_with_commentary
from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Pi")

try:
    import adafruit_dht
    import board
    _DHT_AVAILABLE = True
except Exception as _dht_import_err:
    _DHT_AVAILABLE = False
    logger.warning(f"adafruit_dht indisponible — outil DHT22 désactivé. ({type(_dht_import_err).__name__}: {_dht_import_err})")

# Nombre maximum de tentatives avant d'abandonner (le DHT22 est flaky)
_MAX_RETRIES = 5
# Délai entre deux lectures (spec DHT22 : min 2s)
_READ_DELAY = 2.0
# Durée de vie du cache en secondes (évite de spammer le capteur)
_CACHE_TTL = 30


def build_sensor_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit le widget LayoutView pour les données du capteur DHT22."""
    if "error" in data:
        return None

    temp     = data["temperature"]
    humidity = data["humidity"]
    humidex  = data["humidex"]
    dew      = data["dew_point"]
    read_at  = data.get("read_at", "")

    header = discord.ui.TextDisplay(f"## Chez MARIA")
    sep1   = discord.ui.Separator()
    main   = discord.ui.TextDisplay(
        f"# {temp}°C\n"
        f"-# humidité **{humidity}%**  ·  ressenti **{humidex}°C**  ·  point de rosée {dew}°C"
    )
    sep2   = discord.ui.Separator()
    footer = discord.ui.TextDisplay(f"-# Capteur DHT22 intégré ·  {read_at}")

    container = discord.ui.Container(header, sep1, main, sep2, footer)
    return layout_with_commentary(container, commentary)


def _read_sensor() -> dict:
    """Lecture bloquante du DHT22. Retry sur RuntimeError (bruit électrique fréquent)."""
    if not _DHT_AVAILABLE:
        return {"error": "Capteur DHT22 indisponible sur cette plateforme."}

    device = adafruit_dht.DHT22(board.D4, use_pulseio=False)
    last_error = "inconnue"
    try:
        for _ in range(_MAX_RETRIES):
            try:
                temperature = device.temperature
                humidity    = device.humidity
                if temperature is None or humidity is None:
                    last_error = "valeurs nulles"
                    continue

                # Point de rosée (Magnus-Tetens)
                alpha  = math.log(humidity / 100.0) + (17.27 * temperature) / (237.3 + temperature)
                rosee  = (237.3 * alpha) / (17.27 - alpha)

                # Humidex (Maurice Richard)
                humidex = temperature + 0.5555 * (
                    6.11 * math.exp(5417.753 * (1 / 273.16 - 1 / (273.15 + rosee))) - 10
                )

                result = {
                    "temperature":  round(temperature, 1),
                    "humidity":     round(humidity, 1),
                    "dew_point":    round(rosee, 1),
                    "humidex":      round(humidex, 1),
                    "read_at":      datetime.now(timezone.utc).strftime("%H:%M UTC"),
                }
                result["_tool"] = "get_sensor_data"
                result["_llm_summary"] = (
                    f"Capteur DHT22 (salle serveur) : {result['temperature']}°C, "
                    f"humidité {result['humidity']}%, ressenti {result['humidex']}°C, "
                    f"point de rosée {result['dew_point']}°C. Widget affiché."
                )
                return result
            except RuntimeError as e:
                last_error = str(e)

            time.sleep(_READ_DELAY)

        logger.warning(f"DHT22 : échec après {_MAX_RETRIES} tentatives — dernière erreur : {last_error}")
        return {"error": f"Lecture DHT22 échouée ({last_error})."}
    finally:
        device.exit()


class Pi(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._cache: dict = {}
        self._cache_ts: float = 0.0

    async def _tool_get_sensor_data(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        now = time.monotonic()
        if self._cache and (now - self._cache_ts) < _CACHE_TTL:
            return ToolResponseRecord(tc.id, self._cache, datetime.now(timezone.utc))

        data = await asyncio.to_thread(_read_sensor)

        if "error" not in data:
            self._cache    = data
            self._cache_ts = now

        return ToolResponseRecord(tc.id, data, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_sensor_data",
                description=(
                    "Retourne les données en temps réel du capteur DHT22 installé chez toi "
                    "(Raspberry Pi hébergeant le bot) : température ambiante, humidité, "
                    "point de rosée et humidex. "
                    "Utilise cet outil quand on te demande la température ou l'humidité "
                    "'chez toi', 'dans ta pièce', ou toute question sur l'ambiance de la salle serveur."
                ),
                properties={},
                function=self._tool_get_sensor_data,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Pi(bot))
