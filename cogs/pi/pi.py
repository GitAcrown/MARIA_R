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

from discord.ext import commands

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
_MAX_RETRIES = 10
# Durée de vie du cache en secondes (évite de spammer le capteur)
_CACHE_TTL = 30


def _read_sensor() -> dict:
    """Lecture bloquante du DHT22. Retry sur RuntimeError (bruit électrique fréquent)."""
    if not _DHT_AVAILABLE:
        return {"error": "Capteur DHT22 indisponible sur cette plateforme."}

    device = adafruit_dht.DHT22(board.D4, use_pulseio=False)
    try:
        for _ in range(_MAX_RETRIES):
            try:
                temperature = device.temperature
                humidity    = device.humidity
                if temperature is None or humidity is None:
                    continue

                # Point de rosée (Magnus-Tetens)
                alpha  = math.log(humidity / 100.0) + (17.27 * temperature) / (237.3 + temperature)
                rosee  = (237.3 * alpha) / (17.27 - alpha)

                # Humidex (Maurice Richard)
                humidex = temperature + 0.5555 * (
                    6.11 * math.exp(5417.753 * (1 / 273.16 - 1 / (273.15 + rosee))) - 10
                )

                return {
                    "temperature":  round(temperature, 1),
                    "humidity":     round(humidity, 1),
                    "dew_point":    round(rosee, 1),
                    "humidex":      round(humidex, 1),
                    "read_at":      datetime.now(timezone.utc).strftime("%H:%M UTC"),
                }
            except RuntimeError:
                continue

        return {"error": "Lecture DHT22 échouée après plusieurs tentatives (bruit sur le bus)."}
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
