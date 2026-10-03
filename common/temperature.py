"""« Température » d'un salon : plus on parle d'elle, plus elle est encline à se manifester.

Chaque mention de son nom chauffe le salon ; la chaleur retombe de moitié toutes les
`HALF_LIFE` secondes. Sert à adoucir un verdict JEV « ignorer » (mention passive) quand
on l'appelle beaucoup dans le tchat.
"""

from __future__ import annotations

import time
from collections import OrderedDict

HALF_LIFE_SECONDS = 180.0
MAX_HEAT = 6.0
WARM = 0.9   # ≥ : un « ignorer » JEV devient une réaction
HOT = 1.8    # ≥ : un « ignorer » JEV devient une vraie réponse
_MAX_CHANNELS = 500


class ChannelTemperature:
    def __init__(self, half_life: float = HALF_LIFE_SECONDS) -> None:
        self._half_life = max(1.0, half_life)
        self._heat: OrderedDict[int, tuple[float, float]] = OrderedDict()

    def value(self, channel_id: int) -> float:
        """Chaleur actuelle du salon (après décroissance)."""
        entry = self._heat.get(channel_id)
        if entry is None:
            return 0.0
        heat, at = entry
        elapsed = max(0.0, time.monotonic() - at)
        return heat * 0.5 ** (elapsed / self._half_life)

    def bump(self, channel_id: int, amount: float = 1.0) -> float:
        heat = min(MAX_HEAT, self.value(channel_id) + amount)
        self._heat[channel_id] = (heat, time.monotonic())
        self._heat.move_to_end(channel_id)
        while len(self._heat) > _MAX_CHANNELS:
            self._heat.popitem(last=False)
        return heat

    @staticmethod
    def soften(verdict: str, heat: float) -> str:
        """`ignore` → `react` / `respond` selon la chaleur (avant la mention en cours)."""
        if verdict != "ignore":
            return verdict
        if heat >= HOT:
            return "respond"
        if heat >= WARM:
            return "react"
        return verdict
