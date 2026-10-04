"""Attention par membre + fatigue salon.

Remplace l'ancienne « température » de salon : le bot se tourne vers qui lui parle
(attention), et devient fainéante si on la sollicite trop (fatigue).
"""

from __future__ import annotations

import time
from collections import OrderedDict

# --- Attention membre --------------------------------------------------------
ATTENTION_HALF_LIFE = 210.0
MAX_ATTENTION = 6.0
ATTENTION_WARM = 0.9
ATTENTION_HOT = 1.8
COMPETITIVE_DECAY = 0.85
_MAX_MEMBERS = 2000

# --- Fatigue salon -----------------------------------------------------------
FATIGUE_HALF_LIFE = 150.0
MAX_FATIGUE = 6.0
FATIGUE_TIRED = 1.6
FATIGUE_EXHAUSTED = 3.0
_MAX_CHANNELS = 500


class MemberAttention:
    """Score d'attention (guild, user) avec decay temporel + compétitif."""

    def __init__(self, half_life: float = ATTENTION_HALF_LIFE) -> None:
        self._half_life = max(1.0, half_life)
        self._scores: OrderedDict[tuple[int, int], tuple[float, float]] = OrderedDict()

    def value(self, guild_id: int, user_id: int) -> float:
        key = (guild_id, user_id)
        entry = self._scores.get(key)
        if entry is None:
            return 0.0
        heat, at = entry
        elapsed = max(0.0, time.monotonic() - at)
        return heat * 0.5 ** (elapsed / self._half_life)

    def bump(self, guild_id: int, user_id: int, amount: float = 1.0) -> float:
        now = time.monotonic()
        # Decay compétitif : les autres scores du guild fondent.
        for key in list(self._scores):
            if key[0] != guild_id or key[1] == user_id:
                continue
            cur = self.value(key[0], key[1]) * COMPETITIVE_DECAY
            if cur < 0.05:
                self._scores.pop(key, None)
            else:
                self._scores[key] = (cur, now)
        heat = min(MAX_ATTENTION, self.value(guild_id, user_id) + amount)
        key = (guild_id, user_id)
        self._scores[key] = (heat, now)
        self._scores.move_to_end(key)
        while len(self._scores) > _MAX_MEMBERS:
            self._scores.popitem(last=False)
        return heat

    def is_hot(self, guild_id: int, user_id: int) -> bool:
        return self.value(guild_id, user_id) >= ATTENTION_HOT

    def is_warm(self, guild_id: int, user_id: int) -> bool:
        return self.value(guild_id, user_id) >= ATTENTION_WARM

    def normalized(self, guild_id: int, user_id: int) -> float:
        """0–1 pour JEV (laziness)."""
        return min(1.0, self.value(guild_id, user_id) / ATTENTION_HOT)


class ChannelFatigue:
    """Charge du bot sur un salon (bumpée à chaque vraie réponse texte)."""

    def __init__(self, half_life: float = FATIGUE_HALF_LIFE) -> None:
        self._half_life = max(1.0, half_life)
        self._scores: OrderedDict[int, tuple[float, float]] = OrderedDict()

    def value(self, channel_id: int) -> float:
        entry = self._scores.get(channel_id)
        if entry is None:
            return 0.0
        heat, at = entry
        elapsed = max(0.0, time.monotonic() - at)
        return heat * 0.5 ** (elapsed / self._half_life)

    def bump(self, channel_id: int, amount: float = 1.0) -> float:
        heat = min(MAX_FATIGUE, self.value(channel_id) + amount)
        self._scores[channel_id] = (heat, time.monotonic())
        self._scores.move_to_end(channel_id)
        while len(self._scores) > _MAX_CHANNELS:
            self._scores.popitem(last=False)
        return heat

    def is_tired(self, channel_id: int) -> bool:
        return self.value(channel_id) >= FATIGUE_TIRED

    def is_exhausted(self, channel_id: int) -> bool:
        return self.value(channel_id) >= FATIGUE_EXHAUSTED

    def normalized(self, channel_id: int) -> float:
        return min(1.0, self.value(channel_id) / FATIGUE_EXHAUSTED)


class SocialFocus:
    """Façade attention + fatigue pour le cog Chat."""

    def __init__(self) -> None:
        self.attention = MemberAttention()
        self.fatigue = ChannelFatigue()

    def soften_mention(
        self,
        verdict: str,
        *,
        attention: float,
        fatigue: float,
        confidence: float = 1.0,
        react_min_conf: float = 0.65,
    ) -> str:
        """`ignore` → respond si attention HOT (pas fatiguée) ; react seulement conf haute."""
        if verdict != "ignore":
            return verdict
        if fatigue >= FATIGUE_TIRED:
            return "ignore"
        # Membre très présent : un ignore borderline peut devenir une vraie réponse.
        if attention >= ATTENTION_HOT:
            return "respond"
        # Warm : uniquement si JEV était déjà confiant sur un react (pas un ignore inventé).
        if attention >= ATTENTION_WARM and confidence >= react_min_conf:
            return "react"
        return "ignore"

    def soften_followup(
        self,
        verdict: str,
        *,
        attention: float,
        chain_depth: int,
        fatigue: float,
        confidence: float = 1.0,
        react_min_conf: float = 0.65,
    ) -> str:
        """Follow-up : jamais upgrade ignore→respond ; react seulement conf haute."""
        if verdict == "respond":
            if fatigue >= FATIGUE_TIRED or chain_depth >= 2 or attention < ATTENTION_WARM:
                if confidence >= react_min_conf + (0.1 if fatigue >= FATIGUE_TIRED else 0.0):
                    return "react"
                return "ignore"
            return "respond"
        if verdict == "react":
            need = react_min_conf + (0.1 if fatigue >= FATIGUE_TIRED else 0.0)
            return "react" if confidence >= need else "ignore"
        # ignore : soften léger seulement si attention haute et pas fatiguée
        if (
            fatigue < FATIGUE_TIRED
            and attention >= ATTENTION_HOT
            and chain_depth == 0
            and confidence >= react_min_conf
        ):
            return "react"
        return "ignore"

    def followup_deadline_factor(self, channel_id: int) -> float:
        return 0.6 if self.fatigue.is_tired(channel_id) else 1.0

    def followup_max_checks(self, channel_id: int, default: int = 2) -> int:
        return 1 if self.fatigue.is_tired(channel_id) else default

    def allow_typing_extend(self, channel_id: int) -> bool:
        return not self.fatigue.is_tired(channel_id)
