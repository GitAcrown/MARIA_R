"""Cog Fun — outils de jeu et de hasard pour Maria."""

import logging
import random
import re
from datetime import datetime, timezone

from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Fun")

_DICE_RE = re.compile(r"^(\d+)?[dD](\d+)([+-]\d+)?$")


def _parse_dice(notation: str) -> tuple[int, int, int] | None:
    m = _DICE_RE.match(notation.strip())
    if not m:
        return None
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    mod = int(m.group(3)) if m.group(3) else 0
    return count, sides, mod


class Fun(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _tool_roll(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        notation = (tc.arguments.get("notation") or "d6").strip()
        parsed = _parse_dice(notation)
        if not parsed:
            return ToolResponseRecord(tc.id, {"error": f"Notation invalide : {notation!r}. Exemples : d6, 2d20, 3d6+2"}, datetime.now(timezone.utc))
        count, sides, mod = parsed
        if count > 100 or sides > 10_000 or count < 1 or sides < 2:
            return ToolResponseRecord(tc.id, {"error": "Paramètres hors limites."}, datetime.now(timezone.utc))
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls) + mod
        label = f"{count}d{sides}" + (f"{mod:+d}" if mod else "")
        return ToolResponseRecord(tc.id, {"notation": label, "rolls": rolls, "modifier": mod, "total": total}, datetime.now(timezone.utc))

    async def _tool_flip(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        result = random.choice(["Pile", "Face"])
        return ToolResponseRecord(tc.id, {"result": result}, datetime.now(timezone.utc))

    async def _tool_pick(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        raw = tc.arguments.get("options", "")
        options = [o.strip() for o in raw.split(",") if o.strip()]
        if len(options) < 2:
            return ToolResponseRecord(tc.id, {"error": "Minimum 2 options séparées par des virgules."}, datetime.now(timezone.utc))
        chosen = random.choice(options)
        return ToolResponseRecord(tc.id, {"chosen": chosen, "options": options}, datetime.now(timezone.utc))

    async def _tool_rate(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        subject = (tc.arguments.get("subject") or "ça").strip()
        score = random.randint(0, 10)
        return ToolResponseRecord(tc.id, {"subject": subject, "score": score}, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="roll_dice",
                description="Lance des dés. Notation XdY ou XdY+Z (ex: d6, 2d20, 3d6+2). Utilise quand quelqu'un veut lancer des dés ou a besoin d'un résultat aléatoire chiffré.",
                properties={"notation": {"type": "string", "description": "Notation des dés. Défaut : d6"}},
                function=self._tool_roll,
            ),
            Tool(
                name="flip_coin",
                description="Lance une pièce. Pile ou face.",
                properties={},
                function=self._tool_flip,
            ),
            Tool(
                name="pick_random",
                description="Choisit aléatoirement parmi une liste d'options. Utile pour départager des choix.",
                properties={"options": {"type": "string", "description": "Options séparées par des virgules"}},
                function=self._tool_pick,
            ),
            Tool(
                name="rate",
                description="Donne une note aléatoire /10 à quelque chose. Utilise quand quelqu'un demande de noter ou évaluer quelque chose.",
                properties={"subject": {"type": "string", "description": "Ce qu'on note"}},
                function=self._tool_rate,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Fun(bot))
