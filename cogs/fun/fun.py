"""Cog Fun — outils de jeu et de hasard pour Maria."""

import logging
import random
import re
from datetime import datetime, timezone

import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Fun")

_DICE_RE = re.compile(r"^(\d+)?[dD](\d+)([+-]\d+)?$")


def _parse_dice(notation: str) -> tuple[int, int, int] | None:
    """Parse XdY+Z. Retourne (count, sides, modifier) ou None."""
    m = _DICE_RE.match(notation.strip())
    if not m:
        return None
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    mod = int(m.group(3)) if m.group(3) else 0
    return count, sides, mod


# ---------------------------------------------------------------------------
# LayoutViews
# ---------------------------------------------------------------------------

class _ResultView(discord.ui.LayoutView):
    """Affiche un résultat propre dans un Container."""

    def __init__(self, title: str, main: str, sub: str | None = None):
        super().__init__(timeout=None)
        children: list = [
            discord.ui.TextDisplay(f"### {title}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(f"**{main}**"),
        ]
        if sub:
            children.append(discord.ui.TextDisplay(f"-# {sub}"))
        self.add_item(discord.ui.Container(*children))


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Fun(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _send(self, ctx, title: str, main: str, sub: str | None = None) -> None:
        if ctx and ctx.trigger_message:
            await ctx.trigger_message.channel.send(view=_ResultView(title, main, sub))

    # --- Outils ---

    async def _tool_roll(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        notation = (tc.arguments.get("notation") or "d6").strip()
        parsed = _parse_dice(notation)
        if not parsed:
            return ToolResponseRecord(
                tc.id,
                {"error": f"Notation invalide : {notation!r}. Exemples : d6, 2d20, 3d6+2"},
                datetime.now(timezone.utc),
            )
        count, sides, mod = parsed
        if count > 100 or sides > 10_000 or count < 1 or sides < 2:
            return ToolResponseRecord(tc.id, {"error": "Paramètres hors limites."}, datetime.now(timezone.utc))

        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls) + mod
        label = f"{count}d{sides}" + (f"{mod:+d}" if mod else "")

        if count > 1 or mod:
            parts = " + ".join(str(r) for r in rolls)
            sub = (parts + f" {mod:+d}") if mod else parts
        else:
            sub = None

        await self._send(ctx, f"Lancer — {label}", str(total), sub)
        return ToolResponseRecord(
            tc.id,
            {"notation": label, "rolls": rolls, "modifier": mod, "total": total},
            datetime.now(timezone.utc),
        )

    async def _tool_flip(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        result = random.choice(["Pile", "Face"])
        await self._send(ctx, "Pile ou face", result)
        return ToolResponseRecord(tc.id, {"result": result}, datetime.now(timezone.utc))

    async def _tool_pick(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        raw = tc.arguments.get("options", "")
        options = [o.strip() for o in raw.split(",") if o.strip()]
        if len(options) < 2:
            return ToolResponseRecord(
                tc.id,
                {"error": "Minimum 2 options séparées par des virgules."},
                datetime.now(timezone.utc),
            )
        chosen = random.choice(options)
        sub = " · ".join(f"[{o}]" if o == chosen else o for o in options)
        await self._send(ctx, "Choix aléatoire", chosen, sub)
        return ToolResponseRecord(tc.id, {"chosen": chosen, "options": options}, datetime.now(timezone.utc))

    async def _tool_rate(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        subject = (tc.arguments.get("subject") or "ça").strip()
        score = random.randint(0, 10)
        bar = "█" * score + "░" * (10 - score)
        await self._send(ctx, f"Note — {subject[:50]}", f"{score} / 10", bar)
        return ToolResponseRecord(tc.id, {"subject": subject, "score": score}, datetime.now(timezone.utc))

    # --- Déclaration des outils inter-cogs ---

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="roll_dice",
                description="Lance des dés. Notation XdY ou XdY+Z (ex: d6, 2d20, 3d6+2). Utilise quand quelqu'un veut lancer des dés ou a besoin d'un résultat aléatoire chiffré.",
                properties={
                    "notation": {"type": "string", "description": "Notation des dés. Défaut : d6"},
                },
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
                properties={
                    "options": {"type": "string", "description": "Options séparées par des virgules"},
                },
                function=self._tool_pick,
            ),
            Tool(
                name="rate",
                description="Donne une note aléatoire /10 à quelque chose. Utilise quand quelqu'un demande de noter ou évaluer quelque chose.",
                properties={
                    "subject": {"type": "string", "description": "Ce qu'on note"},
                },
                function=self._tool_rate,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Fun(bot))
