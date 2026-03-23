import logging
import random
from pathlib import Path

import discord
from discord.ext import commands, tasks

logger = logging.getLogger("MARIA.Status")

STATUSES_FILE = Path("data/statuses.txt")
INTERVAL_MINUTES = 20


def _load_statuses() -> list[tuple[str, str]]:
    """Lit le fichier et retourne une liste de (type, texte).

    Format du fichier — une entrée par ligne :
        playing:<texte>    →  "En train de jouer à <texte>"
        watching:<texte>   →  "Regarde <texte>"
        listening:<texte>  →  "Écoute <texte>"
        <texte>            →  statut personnalisé
    Les lignes vides et commençant par # sont ignorées.
    """
    if not STATUSES_FILE.exists():
        logger.warning(f"Fichier de statuts introuvable : {STATUSES_FILE}")
        return []

    entries: list[tuple[str, str]] = []
    for raw in STATUSES_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("playing:"):
            entries.append(("playing", line[len("playing:"):].strip()))
        elif line.startswith("watching:"):
            entries.append(("watching", line[len("watching:"):].strip()))
        elif line.startswith("listening:"):
            entries.append(("listening", line[len("listening:"):].strip()))
        else:
            entries.append(("custom", line))
    return entries


def _format_status(kind: str, text: str) -> str:
    """Retourne une description lisible du statut tel qu'il apparaît sur Discord."""
    match kind:
        case "playing":
            return f"joue à {text}"
        case "watching":
            return f"regarde {text}"
        case "listening":
            return f"écoute {text}"
        case _:
            return text


def _make_activity(kind: str, text: str) -> discord.BaseActivity:
    match kind:
        case "playing":
            return discord.Game(name=text)
        case "watching":
            return discord.Activity(type=discord.ActivityType.watching, name=text)
        case "listening":
            return discord.Activity(type=discord.ActivityType.listening, name=text)
        case _:
            return discord.CustomActivity(name=text)


class Status(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._statuses: list[tuple[str, str]] = []
        self._last: tuple[str, str] | None = None
        self.current_status: str = ""

    async def cog_load(self) -> None:
        self._statuses = _load_statuses()
        logger.info(f"{len(self._statuses)} statut(s) chargé(s) depuis {STATUSES_FILE}")
        self._rotate_task.start()

    async def cog_unload(self) -> None:
        self._rotate_task.cancel()

    @tasks.loop(minutes=INTERVAL_MINUTES)
    async def _rotate_task(self) -> None:
        await self._set_random_status()

    @_rotate_task.before_loop
    async def _before_rotate(self) -> None:
        await self.bot.wait_until_ready()

    async def _set_random_status(self) -> None:
        self._statuses = _load_statuses()
        if not self._statuses:
            return

        pool = [s for s in self._statuses if s != self._last] or self._statuses
        chosen = random.choice(pool)
        self._last = chosen

        kind, text = chosen
        activity = _make_activity(kind, text)
        await self.bot.change_presence(activity=activity)
        self.current_status = _format_status(kind, text)
        logger.debug(f"Statut → [{kind}] {text}")

    @commands.command(name="status", hidden=True)
    @commands.is_owner()
    async def cmd_status(self, ctx: commands.Context) -> None:
        """Force un nouveau statut aléatoire et remet le timer à zéro."""
        await self._set_random_status()
        self._rotate_task.cancel()
        self._rotate_task.start()
        await ctx.message.delete()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Status(bot))
