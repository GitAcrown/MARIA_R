"""Cog Dyn — enregistre les DynamicItem d'onglets et retire les boutons à l'échéance."""

from __future__ import annotations

import logging

from discord.ext import commands, tasks

from common.dyn_widgets import TabButton, TabSelect, sweep_expired
from common.bookmarks import BookmarkButton, sweep_expired as sweep_bookmarks

logger = logging.getLogger("MARIA.Dyn")


class Dyn(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_load(self) -> None:
        self.bot.add_dynamic_items(TabButton, TabSelect, BookmarkButton)
        self.sweep.start()
        logger.info("DynamicItems onglets + bookmark enregistrés")

    async def cog_unload(self) -> None:
        self.sweep.cancel()
        self.bot.remove_dynamic_items(TabButton, TabSelect, BookmarkButton)

    @tasks.loop(seconds=30)
    async def sweep(self) -> None:
        try:
            await sweep_expired(self.bot)
            await sweep_bookmarks(self.bot)
        except Exception:
            logger.exception("sweep onglets")

    @sweep.before_loop
    async def before_sweep(self) -> None:
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Dyn(bot))
