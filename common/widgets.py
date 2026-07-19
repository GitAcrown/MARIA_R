"""Registre centralisé des builders LayoutView pour les outils LLM."""

from typing import Callable, Optional

import discord

Builder = Callable[[dict, str], Optional[discord.ui.LayoutView]]


def _load_builders() -> dict[str, Builder]:
    builders: dict[str, Builder] = {}

    try:
        from cogs.meteo.meteo import build_weather_view
        builders["get_weather"] = build_weather_view
    except ImportError:
        pass

    try:
        from cogs.tmdb.tmdb import build_media_view
        builders["search_media"] = build_media_view
    except ImportError:
        pass

    try:
        from cogs.steam.steam import build_game_view
        builders["search_game"] = build_game_view
    except ImportError:
        pass

    try:
        from cogs.football.football import build_football_view
        builders["get_football"] = build_football_view
    except ImportError:
        pass

    try:
        from cogs.web.web import build_image_view
        builders["search_images"] = build_image_view
    except ImportError:
        pass

    return builders


_BUILDERS: dict[str, Builder] | None = None


def build_widget(tool_name: str, data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit un LayoutView à partir du nom d'outil et des données retournées."""
    global _BUILDERS
    if _BUILDERS is None:
        _BUILDERS = _load_builders()
    builder = _BUILDERS.get(tool_name)
    if builder is None:
        return None
    return builder(data, commentary=commentary)
