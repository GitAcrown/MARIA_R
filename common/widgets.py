"""Registre centralisé des builders de widgets pour les outils LLM.

Chaque cog s'enregistre lui-même via `register_widget`, typiquement depuis son
`setup()` (et se désenregistre dans `cog_unload`) — pas d'import `common -> cogs`.
"""

from typing import Callable, Optional

import discord

Builder = Callable[..., Optional[discord.ui.LayoutView]]

_BUILDERS: dict[str, Builder] = {}


def register_widget(tool_name: str, builder: Builder) -> None:
    """Enregistre (ou remplace) le builder de widget pour un nom d'outil LLM."""
    _BUILDERS[tool_name] = builder


def unregister_widget(tool_name: str) -> None:
    """Retire un builder (à appeler depuis `cog_unload`)."""
    _BUILDERS.pop(tool_name, None)


def build_widget(tool_name: str, data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit un LayoutView à partir du nom d'outil et des données retournées."""
    builder = _BUILDERS.get(tool_name)
    if builder is None:
        return None
    return builder(data, commentary=commentary)
