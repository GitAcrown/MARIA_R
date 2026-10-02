"""Registre centralisé des builders de widgets pour les outils LLM.

Chaque cog s'enregistre lui-même via `register_widget`, typiquement depuis son
`setup()` (et se désenregistre dans `cog_unload`) — pas d'import `common -> cogs`.

Un outil appelé plusieurs fois dans le même tour peut fournir un `group_builder` :
les résultats sont alors regroupés dans une seule vue à onglets.
"""

from typing import Callable, Optional

import discord

Builder = Callable[..., Optional[discord.ui.LayoutView]]
GroupBuilder = Callable[..., Optional[discord.ui.LayoutView]]

_BUILDERS: dict[str, Builder] = {}
_GROUP_BUILDERS: dict[str, GroupBuilder] = {}


def register_widget(
    tool_name: str,
    builder: Builder,
    *,
    group_builder: Optional[GroupBuilder] = None,
) -> None:
    """Enregistre (ou remplace) le builder de widget pour un nom d'outil LLM."""
    _BUILDERS[tool_name] = builder
    if group_builder is not None:
        _GROUP_BUILDERS[tool_name] = group_builder
    else:
        _GROUP_BUILDERS.pop(tool_name, None)


def unregister_widget(tool_name: str) -> None:
    """Retire un builder (à appeler depuis `cog_unload`)."""
    _BUILDERS.pop(tool_name, None)
    _GROUP_BUILDERS.pop(tool_name, None)


def has_widget(tool_name: str) -> bool:
    """True si un builder de widget est enregistré pour cet outil."""
    return tool_name in _BUILDERS


def build_widget(tool_name: str, data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit un LayoutView à partir du nom d'outil et des données retournées."""
    builder = _BUILDERS.get(tool_name)
    if builder is None:
        return None
    return builder(data, commentary=commentary)


def build_widget_group(
    tool_name: str,
    datas: list[dict],
    commentary: str = "",
) -> Optional[discord.ui.LayoutView]:
    """Une vue pour tous les résultats d'un même outil : onglets si plusieurs et supportés.

    Sans `group_builder`, le premier résultat affichable est utilisé.
    """
    group_builder = _GROUP_BUILDERS.get(tool_name)
    if group_builder is not None and len(datas) > 1:
        view = group_builder(datas, commentary=commentary)
        if view is not None:
            return view
    for data in datas:
        view = build_widget(tool_name, data, commentary=commentary)
        if view is not None:
            return view
    return None
