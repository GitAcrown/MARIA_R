"""Helpers de construction de vues Discord (components v2) partagés par les cogs."""

from typing import Optional

import discord


def layout_with_commentary(
    body: discord.ui.Item,
    commentary: str = "",
) -> discord.ui.LayoutView:
    """Assemble un `LayoutView` : commentaire optionnel en tête, puis le corps.

    `body` est le composant principal à afficher (Container, MediaGallery…).
    """
    view = discord.ui.LayoutView(timeout=None)
    if commentary:
        view.add_item(discord.ui.TextDisplay(commentary))
        view.add_item(discord.ui.Separator())
    view.add_item(body)
    return view


def section_with_thumbnail(body: discord.ui.Item, url: Optional[str]):
    """Retourne une `Section` avec vignette, ou `body` seul si l'URL manque/échoue.

    Évite la répétition du try/except autour de `Thumbnail`/`UnfurledMediaItem`
    présente dans plusieurs cogs (météo, films, jeux, foot).
    """
    if not url:
        return body
    try:
        thumb = discord.ui.Thumbnail(url)
        return discord.ui.Section(body, accessory=thumb)
    except Exception:
        return body
