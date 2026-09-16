"""Helpers de construction de vues Discord (components v2) partagés par les cogs."""

from __future__ import annotations

import re
from typing import Optional

import discord

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?!<)(https?://[^)\s]+)\)", re.I)
_BARE_URL_RE = re.compile(r"(?<![<(])https?://[^\s<>]+", re.I)


def md_link(label: str, url: str) -> str:
    """Lien markdown Discord sans aperçu (URL encapsulée dans <>)."""
    return f"[{label}](<{url}>)"


def suppress_link_embeds(text: str) -> str:
    """Empêche l'unfurl : `[texte](<url>)` et URLs nues en `<url>`."""
    if not text:
        return text
    text = _MD_LINK_RE.sub(lambda m: f"[{m.group(1)}](<{m.group(2)}>)", text)

    def _bare(m: re.Match) -> str:
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:!?":
            trail = url[-1] + trail
            url = url[:-1]
        if not url:
            return m.group(0)
        return f"<{url}>{trail}"

    return _BARE_URL_RE.sub(_bare, text)


def layout_with_commentary(
    body: discord.ui.Item,
    commentary: str = "",
) -> discord.ui.LayoutView:
    """Assemble un `LayoutView` : commentaire optionnel en tête, puis le corps.

    `body` est le composant principal à afficher (Container, MediaGallery…).
    """
    view = discord.ui.LayoutView(timeout=None)
    if commentary:
        view.add_item(discord.ui.TextDisplay(suppress_link_embeds(commentary)))
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


def member_accent_colour(user) -> Optional[discord.Colour]:
    """Couleur de rôle du membre, ou None si défaut / pas un membre de serveur."""
    colour = getattr(user, "colour", None)
    if colour is None:
        colour = getattr(user, "color", None)
    value = int(getattr(colour, "value", 0) or 0)
    if not value:
        return None
    return colour if isinstance(colour, discord.Colour) else discord.Colour(value)


def member_accent_value(user) -> Optional[int]:
    colour = member_accent_colour(user)
    return None if colour is None else int(colour.value)
