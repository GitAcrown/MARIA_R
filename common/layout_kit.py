"""Briques de mise en page LayoutView communes (reprises de CRIT)."""

from __future__ import annotations

from typing import Optional

import discord

TEXT_DISPLAY_MAX = 4000


def sep_tight() -> discord.ui.Separator:
    """Séparateur compact, entre deux blocs de contenu ou avant les contrôles."""
    return discord.ui.Separator(spacing=discord.SeparatorSpacing.small)


def sep_wide() -> discord.ui.Separator:
    """Séparateur large, sous un titre."""
    return discord.ui.Separator(spacing=discord.SeparatorSpacing.large)


def _is_separator(item: discord.ui.Item) -> bool:
    return isinstance(item, discord.ui.Separator)


def _is_control_row(item: discord.ui.Item) -> bool:
    return isinstance(item, discord.ui.ActionRow)


def with_control_separators(items: list[discord.ui.Item]) -> list[discord.ui.Item]:
    """Un trait entre le contenu et les boutons/selects, jamais entre deux rangées de contrôles."""
    out: list[discord.ui.Item] = []
    last_control: Optional[bool] = None
    for item in items:
        if _is_separator(item):
            if out and _is_separator(out[-1]):
                continue
            out.append(item)
            last_control = None
            continue
        is_control = _is_control_row(item)
        if out and last_control is not None and last_control != is_control and not _is_separator(out[-1]):
            out.append(sep_tight())
        out.append(item)
        last_control = is_control
    return out


def title_text(title: str, meta: str = "") -> discord.ui.TextDisplay:
    """Titre `##` et ligne méta `-#` dans le même bloc."""
    text = f"## {title}"
    if meta:
        text += f"\n-# {meta}"
    return discord.ui.TextDisplay(text)


def chunk_text_displays(lines: list[str], *, limit: int = TEXT_DISPLAY_MAX) -> list[discord.ui.TextDisplay]:
    """Regroupe des lignes en TextDisplay sans dépasser la limite Discord."""
    chunks: list[str] = []
    current = ""
    for line in lines:
        piece = line if len(line) <= limit else line[: limit - 1] + "…"
        addition = f"{current}\n{piece}" if current else piece
        if current and len(addition) > limit:
            chunks.append(current)
            current = piece
        else:
            current = addition
    if current:
        chunks.append(current)
    return [discord.ui.TextDisplay(chunk) for chunk in chunks]


def card(
    children: list[discord.ui.Item],
    *,
    accent_colour: Optional[discord.Colour] = None,
    sticky_head: int = 0,
) -> discord.ui.Container:
    """Container avec séparateurs de contrôles normalisés."""
    head = children[:sticky_head]
    rest = with_control_separators(children[sticky_head:])
    kwargs: dict = {}
    if accent_colour is not None:
        kwargs["accent_colour"] = accent_colour
    return discord.ui.Container(*head, *rest, **kwargs)
