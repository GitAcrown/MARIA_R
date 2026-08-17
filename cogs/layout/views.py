"""Vues /bookmarks — liste, recherche, ouverture, suppression."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import discord

from common.bookmarks import (
    BOOKMARK_MAX,
    Bookmark,
    delete_bookmark,
    get_bookmark,
    list_for_user,
    search_for_user,
)
from common.emojis import BOOKMARK
from common.timezones import PARIS_TZ
from common.widget_catalog import render_free_widget

_VIEW_TIMEOUT = 300
_PAGE = 5


def _clip(text: str, n: int) -> str:
    raw = (text or "").strip().replace("\n", " ")
    if len(raw) <= n:
        return raw
    return raw[: n - 1] + "…"


def _pages(items: list[Bookmark]) -> list[list[Bookmark]]:
    if not items:
        return [[]]
    return [items[i:i + _PAGE] for i in range(0, len(items), _PAGE)]


def _stamp(dt: datetime) -> str:
    return f"<t:{int(dt.timestamp())}:R>"


def _stamp_plain(dt: datetime) -> str:
    return dt.astimezone(PARIS_TZ).strftime("%d/%m %H:%M")


class BookmarksView(discord.ui.LayoutView):
    def __init__(
        self,
        user_id: int,
        items: list[Bookmark],
        *,
        query: str = "",
        page: int = 0,
        note: str = "",
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        self.user_id = user_id
        pages = _pages(items)
        page = max(0, min(page, len(pages) - 1))
        shown = pages[page]
        total = len(items)

        subtitle = "-# Layouts enregistrés depuis le bouton sous un widget"
        if query:
            subtitle += f" · filtre « {_clip(query, 40)} »"

        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {BOOKMARK} Favoris · {total}/{BOOKMARK_MAX}"),
            discord.ui.TextDisplay(subtitle),
        ]
        if not items:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay(
                    "-# Aucun résultat." if query else "-# Aucun favori pour l'instant."
                ),
            ]
        else:
            lines = []
            for bm in shown:
                lines.append(f"**{_clip(bm.title, 80)}**\n-# {_stamp(bm.created_at)}")
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("\n\n".join(lines)),
                discord.ui.TextDisplay(f"-# Page {page + 1}/{len(pages)}"),
            ]

        rows: list[discord.ui.ActionRow] = []
        if shown:
            rows.append(discord.ui.ActionRow(_PickBookmarkSelect(user_id, shown, query=query, page=page)))
        actions: list[discord.ui.Button] = [
            _SearchBookmarkButton(user_id, query=query, page=page),
        ]
        if query:
            actions.append(_ClearSearchButton(user_id))
        if len(pages) > 1:
            if page > 0:
                actions.append(_BmPageButton("Precedent", user_id, items, query, page - 1))
            if page < len(pages) - 1:
                actions.append(_BmPageButton("Suivant", user_id, items, query, page + 1))
        rows.append(discord.ui.ActionRow(*actions[:5]))
        if note:
            children += [discord.ui.Separator(), discord.ui.TextDisplay(f"-# {note}")]
        children.append(discord.ui.Separator())
        for row in rows:
            children.append(row)
        self.add_item(discord.ui.Container(*children))


class BookmarkDetailView(discord.ui.LayoutView):
    def __init__(self, user_id: int, bm: Bookmark, *, query: str = "", page: int = 0):
        built = render_free_widget(bm.spec, commentary="")
        super().__init__(timeout=_VIEW_TIMEOUT)
        if built is not None:
            for item in list(built.children):
                built.remove_item(item)
                self.add_item(item)
        else:
            self.add_item(discord.ui.TextDisplay(f"## {bm.title}"))
            self.add_item(discord.ui.TextDisplay("-# Layout illisible."))
        self.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.small))
        self.add_item(discord.ui.ActionRow(
            _SendBookmarkButton(user_id, bm),
            _DeleteBookmarkButton(user_id, bm, query=query, page=page),
            _BackBookmarksButton(user_id, query=query, page=page),
        ))


def _reload(user_id: int, *, query: str = "", page: int = 0, note: str = "") -> BookmarksView:
    items = search_for_user(user_id, query) if query else list_for_user(user_id)
    return BookmarksView(user_id, items, query=query, page=page, note=note)


def _deny(interaction: discord.Interaction, user_id: int) -> Optional[str]:
    if interaction.user.id != user_id:
        return "C'est pas tes favoris."
    return None


class _PickBookmarkSelect(discord.ui.Select):
    def __init__(self, user_id: int, items: list[Bookmark], *, query: str, page: int):
        options = [
            discord.SelectOption(
                label=_clip(bm.title, 100) or "Layout",
                value=bm.id,
                description=_stamp_plain(bm.created_at)[:100],
            )
            for bm in items[:25]
        ]
        super().__init__(placeholder="Ouvrir un layout", min_values=1, max_values=1, options=options)
        self.user_id = user_id
        self.query = query
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        bid = (self.values or [None])[0]
        bm = get_bookmark(bid, self.user_id) if bid else None
        if bm is None:
            return await interaction.response.edit_message(
                view=_reload(self.user_id, query=self.query, page=self.page, note="Introuvable."),
            )
        await interaction.response.edit_message(
            view=BookmarkDetailView(self.user_id, bm, query=self.query, page=self.page),
        )


class _SearchBookmarkButton(discord.ui.Button):
    def __init__(self, user_id: int, *, query: str, page: int):
        super().__init__(style=discord.ButtonStyle.secondary, label="Rechercher")
        self.user_id = user_id
        self.query = query
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(
            SearchBookmarkModal(self.user_id, current=self.query),
        )


class SearchBookmarkModal(discord.ui.Modal, title="Rechercher un favori"):
    def __init__(self, user_id: int, *, current: str = ""):
        super().__init__()
        self.user_id = user_id
        self.query = discord.ui.TextInput(
            label="Recherche",
            placeholder="recette, comparatif, un mot du layout…",
            required=True,
            max_length=80,
            default=(current or "")[:80],
        )
        self.add_item(self.query)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        q = (self.query.value or "").strip()
        await interaction.response.edit_message(view=_reload(self.user_id, query=q, page=0))


class _ClearSearchButton(discord.ui.Button):
    def __init__(self, user_id: int):
        super().__init__(style=discord.ButtonStyle.secondary, label="Tout afficher")
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(view=_reload(self.user_id))


class _BmPageButton(discord.ui.Button):
    def __init__(self, label: str, user_id: int, items: list[Bookmark], query: str, page: int):
        super().__init__(style=discord.ButtonStyle.secondary, label=label)
        self.user_id = user_id
        self.items = items
        self.query = query
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=BookmarksView(self.user_id, self.items, query=self.query, page=self.page),
        )


class _BackBookmarksButton(discord.ui.Button):
    def __init__(self, user_id: int, *, query: str, page: int):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self.user_id = user_id
        self.query = query
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=_reload(self.user_id, query=self.query, page=self.page),
        )


class _DeleteBookmarkButton(discord.ui.Button):
    def __init__(self, user_id: int, bm: Bookmark, *, query: str, page: int):
        super().__init__(style=discord.ButtonStyle.danger, label="Supprimer")
        self.user_id = user_id
        self.bm = bm
        self.query = query
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        ok = delete_bookmark(self.bm.id, self.user_id)
        note = "Favori supprimé." if ok else "Déjà plus là."
        await interaction.response.edit_message(
            view=_reload(self.user_id, query=self.query, page=self.page, note=note),
        )


class _SendBookmarkButton(discord.ui.Button):
    def __init__(self, user_id: int, bm: Bookmark):
        super().__init__(style=discord.ButtonStyle.primary, label="Envoyer")
        self.user_id = user_id
        self.bm = bm

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        view = render_free_widget(self.bm.spec, commentary="")
        if view is None:
            return await interaction.response.send_message(
                "Layout illisible.", ephemeral=True,
            )
        channel = interaction.channel
        if channel is None or not isinstance(channel, discord.abc.Messageable):
            return await interaction.response.send_message(
                "Pas de salon où l'envoyer.", ephemeral=True,
            )
        try:
            await channel.send(view=view)
        except discord.HTTPException:
            return await interaction.response.send_message(
                "Impossible d'envoyer ici.", ephemeral=True,
            )
        await interaction.response.send_message("Envoyé dans le salon.", ephemeral=True)
