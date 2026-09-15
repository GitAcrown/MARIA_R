"""Vues /signets — hub unique (liste, recherche, détail, partage)."""

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
from common.emojis import SAVE_SMALL
from common.menu_layout import (
    HubPageButton,
    MENU_TIMEOUT,
    MariaLayout,
    apply_view,
    publish_layout_message,
)
from common.timezones import PARIS_TZ
from common.widget_catalog import render_free_widget

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


def _deny(interaction: discord.Interaction, user_id: int) -> Optional[str]:
    if interaction.user.id != user_id:
        return "C'est pas tes signets."
    return None


class BookmarksView(MariaLayout):
    def __init__(
        self,
        user_id: int,
        items: list[Bookmark],
        *,
        query: str = "",
        page: int = 0,
        note: str = "",
        shared_ids: Optional[set[str]] = None,
    ):
        super().__init__(timeout=MENU_TIMEOUT, viewer_id=user_id)
        self.user_id = user_id
        self.items = items
        self.query = query
        self.page = page
        self.note = note
        self.shared_ids = shared_ids if shared_ids is not None else set()
        self.screen = "catalog"
        self.selected: Optional[Bookmark] = None
        self._build()

    def _refresh_items(self) -> None:
        self.items = search_for_user(self.user_id, self.query) if self.query else list_for_user(self.user_id)

    async def reload(self, interaction: discord.Interaction, note: str = "") -> None:
        self._refresh_items()
        self.screen = "catalog"
        self.selected = None
        self.note = note
        self._build()
        await self.push(interaction)

    def _build_catalog(self) -> None:
        pages = _pages(self.items)
        self.page = max(0, min(self.page, len(pages) - 1))
        shown = pages[self.page]
        subtitle = "-# Fiches enregistrées"
        if self.query:
            subtitle += f" · filtre « {_clip(self.query, 40)} »"
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {SAVE_SMALL} Signets · {len(self.items)}/{BOOKMARK_MAX}"),
            discord.ui.TextDisplay(subtitle),
        ]
        if not self.items:
            body.append(discord.ui.TextDisplay(
                "-# Aucun résultat." if self.query else "-# Aucun signet pour l'instant."
            ))
        else:
            lines = [f"**{_clip(bm.title, 80)}**\n-# {_stamp(bm.created_at)}" for bm in shown]
            body += [
                discord.ui.TextDisplay("\n\n".join(lines)),
                discord.ui.TextDisplay(f"-# Page {self.page + 1}/{len(pages)}"),
            ]
        rows: list[discord.ui.ActionRow] = []
        if shown:
            rows.append(discord.ui.ActionRow(_PickBookmarkSelect(self, shown)))
        actions: list[discord.ui.Button] = [_SearchBookmarkButton(self)]
        if self.query:
            actions.append(_ClearSearchButton(self))
        max_page = max(0, len(pages) - 1)
        if max_page > 0:
            if self.page > 0:
                actions.append(HubPageButton(self, "page", -1, "Precedent", max_page))
            if self.page < max_page:
                actions.append(HubPageButton(self, "page", 1, "Suivant", max_page))
        rows.append(discord.ui.ActionRow(*actions[:5]))
        if self.note:
            body.append(discord.ui.TextDisplay(f"-# {self.note}"))
        self.set_layout(body, *rows)

    def _build_detail(self) -> None:
        bm = self.selected
        self.clear_items()
        if bm is None:
            self.screen = "catalog"
            self._build()
            return
        built = render_free_widget(bm.spec, commentary="")
        if built is not None:
            for item in list(built.children):
                built.remove_item(item)
                self.add_item(item)
        else:
            self.add_item(discord.ui.TextDisplay(f"## {bm.title}"))
            self.add_item(discord.ui.TextDisplay("-# Fiche illisible."))
        self.add_item(discord.ui.ActionRow(
            _SendBookmarkButton(self, bm),
            _DeleteBookmarkButton(self, bm),
            _BackBookmarksButton(self),
        ))

    def _build(self) -> None:
        if self.screen == "detail":
            self._build_detail()
            return
        self._build_catalog()


BookmarkDetailView = BookmarksView


class _PickBookmarkSelect(discord.ui.Select):
    def __init__(self, hub: BookmarksView, items: list[Bookmark]):
        options = [
            discord.SelectOption(
                label=_clip(bm.title, 100) or "Fiche",
                value=bm.id,
                description=_stamp_plain(bm.created_at)[:100],
            )
            for bm in items[:25]
        ]
        super().__init__(placeholder="Ouvrir une fiche", min_values=1, max_values=1, options=options)
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        bid = (self.values or [None])[0]
        bm = get_bookmark(bid, self._hub.user_id) if bid else None
        if bm is None:
            await self._hub.reload(interaction, note="Introuvable.")
            return
        self._hub.selected = bm
        self._hub.screen = "detail"
        self._hub._build()
        await apply_view(interaction, self._hub)


class _SearchBookmarkButton(discord.ui.Button):
    def __init__(self, hub: BookmarksView):
        super().__init__(style=discord.ButtonStyle.secondary, label="Rechercher")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(
            SearchBookmarkModal(self._hub),
        )


class SearchBookmarkModal(discord.ui.Modal, title="Rechercher un signet"):
    def __init__(self, hub: BookmarksView):
        super().__init__()
        self._hub = hub
        self.query = discord.ui.TextInput(
            label="Recherche",
            placeholder="recette, comparatif, un mot de la fiche…",
            required=True,
            max_length=80,
            default=(hub.query or "")[:80],
        )
        self.add_item(self.query)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.query = (self.query.value or "").strip()
        self._hub.page = 0
        await self._hub.reload(interaction)


class _ClearSearchButton(discord.ui.Button):
    def __init__(self, hub: BookmarksView):
        super().__init__(style=discord.ButtonStyle.secondary, label="Tout afficher")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.query = ""
        self._hub.page = 0
        await self._hub.reload(interaction)


class _BackBookmarksButton(discord.ui.Button):
    def __init__(self, hub: BookmarksView):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.screen = "catalog"
        self._hub.selected = None
        self._hub._refresh_items()
        self._hub._build()
        await apply_view(interaction, self._hub)


class _DeleteBookmarkButton(discord.ui.Button):
    def __init__(self, hub: BookmarksView, bm: Bookmark):
        super().__init__(style=discord.ButtonStyle.danger, label="Supprimer")
        self._hub = hub
        self.bm = bm

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        ok = delete_bookmark(self.bm.id, self._hub.user_id)
        await self._hub.reload(
            interaction,
            note="Signet supprimé." if ok else "Déjà plus là.",
        )


class _SendBookmarkButton(discord.ui.Button):
    def __init__(self, hub: BookmarksView, bm: Bookmark):
        already = bm.id in hub.shared_ids
        super().__init__(
            style=discord.ButtonStyle.secondary if already else discord.ButtonStyle.primary,
            label="Partager",
            disabled=already,
        )
        self._hub = hub
        self.bm = bm

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        if self.bm.id in self._hub.shared_ids:
            self._hub._build()
            await apply_view(interaction, self._hub)
            return
        view = render_free_widget(self.bm.spec, commentary="")
        if view is None:
            return await interaction.response.send_message("Fiche illisible.", ephemeral=True)
        posted = await publish_layout_message(interaction, view)
        if posted is None:
            return await interaction.response.send_message(
                "Pas de salon où le partager.", ephemeral=True,
            )
        self._hub.shared_ids.add(self.bm.id)
        self._hub._build()
        await apply_view(interaction, self._hub)
