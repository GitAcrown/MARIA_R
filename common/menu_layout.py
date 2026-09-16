"""Menus LayoutView type CRIT — hub unique, pas de vue recréée à chaque clic.

Calqué sur CRIT `ReviewsLayout` / `apply_view` / `HubTabButton` / `HubPageButton`.
Les widgets lecture seule continuent d'utiliser `discord_ui.layout_with_commentary`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import discord

from common.discord_ui import layout_with_commentary, section_with_thumbnail  # noqa: F401

logger = logging.getLogger("MARIA.Menu")

NO_PINGS = discord.AllowedMentions.none()
MENU_TIMEOUT = 840.0


def sep_tight() -> discord.ui.Separator:
    return discord.ui.Separator(spacing=discord.SeparatorSpacing.small)


def _disable_interactive(item: discord.ui.Item) -> None:
    if getattr(item, "disabled", None) is False:
        item.disabled = True  # type: ignore[attr-defined]
    children = getattr(item, "children", None)
    if children:
        for child in children:
            _disable_interactive(child)


def _remember_session_view(
    interaction: discord.Interaction,
    view: discord.ui.LayoutView,
    message_id: int | None,
) -> None:
    """Après un defer éphémère, l'id webhook peut différer de celui du clic."""
    if view.is_finished() or not view.is_dispatchable():
        return
    store = interaction.client._connection.store_view
    if message_id is not None:
        store(view, message_id)
    store(view, None)


def bind_view_message(
    view: discord.ui.LayoutView,
    message: discord.Message | discord.WebhookMessage | None,
) -> None:
    if message is None:
        return
    view.message = message
    if hasattr(view, "_message"):
        view._message = message


async def apply_view(interaction: discord.Interaction, view: discord.ui.LayoutView) -> None:
    """Met à jour le message qui porte les boutons, pas un autre webhook."""
    kwargs: dict[str, Any] = {"view": view, "allowed_mentions": NO_PINGS}
    message: discord.Message | discord.WebhookMessage | None = None
    if not interaction.response.is_done():
        await interaction.response.edit_message(**kwargs)
        message = interaction.message
    else:
        try:
            message = await interaction.edit_original_response(**kwargs)
        except discord.HTTPException:
            if interaction.message is None:
                raise
            message = await interaction.message.edit(**kwargs)
    bind_view_message(view, message or interaction.message)


async def send_ephemeral_menu(
    interaction: discord.Interaction,
    view: discord.ui.LayoutView,
) -> None:
    """Ouvre un menu perso sans toucher au message public du salon."""
    if not interaction.response.is_done():
        await interaction.response.send_message(
            view=view, ephemeral=True, allowed_mentions=NO_PINGS,
        )
    else:
        await interaction.followup.send(
            view=view, ephemeral=True, allowed_mentions=NO_PINGS,
        )
    try:
        bind_view_message(view, await interaction.original_response())
    except discord.HTTPException:
        bind_view_message(view, interaction.message)
    mid = getattr(view.message, "id", None)
    _remember_session_view(interaction, view, mid)


async def publish_layout_message(
    interaction: discord.Interaction,
    view: discord.ui.LayoutView,
    files: list[discord.File] | None = None,
) -> discord.Message | None:
    channel = interaction.channel
    if channel is None or not isinstance(channel, discord.abc.Messageable):
        return None
    try:
        kwargs: dict[str, Any] = {"view": view, "allowed_mentions": NO_PINGS}
        if files:
            kwargs["files"] = files
        return await channel.send(**kwargs)
    except (AttributeError, discord.HTTPException) as exc:
        logger.warning("Impossible de publier dans le salon : %s", exc)
        return None


class MariaLayout(discord.ui.LayoutView):
    """Base des hubs interactifs : un Container, état objet, `_build()` + `push`."""

    def __init__(
        self,
        *,
        timeout: float | None = MENU_TIMEOUT,
        viewer_id: int | None = None,
        accent_colour: discord.Colour | None = None,
    ):
        super().__init__(timeout=timeout)
        self.viewer_id = viewer_id
        self.accent_colour = accent_colour
        self._interaction: discord.Interaction | None = None
        self._message: discord.WebhookMessage | discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.viewer_id is None or interaction.user.id == self.viewer_id:
            return True
        try:
            await interaction.response.send_message(
                "C'est pas ton menu.", ephemeral=True,
            )
        except discord.HTTPException:
            pass
        return False

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item,
    ) -> None:
        logger.exception("Vue %s / %s : %s", type(self).__name__, type(item).__name__, error)
        try:
            msg = "**Erreur ·** Le bouton a planté. Réessaie."
            if not interaction.response.is_done():
                await interaction.response.send_message(msg, ephemeral=True)
            else:
                await interaction.followup.send(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    async def on_timeout(self) -> None:
        for item in list(self.children):
            _disable_interactive(item)
        message = self.message or self._message
        if message is None:
            return
        try:
            await message.edit(view=self, allowed_mentions=NO_PINGS)
        except discord.HTTPException:
            pass

    async def attach(self, interaction: discord.Interaction) -> None:
        self._interaction = interaction
        try:
            bind_view_message(self, await interaction.original_response())
        except discord.HTTPException:
            bind_view_message(self, interaction.message)

    async def push(self, interaction: discord.Interaction | None = None) -> None:
        try:
            if interaction is not None:
                await apply_view(interaction, self)
                mid = getattr(self.message, "id", None) or (
                    interaction.message.id if interaction.message else None
                )
                _remember_session_view(interaction, self, mid)
                return
            message = self.message or self._message
            if message is not None:
                await message.edit(view=self, allowed_mentions=NO_PINGS)
                return
            if self._interaction is not None:
                await apply_view(self._interaction, self)
        except discord.HTTPException as exc:
            logger.warning("Impossible de rafraîchir %s : %s", type(self).__name__, exc)

    def set_layout(self, body: list[discord.ui.Item], *rows: discord.ui.Item | None) -> None:
        self.clear_items()
        children = list(body)
        for row in rows:
            if row is None:
                continue
            if children:
                children.append(sep_tight())
            children.append(row)
        if children:
            kwargs: dict = {}
            if self.accent_colour is not None:
                kwargs["accent_colour"] = self.accent_colour
            self.add_item(discord.ui.Container(*children, **kwargs))

    def _build(self) -> None:
        """À surcharger : reconstruit le layout depuis l'état de l'instance."""
        raise NotImplementedError


class HubTabButton(discord.ui.Button):
    def __init__(self, parent: MariaLayout, tab: str, label: str, *, emoji: str | None = None):
        current = getattr(parent, "tab", None)
        super().__init__(
            label=label,
            emoji=discord.PartialEmoji.from_str(emoji) if emoji else None,
            style=discord.ButtonStyle.primary if current == tab else discord.ButtonStyle.secondary,
        )
        self._hub = parent
        self._tab = tab

    async def callback(self, interaction: discord.Interaction) -> None:
        self._hub.tab = self._tab
        if hasattr(self._hub, "page"):
            self._hub.page = 0
        self._hub._build()
        await apply_view(interaction, self._hub)


class HubPageButton(discord.ui.Button):
    def __init__(self, parent: MariaLayout, attr: str, delta: int, label: str, max_page: int):
        super().__init__(label=label, style=discord.ButtonStyle.secondary)
        self._hub = parent
        self._attr = attr
        self._delta = delta
        self._max_page = max_page

    async def callback(self, interaction: discord.Interaction) -> None:
        current = getattr(self._hub, self._attr)
        setattr(self._hub, self._attr, max(0, min(self._max_page, current + self._delta)))
        self._hub._build()
        await apply_view(interaction, self._hub)
