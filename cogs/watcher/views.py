"""Vues LayoutView pour consulter et valider les suggestions de l'IA passive.

Une seule commande `/suggestions` :
- tout le monde voit ses suggestions personnelles (rappels, profil) ;
- les modérateurs voient en plus les événements suggérés pour le serveur.

Quand une suggestion est incomplète (date manquante…), le bouton « Accepter »
ouvre un modal pour préciser ; nano normalise ensuite les champs saisis.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord

from common.suggestions import (
    KIND_GROUP_ACTIVITY,
    KIND_PERSONAL_REMINDER,
    KIND_PROFILE_UPDATE,
    KIND_SERVER_EVENT,
    Suggestion,
)
from common.timezones import PARIS_TZ

if TYPE_CHECKING:
    from cogs.watcher.watcher import Watcher

# Plafonds d'affichage (limite de composants d'un LayoutView).
_PERSONAL_CAP_SOLO = 6
_SECTION_CAP_MOD = 3

_KIND_HEADERS = {
    KIND_PERSONAL_REMINDER: "◆ Rappel suggéré",
    KIND_PROFILE_UPDATE: "◆ Mise à jour de profil",
    KIND_SERVER_EVENT: "◆ Événement serveur",
    KIND_GROUP_ACTIVITY: "◆ Activité de groupe",
}


def parse_when(when: str) -> Optional[datetime]:
    """Parse une date ISO 8601 (fuseau Paris si naïf) en datetime UTC."""
    when = (when or "").strip()
    if not when:
        return None
    try:
        dt = datetime.fromisoformat(when)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PARIS_TZ)
    return dt.astimezone(timezone.utc)


def has_valid_date(s: Suggestion) -> bool:
    """Vrai si la suggestion porte une date future exploitable."""
    dt = parse_when(s.payload.get("when", ""))
    return dt is not None and dt > datetime.now(timezone.utc)


def _when_label(when: str) -> str:
    dt = parse_when(when)
    if dt is None:
        return "_à préciser_"
    return f"<t:{int(dt.timestamp())}:f> (<t:{int(dt.timestamp())}:R>)"


def _suggestion_text(s: Suggestion) -> str:
    header = _KIND_HEADERS.get(s.kind, "◆ Suggestion")
    p = s.payload
    if s.kind == KIND_PROFILE_UPDATE:
        cat = p.get("category", "perso")
        return f"**{header}** · _{cat}_\n{p.get('info', '')}"
    if s.kind == KIND_PERSONAL_REMINDER:
        rec = p.get("recurrence", "none")
        rec_str = {"daily": " · ↻ quotidien", "weekly": " · ↻ hebdo"}.get(rec, "")
        return f"**{header}**{rec_str}\n{p.get('description', '')}\n-# {_when_label(p.get('when', ''))}"
    # server_event / group_activity
    return f"**{header}**\n{p.get('title', s.description)}\n-# {_when_label(p.get('when', ''))}"


# ---------------------------------------------------------------------------
# Boutons
# ---------------------------------------------------------------------------

class _PersonalActionButton(discord.ui.Button):
    def __init__(self, cog: "Watcher", ctx: "ViewContext", suggestion_id: int, *, accept: bool):
        super().__init__(
            style=discord.ButtonStyle.success if accept else discord.ButtonStyle.secondary,
            label="Accepter" if accept else "Ignorer",
            custom_id=f"sugg_{'ok' if accept else 'no'}_{suggestion_id}_{ctx.user_id}",
        )
        self.cog = cog
        self.ctx = ctx
        self.suggestion_id = suggestion_id
        self.accept = accept

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.ctx.user_id:
            return await interaction.response.send_message(
                "Cette suggestion ne te concerne pas.", ephemeral=True
            )
        await self.cog.handle_personal_action(
            interaction, self.suggestion_id, accept=self.accept, ctx=self.ctx
        )


class _EventActionButton(discord.ui.Button):
    def __init__(self, cog: "Watcher", ctx: "ViewContext", suggestion_id: int, *, accept: bool):
        super().__init__(
            style=discord.ButtonStyle.success if accept else discord.ButtonStyle.secondary,
            label="Accepter" if accept else "Refuser",
            custom_id=f"event_{'ok' if accept else 'no'}_{suggestion_id}",
        )
        self.cog = cog
        self.ctx = ctx
        self.suggestion_id = suggestion_id
        self.accept = accept

    async def callback(self, interaction: discord.Interaction) -> None:
        await self.cog.handle_event_action(
            interaction, self.suggestion_id, accept=self.accept, ctx=self.ctx
        )


# ---------------------------------------------------------------------------
# Modals de complétion (nano remplit les champs depuis le texte libre)
# ---------------------------------------------------------------------------

class ReminderCompleteModal(discord.ui.Modal, title="Préciser le rappel"):
    def __init__(self, cog: "Watcher", ctx: "ViewContext", suggestion_id: int, *, description: str):
        super().__init__()
        self.cog = cog
        self.ctx = ctx
        self.suggestion_id = suggestion_id
        self.desc_input = discord.ui.TextInput(
            label="Quoi te rappeler",
            default=description[:200],
            max_length=200,
            required=True,
        )
        self.when_input = discord.ui.TextInput(
            label="Quand (langage naturel)",
            placeholder="ex : demain 18h, vendredi soir, dans 3 jours",
            max_length=100,
            required=True,
        )
        self.add_item(self.desc_input)
        self.add_item(self.when_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.cog.finish_reminder_from_modal(
            interaction,
            self.suggestion_id,
            description=self.desc_input.value.strip(),
            when_text=self.when_input.value.strip(),
            ctx=self.ctx,
        )


class EventCompleteModal(discord.ui.Modal, title="Créer l'événement"):
    def __init__(self, cog: "Watcher", ctx: "ViewContext", suggestion_id: int, *, name: str, when_raw: str):
        super().__init__()
        self.cog = cog
        self.ctx = ctx
        self.suggestion_id = suggestion_id
        self.name_input = discord.ui.TextInput(
            label="Nom de l'événement",
            default=name[:100],
            max_length=100,
            required=True,
        )
        self.when_input = discord.ui.TextInput(
            label="Quand (langage naturel)",
            default=when_raw[:100],
            placeholder="ex : samedi 20h, le 14/07 à 19h",
            max_length=100,
            required=True,
        )
        self.location_input = discord.ui.TextInput(
            label="Lieu",
            placeholder="ex : en vocal, chez Théo, en ligne",
            max_length=100,
            required=False,
        )
        self.add_item(self.name_input)
        self.add_item(self.when_input)
        self.add_item(self.location_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.cog.finish_event_from_modal(
            interaction,
            self.suggestion_id,
            name=self.name_input.value.strip(),
            when_text=self.when_input.value.strip(),
            location=self.location_input.value.strip(),
            ctx=self.ctx,
        )


# ---------------------------------------------------------------------------
# Vue unifiée
# ---------------------------------------------------------------------------

class ViewContext:
    """Contexte de rendu : qui consulte, dans quel serveur, et s'il est modo."""

    __slots__ = ("user_id", "guild_id", "is_mod")

    def __init__(self, user_id: int, guild_id: int, is_mod: bool):
        self.user_id = user_id
        self.guild_id = guild_id
        self.is_mod = is_mod


def _empty_view(message: str, header: Optional[str] = None) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=180)
    children: list[discord.ui.Item] = []
    if header:
        children.append(discord.ui.TextDisplay(header))
        children.append(discord.ui.Separator())
    children.append(discord.ui.TextDisplay(message))
    view.add_item(discord.ui.Container(*children))
    return view


class SuggestionsView(discord.ui.LayoutView):
    """Vue unique : suggestions perso + (pour les modos) événements serveur."""

    def __init__(self, cog: "Watcher", ctx: ViewContext, header: Optional[str] = None):
        super().__init__(timeout=180)
        personal = cog.store.list_personal(ctx.user_id)
        events = cog.store.list_events(ctx.guild_id) if (ctx.is_mod and ctx.guild_id) else []

        personal_cap = _SECTION_CAP_MOD if (ctx.is_mod and events) else _PERSONAL_CAP_SOLO
        children: list[discord.ui.Item] = []
        if header:
            children.append(discord.ui.TextDisplay(header))
            children.append(discord.ui.Separator())
        children.append(discord.ui.TextDisplay("## Suggestions de MARIA"))
        children.append(discord.ui.Separator())

        children.append(discord.ui.TextDisplay("**Pour toi**"))
        if personal:
            self._add_items(children, cog, ctx, personal, personal_cap, event=False)
        else:
            children.append(discord.ui.TextDisplay("-# Aucune suggestion en attente."))

        if ctx.is_mod and ctx.guild_id:
            children.append(discord.ui.Separator())
            children.append(discord.ui.TextDisplay("**Événements serveur** · ⚙ modérateurs"))
            if events:
                self._add_items(children, cog, ctx, events, _SECTION_CAP_MOD, event=True)
            else:
                children.append(discord.ui.TextDisplay("-# Aucun événement suggéré en attente."))

        self.add_item(discord.ui.Container(*children))

    @staticmethod
    def _add_items(
        children: list, cog: "Watcher", ctx: ViewContext,
        suggestions: list[Suggestion], cap: int, *, event: bool,
    ) -> None:
        for s in suggestions[:cap]:
            children.append(discord.ui.TextDisplay(_suggestion_text(s)))
            if event:
                row = discord.ui.ActionRow(
                    _EventActionButton(cog, ctx, s.id, accept=True),
                    _EventActionButton(cog, ctx, s.id, accept=False),
                )
            else:
                row = discord.ui.ActionRow(
                    _PersonalActionButton(cog, ctx, s.id, accept=True),
                    _PersonalActionButton(cog, ctx, s.id, accept=False),
                )
            children.append(row)
        if len(suggestions) > cap:
            children.append(discord.ui.TextDisplay(f"-# +{len(suggestions) - cap} autre(s) en attente."))


def build_suggestions_view(
    cog: "Watcher", ctx: ViewContext, header: Optional[str] = None
) -> discord.ui.LayoutView:
    has_personal = bool(cog.store.list_personal(ctx.user_id))
    has_events = bool(ctx.is_mod and ctx.guild_id and cog.store.list_events(ctx.guild_id))
    if not has_personal and not has_events:
        return _empty_view(
            "Aucune suggestion en attente — MARIA t'en proposera au fil des conversations dans les salons surveillés.",
            header,
        )
    return SuggestionsView(cog, ctx, header)
