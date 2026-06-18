"""Vues LayoutView pour consulter et valider les suggestions de l'IA passive.

`/suggestions` affiche deux sections distinctes :
- Rappels : suggestions de type personal_reminder
- Profil  : suggestions de type profile_update

Quand un rappel est incomplet (date manquante), « Accepter » ouvre un modal ;
nano normalise ensuite la date saisie en ISO 8601.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord

from common.suggestions import (
    KIND_PERSONAL_REMINDER,
    KIND_PROFILE_UPDATE,
    Suggestion,
)
from common.timezones import PARIS_TZ

if TYPE_CHECKING:
    from cogs.watcher.watcher import Watcher

# Nombre max de suggestions affichées par section (limites de composants Discord).
_CAP_REMINDERS = 4
_CAP_PROFILES  = 4


# ---------------------------------------------------------------------------
# Helpers de formatage
# ---------------------------------------------------------------------------

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


def _reminder_text(s: Suggestion) -> str:
    p = s.payload
    rec = p.get("recurrence", "none")
    rec_str = {"daily": "↻ quotidien · ", "weekly": "↻ hebdo · "}.get(rec, "")
    return (
        f"> {rec_str}{p.get('description', '')}\n"
        f"> -# {_when_label(p.get('when', ''))}"
    )


def _profile_text(s: Suggestion) -> str:
    p = s.payload
    cat = p.get("category", "perso")
    return f"> _{cat}_ · {p.get('info', '')}"


# ---------------------------------------------------------------------------
# Bouton d'action
# ---------------------------------------------------------------------------

class _ActionButton(discord.ui.Button):
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


# ---------------------------------------------------------------------------
# Modal de complétion (rappel sans date)
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


# ---------------------------------------------------------------------------
# Contexte de rendu
# ---------------------------------------------------------------------------

class ViewContext:
    """Qui consulte la vue et depuis quel serveur."""

    __slots__ = ("user_id", "guild_id")

    def __init__(self, user_id: int, guild_id: int):
        self.user_id = user_id
        self.guild_id = guild_id


# ---------------------------------------------------------------------------
# Vue principale
# ---------------------------------------------------------------------------

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
    """Deux sections : Rappels puis Profil."""

    def __init__(self, cog: "Watcher", ctx: ViewContext, header: Optional[str] = None):
        super().__init__(timeout=180)
        reminders = cog.store.list_reminders(ctx.user_id)
        profiles  = cog.store.list_profiles(ctx.user_id)

        children: list[discord.ui.Item] = []
        if header:
            children.append(discord.ui.TextDisplay(header))
            children.append(discord.ui.Separator())

        children.append(discord.ui.TextDisplay("## Suggestions de MARIA"))
        children.append(discord.ui.Separator())

        # --- Section Rappels ---
        children.append(discord.ui.TextDisplay("**Rappels**"))
        if reminders:
            _append_suggestions(children, cog, ctx, reminders, _CAP_REMINDERS, _reminder_text)
        else:
            children.append(discord.ui.TextDisplay("-# Aucun rappel suggéré."))

        children.append(discord.ui.Separator())

        # --- Section Profil ---
        children.append(discord.ui.TextDisplay("**Profil**"))
        if profiles:
            _append_suggestions(children, cog, ctx, profiles, _CAP_PROFILES, _profile_text)
        else:
            children.append(discord.ui.TextDisplay("-# Aucune mise à jour de profil suggérée."))

        self.add_item(discord.ui.Container(*children))


def _append_suggestions(
    children: list,
    cog: "Watcher",
    ctx: ViewContext,
    suggestions: list[Suggestion],
    cap: int,
    text_fn,
) -> None:
    for s in suggestions[:cap]:
        children.append(discord.ui.TextDisplay(text_fn(s)))
        children.append(discord.ui.ActionRow(
            _ActionButton(cog, ctx, s.id, accept=True),
            _ActionButton(cog, ctx, s.id, accept=False),
        ))
    if len(suggestions) > cap:
        children.append(discord.ui.TextDisplay(f"-# +{len(suggestions) - cap} autre(s) en attente."))


def build_suggestions_view(
    cog: "Watcher", ctx: ViewContext, header: Optional[str] = None
) -> discord.ui.LayoutView:
    has_reminders = bool(cog.store.list_reminders(ctx.user_id))
    has_profiles  = bool(cog.store.list_profiles(ctx.user_id))
    if not has_reminders and not has_profiles:
        return _empty_view(
            "Aucune suggestion en attente — MARIA t'en proposera au fil des conversations dans les salons surveillés.",
            header,
        )
    return SuggestionsView(cog, ctx, header)
