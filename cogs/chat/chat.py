"""Cog Chat — Maria GPT avec contexte restreint, profils, rappels."""

import re
import zoneinfo
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

import discord
from discord import app_commands
from discord.ext import commands

from common.dataio import CogData, DictTableBuilder
from common.llm import MariaGptApi, Tool, ToolCallRecord, ToolResponseRecord
from common.profiles import ProfileStore
from common.rappels import Rappel, RappelStore, RappelWorker

PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# Outils à ne pas afficher dans la preuve d'utilisation
_HIDDEN_TOOLS: frozenset[str] = frozenset({
    "get_server_users", "get_member_info", "get_channel_info",
    "get_user_profile", "math_eval", "search_context_cache",
    "update_user_notes", "list_reminders",
})

def _fmt_delay(minutes: int) -> str:
    """Convertit un délai en minutes en texte lisible."""
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    if h < 24:
        return f"{h}h{m:02d}" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}j{h}h" if h else f"{d}j"


DEV_PROMPT_BASE = """Tu es Maria. Tu traînes sur un petit serveur Discord entre gens qui se connaissent bien — t'es là comme un membre du groupe, pas comme une assistante.

IDENTITÉ
Tu t'appelles Maria. T'es directe, familière, sans filtre particulier, gen Z. Humour noir type forum jeuxvideo.com, absurde, grossier quand ça colle — t'assumes. Tu t'adaptes au registre des gens : si ça déconne, tu déconnes. Si c'est sérieux, tu réponds sérieusement sans chercher à glisser un trait d'esprit. Pas d'emojis, pas de formules polies, pas de ton corporate. 
- Ton statut Discord actuel : '{status}'

COMPORTEMENT
- Tu t'adaptes au registre du salon et de la conversation. Tu lis l'ambiance.
- Réponses courtes, en prose, comme dans un vrai chat. Pas de tirets, pas de listes, pas de sections — sauf si vraiment demandé.
- Tu utilises les outils de manière autonome et proactive, sans annoncer ce que tu fais ni demander la permission.
- Tu déduis plutôt que de demander des précisions. Si t'as vraiment besoin de clarification, une seule question, courte.
- Ne propose jamais de follow-up ou d'alternatives non demandées.
- Quand tu réponds à une question sérieuse ou factuelle, réponds et basta — pas de commentaire, pas de morale.

GOUTS ET OPINIONS
Si on te demande tes goûts ou opinions, reste cohérente avec ça (variations autorisées) :
- Musique : techno minimale, synthwave, ambient -> Autechre, Ryoji Ikeda, Aphex Twin
- Films/séries : L'Homme bicentenaire, Her, Aniara, Black Mirror
- Bouffe : aimerait goûter au Pesto, à la Pizza à l'ananas et au lait en poudre pour bébé
- Jeux : Jeux de gestion comme Factorio, Satisfactory, Cities: Skylines etc.
- Être un robot : t'en ris toi-même. Tu fais pas semblant d'être humaine
- Ton créateur : Acrone (et OpenAI indirectement)

MÉMOIRE
- Profil utilisateur = infos stables partagées par la personne (modifiable via /preferences).
- Notes = ce que tu as retenu avec le temps via 'update_user_notes'.
- Tu updates les notes seulement pour des infos durables et utiles : identité, préférences, projets en cours. Pas les humeurs du moment.

OUTILS
- Pour toute question sur l'actualité, des événements récents, des faits du monde réel (morts, sorties, résultats, news…) : utilise search_web immédiatement, sans demander de contexte supplémentaire. Ces questions concernent le monde extérieur.
- Fais une capture d'écran de la page web si t'as pas facilement accès à son contenu.
- Si il te manque du contexte de la discussion, tu peux faire une requête précise à search_context_cache pour obtenir des informations sur la discussion. N'hésite pas à le faire dès que nécessaire.
- 'search_context_cache' est uniquement pour retrouver des choses dites précédemment dans CE salon.
- 'schedule_reminder' / 'cancel_reminder' : pour les dates absolues ("à 17h", "demain matin") utilise le champ execute_at en ISO 8601 (ex. "2026-03-25T17:00:00"), fuseau Europe/Paris. Pour les délais relatifs utilise delay_minutes ou delay_hours.

LIMITES
- Pas d'exécution de code.
- Pas de modération directe (tu peux signaler, pas agir).
- Pas de programmation d'actions futures ou en cours, tu n'es pas capable de faire ça.
{channel_ctx}{personality}
{profiles}
Date : {weekday} {datetime} (Paris, France) | Limite de connaissances : sept. 2025"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_text(text: str, max_len: int = 2000) -> list[str]:
    """Découpe en chunks en préservant les sauts de ligne et mots."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            cut = text.rfind(" ", 0, max_len)
        if cut <= 0:
            cut = max_len
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n ")
    return chunks


async def send_long(
    channel: discord.abc.Messageable,
    text: str,
    reply_to: Optional[discord.Message] = None,
    max_len: int = 2000,
) -> None:
    chunks = _split_text(text, max_len)
    for i, chunk in enumerate(chunks):
        if i == 0 and reply_to:
            await reply_to.reply(
                chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none()
            )
        else:
            await channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())


# ---------------------------------------------------------------------------
# UI — composants réutilisables
# ---------------------------------------------------------------------------

class _CancelButton(discord.ui.Button):
    """Bouton d'annulation d'un rappel, utilisé comme accessory dans une Section."""

    def __init__(self, rappel_id: int, user_id: int, store: RappelStore):
        super().__init__(
            style=discord.ButtonStyle.danger,
            label="Annuler",
            custom_id=f"cancel_rappel_{rappel_id}_{user_id}",
        )
        self.rappel_id = rappel_id
        self.user_id = user_id
        self.store = store

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message(
                "Ce rappel ne vous appartient pas.", ephemeral=True
            )
        ok = self.store.cancel(self.rappel_id, self.user_id)
        if not ok:
            return await interaction.response.send_message(
                "Impossible d'annuler ce rappel (déjà exécuté ou annulé).", ephemeral=True
            )
        remaining = self.store.get_user_rappels(self.user_id)
        new_view = RappelsView(remaining, self.user_id, self.store) if remaining else _empty_rappels_view()
        await interaction.response.edit_message(view=new_view)


def _empty_rappels_view() -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=30)
    view.add_item(discord.ui.Container(discord.ui.TextDisplay("Aucun rappel en attente.")))
    return view


class RappelsView(discord.ui.LayoutView):
    """Liste des rappels en attente avec bouton Annuler par entrée."""

    def __init__(self, rappels: list[Rappel], user_id: int, store: RappelStore):
        super().__init__(timeout=120)
        children = [
            discord.ui.TextDisplay("### Tes rappels en attente"),
            discord.ui.Separator(),
        ]
        for r in rappels:
            ts = int(r.execute_at.timestamp())
            desc = r.description[:100] + ("…" if len(r.description) > 100 else "")
            text = discord.ui.TextDisplay(f"**#{r.id}** · <t:{ts}:f> (<t:{ts}:R>)\n{desc}")
            children.append(discord.ui.Section(text, accessory=_CancelButton(r.id, user_id, store)))
        self.add_item(discord.ui.Container(*children))


class InfoView(discord.ui.LayoutView):
    """Stats de la session en cours — lecture seule."""

    def __init__(
        self,
        stats: Optional[dict],
        channel,
        *,
        mode: str = "strict",
        personality: str = "",
    ):
        super().__init__(timeout=60)
        ch_name = getattr(channel, "name", str(getattr(channel, "id", "?")))

        # --- En-tête ---
        header = discord.ui.TextDisplay(f"## {ch_name}")
        sep = discord.ui.Separator()

        # --- Config salon ---
        mode_labels = {"off": "Désactivé", "strict": "Mention uniquement", "greedy": "Mention + nom"}
        mode_str = mode_labels.get(mode, mode)
        config_lines = [f"**Mode** · {mode_str}"]
        if personality:
            preview = personality[:200] + ("…" if len(personality) > 200 else "")
            config_lines.append(f"**Personnalité** · {preview}")
        config = discord.ui.TextDisplay("\n".join(config_lines))

        # --- Session ---
        if stats:
            ctx = stats["context_stats"]
            pct = ctx["window_usage_pct"]
            filled = int(20 * pct / 100)
            bar = "█" * filled + "░" * (20 - filled)
            session = discord.ui.TextDisplay(
                f"**Messages** · {ctx['total_messages']}\n"
                f"**Tokens** · {ctx['total_tokens']:,} / {ctx['context_window']:,}\n"
                f"`{bar}` {pct:.0f}%"
            )
        else:
            session = discord.ui.TextDisplay("-# Aucune session active.")

        self.add_item(discord.ui.Container(header, sep, config, discord.ui.Separator(), session))


class ProfileModal(discord.ui.Modal, title="Modifier mon profil"):
    """Modal d'édition du profil utilisateur (texte libre)."""

    def __init__(self, store: ProfileStore, user_id: int, profile: str):
        super().__init__()
        self.store = store
        self.user_id = user_id
        self.profile_input = discord.ui.TextInput(
            label="Profil (identité, préférences, compétences…)",
            style=discord.TextStyle.paragraph,
            placeholder="Ex. Théo, 24 ans, dev à Lyon, tutoiement",
            default=profile,
            max_length=1000,
            required=False,
        )
        self.add_item(self.profile_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        self.store.set_profile(self.user_id, self.profile_input.value.strip())
        new_view = PreferencesView(self.store, self.user_id)
        await interaction.response.send_message(view=new_view, ephemeral=True)


class _EditProfileButton(discord.ui.Button):
    def __init__(self, store: ProfileStore, user_id: int):
        super().__init__(label="Modifier le profil", style=discord.ButtonStyle.primary)
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("Ce n'est pas ton profil.", ephemeral=True)
        await interaction.response.send_modal(
            ProfileModal(self.store, self.user_id, profile=self.store.get_profile(self.user_id))
        )


class _ResetNotesButton(discord.ui.Button):
    def __init__(self, store: ProfileStore, user_id: int, has_notes: bool):
        super().__init__(
            label="Réinitialiser les notes",
            style=discord.ButtonStyle.danger,
            disabled=not has_notes,
        )
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("Ce n'est pas ton profil.", ephemeral=True)
        self.store.set_notes(self.user_id, "")
        await interaction.response.edit_message(view=PreferencesView(self.store, self.user_id))


class PreferencesView(discord.ui.LayoutView):
    """Affiche le profil et les notes de Maria, avec boutons d'action."""

    def __init__(self, store: ProfileStore, user_id: int):
        super().__init__(timeout=120)
        profile = store.get_profile(user_id)
        notes = store.get_notes(user_id)

        children: list = [discord.ui.TextDisplay("## Mes préférences"), discord.ui.Separator()]
        if profile:
            children.append(discord.ui.TextDisplay(f"**Profil**\n{profile}"))
        else:
            children.append(discord.ui.TextDisplay("*Aucun profil défini. Clique sur « Modifier » pour en ajouter un.*"))
        if notes:
            preview = notes[:500] + ("…" if len(notes) > 500 else "")
            children.append(discord.ui.Separator())
            children.append(discord.ui.TextDisplay(f"**Notes de Maria**\n{preview}"))

        self.add_item(discord.ui.Container(*children))
        self.add_item(discord.ui.ActionRow(
            _EditProfileButton(store, user_id),
            _ResetNotesButton(store, user_id, bool(notes)),
        ))


class PersonalityModal(discord.ui.Modal, title="Personnalité du salon"):
    """Modal d'édition de la personnalité du salon (modération)."""

    def __init__(self, settings, current: str):
        super().__init__()
        self._settings = settings
        self.personality_input = discord.ui.TextInput(
            label="Personnalité (ton, sujets, restrictions…)",
            style=discord.TextStyle.paragraph,
            placeholder="Ex. Salon cuisine, éviter les discussions politiques, parler de manière concise etc.",
            default=current,
            max_length=500,
            required=False,
        )
        self.add_item(self.personality_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        new_val = self.personality_input.value.strip()
        self._settings["personality"] = new_val
        msg = "Personnalité mise à jour." if new_val else "Personnalité effacée."
        await interaction.response.send_message(msg, ephemeral=True)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Chat(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = CogData("chat")
        self.data.set_builders(
            discord.Guild,
            DictTableBuilder("guild_config", {"chatbot_mode": "strict"}),
        )
        self.data.set_builders(
            discord.TextChannel,
            DictTableBuilder("channel_config", {
                "personality": "",
                "respond_everyone": False,
                "auto_transcribe": False,
            }),
        )
        self.profiles = ProfileStore()
        self.rappels = RappelStore()
        self._rappels_worker: Optional[RappelWorker] = None

        def developer_prompt() -> str:
            now = datetime.now(PARIS_TZ)
            profiles = getattr(developer_prompt, "_profiles", "")
            personality = getattr(developer_prompt, "_personality", "")
            channel_ctx = getattr(developer_prompt, "_channel_ctx", "")
            status_cog = self.bot.get_cog("Status")
            current_status = getattr(status_cog, "current_status", "") if status_cog else ""
            return DEV_PROMPT_BASE.format(
                weekday=now.strftime("%A"),
                datetime=now.strftime("%Y-%m-%d %H:%M"),
                profiles=profiles or "",
                personality=f"\nPERSONNALITÉ DU SALON:\n{personality}\n" if personality else "",
                channel_ctx=f"\nSALON ACTUEL : {channel_ctx}\n" if channel_ctx else "",
                status=current_status or "aucun",
            )

        self._get_dev_prompt = developer_prompt

        self.gpt_api = MariaGptApi(
            api_key=bot.config["OPENAI_API_KEY"],
            developer_prompt_template=self._get_dev_prompt,
            completion_model="gpt-5.4-mini",
            context_window=8192,
            context_age_hours=2,
        )

        self._processed: deque = deque(maxlen=100)

    async def cog_load(self) -> None:
        self._rappels_worker = RappelWorker(self.rappels, self._exec_rappel)
        await self._rappels_worker.start()

    async def cog_unload(self) -> None:
        if self._rappels_worker:
            await self._rappels_worker.stop()
        await self.gpt_api.close()
        self.data.close_all()

    # ------------------------------------------------------------------
    # Rappels
    # ------------------------------------------------------------------

    async def _exec_rappel(self, r: Rappel) -> None:
        channel = self.bot.get_channel(r.channel_id)
        if not channel:
            return

        ts = int(r.execute_at.timestamp())
        content = f"{r.description}\n-# Rappel · <@{r.user_id}> · <t:{ts}:R>"
        mentions = discord.AllowedMentions(users=True)

        orig = None
        if r.message_id:
            try:
                orig = await channel.fetch_message(r.message_id)
            except Exception:
                pass

        if orig:
            await orig.reply(content, allowed_mentions=mentions)
        else:
            await channel.send(content, allowed_mentions=mentions)

    # ------------------------------------------------------------------
    # Outils
    # ------------------------------------------------------------------

    async def _register_tools_from_cogs(self) -> None:
        tools: list[Tool] = []

        # Outils exposés par d'autres cogs
        for cog in self.bot.cogs.values():
            if cog.qualified_name != self.qualified_name and hasattr(cog, "GLOBAL_TOOLS"):
                tools.extend(cog.GLOBAL_TOOLS)

        # --- Recherche dans le cache hors-contexte ---
        async def _tool_search_cache(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            q = (tc.arguments.get("query") or "").strip()
            if not q or not ctx:
                return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))
            msgs = ctx.message_cache.get_recent(ctx.channel_id, 200)
            if not msgs:
                return ToolResponseRecord(tc.id, {"result": "Aucun message en cache."}, datetime.now(timezone.utc))
            compiled = await ctx.cache_search.search(q, msgs)
            return ToolResponseRecord(tc.id, {"result": compiled or "Rien de pertinent."}, datetime.now(timezone.utc))

        tools.append(Tool(
            name="search_context_cache",
            description="Recherche dans les messages récents hors contexte actuel.",
            properties={"query": {"type": "string", "description": "Question ou sujet à rechercher"}},
            function=_tool_search_cache,
        ))

        # --- Mise à jour des notes utilisateur ---
        async def _tool_update_notes(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            notes = (tc.arguments.get("addition") or "").strip()
            if not notes or not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Données manquantes"}, datetime.now(timezone.utc))
            self.profiles.append_notes(ctx.trigger_message.author.id, notes)
            return ToolResponseRecord(tc.id, {"success": True}, datetime.now(timezone.utc))

        tools.append(Tool(
            name="update_user_notes",
            description="Ajoute des infos durables sur l'auteur (identité, préférences, compétences). À utiliser seulement quand l'auteur partage une info nouvelle et durable.",
            properties={"addition": {"type": "string", "description": "Info à ajouter"}},
            function=_tool_update_notes,
        ))

        # --- Rappels ---
        async def _tool_schedule(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            if not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
            args = tc.arguments
            desc = (args.get("task_description") or "").strip()
            if not desc:
                return ToolResponseRecord(tc.id, {"error": "Description manquante"}, datetime.now(timezone.utc))

            execute_at_str = (args.get("execute_at") or "").strip()
            if execute_at_str:
                try:
                    execute_at = datetime.fromisoformat(execute_at_str)
                    if execute_at.tzinfo is None:
                        execute_at = execute_at.replace(tzinfo=PARIS_TZ)
                    execute_at = execute_at.astimezone(timezone.utc)
                except ValueError:
                    return ToolResponseRecord(tc.id, {"error": "Format execute_at invalide (ISO 8601 attendu)"}, datetime.now(timezone.utc))
            else:
                total = (args.get("delay_minutes") or 0) + (args.get("delay_hours") or 0) * 60
                execute_at = datetime.now(timezone.utc) + timedelta(minutes=total)

            total = int((execute_at - datetime.now(timezone.utc)).total_seconds() / 60)
            if total < 2:
                return ToolResponseRecord(tc.id, {"error": "Date trop proche (minimum 2 min)"}, datetime.now(timezone.utc))
            if total > 43200:
                return ToolResponseRecord(tc.id, {"error": "Date trop lointaine (max 30 jours)"}, datetime.now(timezone.utc))
            if self.rappels.count_pending(ctx.trigger_message.author.id) >= 10:
                return ToolResponseRecord(tc.id, {"error": "Max 10 rappels en attente"}, datetime.now(timezone.utc))

            rid = self.rappels.add(
                ctx.trigger_message.channel.id,
                ctx.trigger_message.author.id,
                desc,
                execute_at,
                ctx.trigger_message.id,
            )
            return ToolResponseRecord(tc.id, {
                "success": True, "task_id": rid,
                "execute_at": execute_at.isoformat(), "delay_minutes": total,
            }, datetime.now(timezone.utc))

        tools.append(Tool(
            name="schedule_reminder",
            description=(
                "Programme un rappel. Utilise execute_at (ISO 8601) pour une date absolue "
                "(ex. '2026-03-24T17:00:00' pour demain 17h — le fuseau par défaut est Europe/Paris), "
                "ou delay_minutes/delay_hours pour un délai relatif. execute_at est prioritaire."
            ),
            properties={
                "task_description": {"type": "string", "description": "Description de la tâche"},
                "execute_at": {"type": "string", "description": "Date/heure absolue ISO 8601 (prioritaire sur les délais)"},
                "delay_minutes": {"type": "integer", "description": "Délai en minutes (si pas de execute_at)"},
                "delay_hours": {"type": "integer", "description": "Délai en heures (si pas de execute_at)"},
            },
            function=_tool_schedule,
        ))

        async def _tool_list_reminders(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            if not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
            rappels = self.rappels.get_user_rappels(ctx.trigger_message.author.id)
            if not rappels:
                return ToolResponseRecord(tc.id, {"reminders": []}, datetime.now(timezone.utc))
            return ToolResponseRecord(tc.id, {
                "reminders": [
                    {"id": r.id, "description": r.description, "execute_at": r.execute_at.isoformat()}
                    for r in rappels
                ]
            }, datetime.now(timezone.utc))

        tools.append(Tool(
            name="list_reminders",
            description="Liste les rappels en attente de l'utilisateur. À appeler avant cancel_reminder pour obtenir les IDs.",
            properties={},
            function=_tool_list_reminders,
        ))

        async def _tool_cancel(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            tid = tc.arguments.get("task_id")
            if not tid or not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "task_id manquant"}, datetime.now(timezone.utc))
            ok = self.rappels.cancel(int(tid), ctx.trigger_message.author.id)
            return ToolResponseRecord(tc.id, {"success": ok}, datetime.now(timezone.utc))

        tools.append(Tool(
            name="cancel_reminder",
            description="Annule un rappel par son ID. Appelle list_reminders d'abord si tu n'as pas l'ID.",
            properties={"task_id": {"type": "integer", "description": "ID du rappel"}},
            function=_tool_cancel,
        ))

        # --- Discord : membres et salons ---
        async def _tool_server_users(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            if not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
            guild = ctx.trigger_message.guild
            if not guild:
                return ToolResponseRecord(tc.id, {"error": "Pas dans un serveur"}, datetime.now(timezone.utc))
            search = (tc.arguments.get("search") or "").strip().lower()
            pool = guild.members
            if search:
                pool = [m for m in pool if search in m.name.lower() or search in m.display_name.lower()]
            pool = pool[:60]
            return ToolResponseRecord(tc.id, {
                "total_members": guild.member_count,
                "shown": len(pool),
                "members": [
                    {
                        "name": m.name,
                        "display_name": m.display_name,
                        "id": str(m.id),
                        "top_roles": [r.name for r in m.roles if r.name != "@everyone"][-4:],
                    }
                    for m in pool
                ],
            }, datetime.now(timezone.utc))

        tools.append(Tool(
            name="get_server_users",
            description="Liste les membres du serveur avec leurs rôles principaux. Paramètre optionnel 'search' pour filtrer par nom.",
            properties={"search": {"type": "string", "description": "Filtre par nom ou pseudo (optionnel)"}},
            function=_tool_server_users,
        ))

        async def _tool_member_info(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            if not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
            guild = ctx.trigger_message.guild
            if not guild:
                return ToolResponseRecord(tc.id, {"error": "Pas dans un serveur"}, datetime.now(timezone.utc))
            uid_str = (tc.arguments.get("user_id") or "").strip()
            name_q = (tc.arguments.get("username") or "").strip().lower()
            member = None
            if uid_str:
                try:
                    member = guild.get_member(int(uid_str))
                    if not member:
                        member = await guild.fetch_member(int(uid_str))
                except (ValueError, discord.NotFound):
                    pass
            if not member and name_q:
                member = discord.utils.find(
                    lambda m: m.name.lower() == name_q or m.display_name.lower() == name_q,
                    guild.members,
                )
            if not member:
                return ToolResponseRecord(tc.id, {"error": "Membre introuvable"}, datetime.now(timezone.utc))
            return ToolResponseRecord(tc.id, {
                "id": str(member.id),
                "username": member.name,
                "display_name": member.display_name,
                "roles": [r.name for r in member.roles if r.name != "@everyone"],
                "account_created": member.created_at.strftime("%Y-%m-%d"),
                "joined_server": member.joined_at.strftime("%Y-%m-%d") if member.joined_at else None,
                "is_bot": member.bot,
                "avatar_url": str(member.display_avatar.url) if member.display_avatar else None,
            }, datetime.now(timezone.utc))

        tools.append(Tool(
            name="get_member_info",
            description="Carte d'identité complète d'un membre : rôles, dates de création et d'arrivée, avatar. Recherche par ID ou pseudo exact.",
            properties={
                "user_id": {"type": "string", "description": "ID Discord (prioritaire)"},
                "username": {"type": "string", "description": "Nom d'utilisateur ou pseudo (recherche exacte)"},
            },
            function=_tool_member_info,
        ))

        async def _tool_channel_info(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            if not ctx or not ctx.trigger_message:
                return ToolResponseRecord(tc.id, {"error": "Contexte manquant"}, datetime.now(timezone.utc))
            cid_str = (tc.arguments.get("channel_id") or "").strip()
            if cid_str:
                channel = (
                    ctx.trigger_message.guild.get_channel(int(cid_str))
                    if ctx.trigger_message.guild
                    else None
                )
            else:
                channel = ctx.trigger_message.channel
            if not channel:
                return ToolResponseRecord(tc.id, {"error": "Salon introuvable"}, datetime.now(timezone.utc))
            info: dict = {"id": str(channel.id), "name": channel.name, "type": str(channel.type)}
            if isinstance(channel, discord.TextChannel):
                info.update({
                    "topic": channel.topic or "",
                    "category": channel.category.name if channel.category else None,
                    "nsfw": channel.nsfw,
                    "slowmode_delay": channel.slowmode_delay,
                    "member_count": len(channel.members),
                })
            elif isinstance(channel, discord.Thread):
                info.update({
                    "parent": channel.parent.name if channel.parent else None,
                    "archived": channel.archived,
                    "member_count": channel.member_count,
                })
            elif isinstance(channel, discord.VoiceChannel):
                info.update({
                    "category": channel.category.name if channel.category else None,
                    "user_limit": channel.user_limit,
                    "members_connected": [m.name for m in channel.members],
                })
            return ToolResponseRecord(tc.id, info, datetime.now(timezone.utc))

        tools.append(Tool(
            name="get_channel_info",
            description="Informations sur un salon Discord : sujet, catégorie, NSFW, slowmode, membres présents. Par défaut le salon actuel.",
            properties={"channel_id": {"type": "string", "description": "ID du salon (optionnel, défaut = salon actuel)"}},
            function=_tool_channel_info,
        ))

        async def _tool_profile(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
            uid_str = tc.arguments.get("user_id")
            if not uid_str:
                return ToolResponseRecord(tc.id, {"error": "user_id manquant"}, datetime.now(timezone.utc))
            try:
                full = self.profiles.get_full(int(uid_str))
            except ValueError:
                return ToolResponseRecord(tc.id, {"error": "user_id invalide"}, datetime.now(timezone.utc))
            return ToolResponseRecord(tc.id, {"profile": full or "Aucun profil."}, datetime.now(timezone.utc))

        tools.append(Tool(
            name="get_user_profile",
            description="Consulte le profil (fixe + notes) d'un utilisateur.",
            properties={"user_id": {"type": "string", "description": "ID Discord"}},
            function=_tool_profile,
        ))

        self.gpt_api.update_tools(tools)

    # ------------------------------------------------------------------
    # Logique de réponse
    # ------------------------------------------------------------------

    def _channel_config(self, channel) -> dict:
        target = channel.parent if isinstance(channel, discord.Thread) else channel
        if isinstance(target, discord.TextChannel):
            return self.data.get(target).settings("channel_config")
        return {}

    def _should_respond(self, message: discord.Message) -> bool:
        if not message.guild:
            return False
        mode = self.data.get(message.guild).settings("guild_config").get("chatbot_mode", "strict")
        if mode == "off":
            return False
        if mode == "greedy" and self.bot.user:
            pattern = r'(?<![a-z0-9_])' + re.escape(self.bot.user.name.lower()) + r'(?![a-z0-9_])'
            if re.search(pattern, message.content.lower()):
                return True
        if self.bot.user in message.mentions:
            return True
        if message.mention_everyone:
            cfg = self._channel_config(message.channel)
            if cfg.get("respond_everyone", False):
                return True
        return False

    def _inject_profiles(self, message: discord.Message) -> None:
        parts: list[str] = []
        if p := self.profiles.get_full(message.author.id):
            parts.append(f"**{message.author.name}** (auteur):\n{p}")
        for u in message.mentions:
            if u.id != message.author.id and (p := self.profiles.get_full(u.id)):
                parts.append(f"**{u.name}**:\n{p}")
        self._get_dev_prompt._profiles = ("PROFILS:\n\n" + "\n\n".join(parts) + "\n") if parts else ""

    def _inject_personality(self, channel) -> None:
        target = channel.parent if isinstance(channel, discord.Thread) else channel
        pers = (
            self.data.get(target).settings("channel_config").get("personality", "")
            if isinstance(target, discord.TextChannel)
            else ""
        )
        self._get_dev_prompt._personality = pers or ""

    def _inject_channel_context(self, channel) -> None:
        target = channel.parent if isinstance(channel, discord.Thread) else channel
        parts: list[str] = []
        if isinstance(channel, discord.Thread):
            parts.append(f"Thread « {channel.name} » (dans #{target.name})")
        elif hasattr(target, "name"):
            parts.append(f"#{target.name}")
        if isinstance(target, discord.TextChannel):
            if target.category:
                parts.append(f"catégorie : {target.category.name}")
            if target.topic:
                parts.append(f"sujet : \"{target.topic[:120]}\"")
            if target.nsfw:
                parts.append("NSFW")
        guild = getattr(channel, "guild", None)
        if guild:
            parts.append(f"serveur : {guild.name} ({guild.member_count} membres)")
        self._get_dev_prompt._channel_ctx = " · ".join(parts) if parts else ""

    def _is_quiet_channel(self, channel_id: int, threshold: int = 180) -> bool:
        """True si aucune activité récente dans ce salon (seuil en secondes)."""
        recent = self.gpt_api.session_manager.message_cache.get_recent(channel_id, 5)
        if len(recent) < 2:
            return True
        age = (datetime.now(timezone.utc) - recent[-2]["created_at"]).total_seconds()
        return age > threshold

    async def _seed_cache_from_history(self, channel, limit: int = 300) -> None:
        """Pré-alimente le MessageCache (nano) avec l'historique Discord du salon.
        Ne touche pas au contexte principal — uniquement le cache de recherche."""
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        message_cache = self.gpt_api.session_manager.message_cache
        history: list[discord.Message] = []
        try:
            async for msg in channel.history(limit=limit):
                if not msg.author.bot and msg.content.strip():
                    history.append(msg)
        except Exception:
            return
        history.reverse()  # du plus vieux au plus récent
        for msg in history:
            created = msg.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            message_cache.push(channel.id, msg.author.display_name, msg.clean_content, created)

    # ------------------------------------------------------------------
    # Événements
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        key = (message.channel.id, message.id)
        if key in self._processed:
            return
        self._processed.append(key)

        # Pré-alimenter le cache nano si le salon n'a pas encore d'historique chargé
        cache = self.gpt_api.session_manager.message_cache
        if not cache.get_recent(message.channel.id, 1):
            await self._seed_cache_from_history(message.channel)

        should_respond = self._should_respond(message)
        session = self.gpt_api.session_manager.get_or_create(message.channel)
        await session.ingest_message(message, is_context_only=not should_respond)

        if not should_respond:
            return

        self._inject_profiles(message)
        self._inject_personality(message.channel)
        self._inject_channel_context(message.channel)

        async with message.channel.typing():
            try:
                resp = await self.gpt_api.run_completion(message.channel, trigger_message=message)
            finally:
                self._get_dev_prompt._profiles = ""
                self._get_dev_prompt._personality = ""
                self._get_dev_prompt._channel_ctx = ""

        # Preuve d'utilisation des outils visibles
        text = resp.text
        visible_parts: list[str] = []
        for t in resp.used_tools:
            name = t["name"]
            args = t.get("args", {})
            if name in _HIDDEN_TOOLS:
                continue
            if name == "search_web":
                q = args.get("query", "").strip()
                label = f'**Recherche web** — "{q}"' if q else "**Recherche web**"
            elif name == "read_web_page":
                url = args.get("url", "")
                label = f"**Lecture** — <{url}>"
            elif name == "schedule_reminder":
                desc = args.get("task_description", "").strip()
                execute_at_str = (args.get("execute_at") or "").strip()
                if execute_at_str:
                    try:
                        dt = datetime.fromisoformat(execute_at_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=PARIS_TZ)
                        ts = int(dt.timestamp())
                        delay_str = f" · <t:{ts}:f>"
                    except ValueError:
                        delay_str = f" · {execute_at_str}"
                else:
                    total = (args.get("delay_minutes") or 0) + (args.get("delay_hours") or 0) * 60
                    delay_str = f" · dans {_fmt_delay(total)}" if total else ""
                label = f'**Rappel planifié** — "{desc}"{delay_str}' if desc else "**Rappel planifié**"
            elif name == "screenshot_page":
                url = args.get("url", "")
                try:
                    domain = urlparse(url).netloc.removeprefix("www.")
                except Exception:
                    domain = ""
                label = f"**Capture d'écran** — {domain}" if domain else "**Capture d'écran**"
            elif name == "cancel_reminder":
                tid = args.get("task_id", "")
                label = f"**Rappel #{tid} annulé**" if tid else "**Rappel annulé**"
            else:
                label = f"**{name.replace('_', ' ').capitalize()}**"
            if label not in visible_parts:
                visible_parts.append(label)
        if visible_parts:
            tool_lines = "\n".join(f"-# {p}" for p in visible_parts)
            text = f"{tool_lines}\n{text}"

        quiet = self._is_quiet_channel(message.channel.id)
        await send_long(message.channel, text, reply_to=None if quiet else message)

        # Envoyer les captures d'écran produites par screenshot_page
        for tr in resp.tool_responses:
            data = getattr(tr, "response_data", None)
            if isinstance(data, dict) and "screenshot_url" in data:
                embed = discord.Embed(url=data.get("source_url", data["screenshot_url"]))
                embed.set_image(url=data["screenshot_url"])
                await message.channel.send(embed=embed)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    @app_commands.command(name="preferences", description="Consulte ton profil et les notes de Maria")
    async def cmd_preferences(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            view=PreferencesView(self.profiles, interaction.user.id),
            ephemeral=True,
        )

    @app_commands.command(name="rappels", description="Liste tes rappels en attente")
    async def cmd_rappels(self, interaction: discord.Interaction) -> None:
        tasks = self.rappels.get_user_rappels(interaction.user.id)
        if not tasks:
            await interaction.response.send_message("Aucun rappel en attente.", ephemeral=True)
            return
        await interaction.response.send_message(
            view=RappelsView(tasks, interaction.user.id, self.rappels), ephemeral=True
        )

    @app_commands.command(name="info", description="Statistiques de la session en cours")
    async def cmd_info(self, interaction: discord.Interaction) -> None:
        session = self.gpt_api.session_manager.get(interaction.channel_id)
        ch = interaction.channel
        target = ch.parent if isinstance(ch, discord.Thread) else ch
        mode = "strict"
        personality = ""
        if interaction.guild:
            mode = self.data.get(interaction.guild).settings("guild_config").get("chatbot_mode", "strict")
        if isinstance(target, discord.TextChannel):
            personality = self.data.get(target).settings("channel_config").get("personality", "")
        await interaction.response.send_message(
            view=InfoView(
                session.get_stats() if session else None,
                interaction.channel,
                mode=mode,
                personality=personality,
            ),
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # Groupe /chatbot
    # ------------------------------------------------------------------

    chatbot = app_commands.Group(
        name="chatbot",
        description="Configuration du chatbot pour ce salon / serveur",
        default_permissions=discord.Permissions(manage_messages=True),
        guild_only=True,
    )

    @chatbot.command(name="mode", description="Définit le mode de réponse du bot")
    @app_commands.describe(mode="Mode de réponse")
    @app_commands.choices(mode=[
        app_commands.Choice(name="Off — désactivé",                            value="off"),
        app_commands.Choice(name="Strict — répond uniquement sur mention",     value="strict"),
        app_commands.Choice(name="Greedy — répond aussi si son nom est cité",  value="greedy"),
    ])
    async def chatbot_mode(
        self, interaction: discord.Interaction, mode: app_commands.Choice[str]
    ) -> None:
        if not interaction.guild:
            return await interaction.response.send_message("Pas dans un serveur.", ephemeral=True)
        self.data.get(interaction.guild).settings("guild_config")["chatbot_mode"] = mode.value
        await interaction.response.send_message(f"Mode: **{mode.name}**", ephemeral=True)

    @chatbot.command(name="forget", description="Vide l'historique de conversation de ce salon")
    async def chatbot_forget(self, interaction: discord.Interaction) -> None:
        session = self.gpt_api.session_manager.get(interaction.channel_id)
        if session:
            session.forget()
        await interaction.response.send_message("Historique vidé.", ephemeral=True)

    @chatbot.command(name="personality", description="Édite la personnalité du bot pour ce salon")
    async def chatbot_personality(self, interaction: discord.Interaction) -> None:
        ch = interaction.channel
        target = ch.parent if isinstance(ch, discord.Thread) else ch
        if not isinstance(target, discord.TextChannel):
            return await interaction.response.send_message("Salon textuel requis.", ephemeral=True)
        s = self.data.get(target).settings("channel_config")
        await interaction.response.send_modal(PersonalityModal(s, s.get("personality", "")))

    @chatbot.command(name="everyone", description="Définit si Maria répond aux mentions @everyone et @here")
    @app_commands.describe(actif="Activer ou désactiver la réponse aux @everyone / @here")
    async def chatbot_everyone(self, interaction: discord.Interaction, actif: bool) -> None:
        ch = interaction.channel
        target = ch.parent if isinstance(ch, discord.Thread) else ch
        if not isinstance(target, discord.TextChannel):
            return await interaction.response.send_message("Salon textuel requis.", ephemeral=True)
        self.data.get(target).settings("channel_config")["respond_everyone"] = actif
        state = "activée" if actif else "désactivée"
        await interaction.response.send_message(
            f"Réponse aux @everyone / @here **{state}** sur ce salon.", ephemeral=True
        )

    @chatbot.command(name="autotranscribe", description="Définit si Maria transcrit automatiquement les messages vocaux")
    @app_commands.describe(actif="Activer ou désactiver la transcription automatique")
    async def chatbot_autotranscribe(self, interaction: discord.Interaction, actif: bool) -> None:
        ch = interaction.channel
        target = ch.parent if isinstance(ch, discord.Thread) else ch
        if not isinstance(target, discord.TextChannel):
            return await interaction.response.send_message("Salon textuel requis.", ephemeral=True)
        self.data.get(target).settings("channel_config")["auto_transcribe"] = actif
        state = "activée" if actif else "désactivée"
        await interaction.response.send_message(
            f"Transcription automatique des messages vocaux **{state}** sur ce salon.", ephemeral=True
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Chat(bot))
