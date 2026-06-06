"""Cog Watcher — IA passive qui lit les salons opt-in et génère des suggestions.

Fonctionnement (pull, pas de notification proactive) :
- `on_message` empile les messages des salons surveillés dans un buffer borné.
- Une boucle de fond analyse périodiquement (1 appel nano par salon ayant assez de
  nouveaux messages ou inactif avec du contenu) et écrit des suggestions typées.
- Les suggestions se consultent via `/suggestions` (perso, MP) et `/events` (modos).
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

from common.dataio import CogData, DictTableBuilder
from common.llm import MariaLLMClient
from common.profiles import ProfileStore
from common.rappels import (
    RECURRENCE_DAILY,
    RECURRENCE_NONE,
    RECURRENCE_WEEKLY,
    RappelStore,
)
from common.suggestions import (
    EVENT_KINDS,
    KIND_GROUP_ACTIVITY,
    KIND_PERSONAL_REMINDER,
    KIND_PROFILE_UPDATE,
    KIND_SERVER_EVENT,
    MAX_PENDING_EVENTS,
    MAX_PENDING_PERSONAL,
    PERSONAL_KINDS,
    SuggestionStore,
)
from common.timezones import PARIS_TZ

from cogs.watcher.views import (
    EventCompleteModal,
    ReminderCompleteModal,
    ViewContext,
    build_suggestions_view,
    has_valid_date,
    parse_when,
)

logger = logging.getLogger("MARIA.Watcher")

MODEL_NANO = "gpt-5.4-nano"

# Déclencheurs d'analyse
MIN_NEW_MESSAGES = 15          # nb de nouveaux messages avant analyse
IDLE_FLUSH_SECONDS = 30 * 60   # analyse un buffer non vide resté inactif
ANALYZE_INTERVAL_MIN = 5       # période de la boucle de fond
BUFFER_MAX = 60                # messages conservés par salon
MIN_MESSAGES_TO_ANALYZE = 5    # ne rien faire en dessous (bruit)

CONFIDENCE_THRESHOLD = 0.6
ANALYSIS_MAX_TOKENS = 1200

_VALID_KINDS = (KIND_PERSONAL_REMINDER, KIND_SERVER_EVENT, KIND_PROFILE_UPDATE, KIND_GROUP_ACTIVITY)
_VALID_RECURRENCES = (RECURRENCE_NONE, RECURRENCE_DAILY, RECURRENCE_WEEKLY)

# Schéma JSON strict de la sortie nano
_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "suggestions_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "enum": list(_VALID_KINDS)},
                            "target_user_id": {"type": "string"},
                            "content": {"type": "string"},
                            "when": {"type": "string"},
                            "recurrence": {"type": "string", "enum": list(_VALID_RECURRENCES)},
                            "category": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "kind", "target_user_id", "content",
                            "when", "recurrence", "category", "confidence",
                        ],
                    },
                }
            },
            "required": ["suggestions"],
        },
    },
}

# Durée par défaut d'un événement Discord créé (heures)
EVENT_DEFAULT_DURATION_HOURS = 2

# Schéma de normalisation d'une expression temporelle en ISO 8601
_DATETIME_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "datetime_parse",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"iso": {"type": "string"}},
            "required": ["iso"],
        },
    },
}

_SYSTEM_PROMPT = """Tu es un analyste discret. Tu lis un extrait de conversation Discord et tu proposes des SUGGESTIONS utiles, sans jamais inventer.

Types de suggestions :
- personal_reminder : un utilisateur veut clairement se souvenir/être rappelé de quelque chose à un moment donné. target_user_id = l'auteur concerné. when = date/heure ISO 8601 (heure de Paris) si déductible, sinon "". recurrence = daily/weekly si récurrent, sinon none.
- server_event : un évènement qui concerne tout le serveur (soirée, session de jeu commune, sortie). target_user_id = "". when = ISO 8601 si connu.
- group_activity : une activité de groupe envisagée mais sans date ferme. target_user_id = "".
- profile_update : un fait stable et personnel révélé sur un utilisateur (ville, âge, métier, préférence forte). target_user_id = l'utilisateur concerné. category = identité/préférences/projets/perso. content = le fait, court, à la 3e personne.

Règles :
- N'utilise QUE les informations présentes dans l'extrait. Ne devine pas, n'extrapole pas.
- target_user_id doit être un id présent dans la liste PARTICIPANTS, sinon "".
- Ne propose pas de profile_update déjà présent dans les NOTES EXISTANTES.
- Ne répète pas une suggestion déjà présente dans SUGGESTIONS EN ATTENTE.
- confidence entre 0 et 1. Sois sévère : en dessous de 0.6, abstiens-toi.
- S'il n'y a rien de pertinent, renvoie une liste vide.
- content concis et clair, en français."""


class Watcher(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = CogData("watcher")
        self.data.set_builders(
            discord.Guild,
            DictTableBuilder("guild_config", {
                "watched_channels": "",
            }),
        )
        self.store = SuggestionStore()
        self.profiles = ProfileStore()
        self.rappels = RappelStore()
        self._client: Optional[MariaLLMClient] = None

        # État volatil par salon surveillé (clé = channel.id du salon parent)
        self._buffers: dict[int, deque] = {}
        self._new_counts: dict[int, int] = {}
        self._last_activity: dict[int, float] = {}
        self._guild_of: dict[int, int] = {}

    async def cog_load(self) -> None:
        self.analysis_loop.start()

    async def cog_unload(self) -> None:
        self.analysis_loop.cancel()
        if self._client:
            await self._client.close()
        self.data.close_all()

    def _get_client(self) -> MariaLLMClient:
        if self._client is None:
            self._client = MariaLLMClient(
                api_key=self.bot.config["OPENAI_API_KEY"],
                completion_model=MODEL_NANO,
                max_tokens=ANALYSIS_MAX_TOKENS,
            )
        return self._client

    # ------------------------------------------------------------------
    # Config de surveillance (stockée par guild)
    # ------------------------------------------------------------------

    def _watched_channels(self, guild: discord.Guild) -> set[int]:
        raw = self.data.get(guild).settings("guild_config").get("watched_channels", default="") or ""
        out: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
        return out

    def _set_watched_channels(self, guild: discord.Guild, channels: set[int]) -> None:
        value = ",".join(str(c) for c in sorted(channels))
        self.data.get(guild).settings("guild_config")["watched_channels"] = value

    @staticmethod
    def _target_channel(channel) -> Optional[discord.TextChannel]:
        target = channel.parent if isinstance(channel, discord.Thread) else channel
        return target if isinstance(target, discord.TextChannel) else None

    # ------------------------------------------------------------------
    # Collecte passive
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        target = self._target_channel(message.channel)
        if target is None:
            return
        if target.id not in self._watched_channels(message.guild):
            return
        text = (message.content or "").strip()
        if not text:
            return

        cid = target.id
        buf = self._buffers.get(cid)
        if buf is None:
            buf = deque(maxlen=BUFFER_MAX)
            self._buffers[cid] = buf
        buf.append((message.author.id, message.author.display_name, datetime.now(timezone.utc), text))
        self._new_counts[cid] = self._new_counts.get(cid, 0) + 1
        self._last_activity[cid] = time.time()
        self._guild_of[cid] = message.guild.id

    # ------------------------------------------------------------------
    # Boucle d'analyse
    # ------------------------------------------------------------------

    @tasks.loop(minutes=ANALYZE_INTERVAL_MIN)
    async def analysis_loop(self) -> None:
        try:
            self.store.expire_old()
        except Exception as e:
            logger.warning(f"expire_old a échoué : {e}")

        now = time.time()
        for cid in list(self._buffers.keys()):
            buf = self._buffers.get(cid)
            if not buf:
                continue
            count = self._new_counts.get(cid, 0)
            idle = now - self._last_activity.get(cid, now)
            ready = count >= MIN_NEW_MESSAGES or (idle >= IDLE_FLUSH_SECONDS and count > 0)
            if not ready or len(buf) < MIN_MESSAGES_TO_ANALYZE:
                continue
            try:
                await self._analyze_channel(cid)
            except Exception as e:
                logger.error(f"Analyse salon {cid} échouée : {e}", exc_info=True)
            finally:
                self._new_counts[cid] = 0

    @analysis_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    async def _analyze_channel(self, channel_id: int) -> int:
        buf = self._buffers.get(channel_id)
        if not buf:
            return 0
        guild_id = self._guild_of.get(channel_id, 0)
        return await self._run_analysis(guild_id=guild_id, channel_id=channel_id, snapshot=list(buf))

    async def _run_analysis(
        self, *, guild_id: int, channel_id: int, snapshot: list[tuple]
    ) -> int:
        """Analyse un extrait de conversation et écrit les suggestions retenues. Retourne le nombre stocké."""
        if not snapshot:
            return 0

        participants: dict[int, str] = {}
        lines: list[str] = []
        for uid, name, ts, text in snapshot:
            participants.setdefault(uid, name)
            lines.append(f"[{ts.astimezone(PARIS_TZ):%d/%m %H:%M}] {name} ({uid}): {text}")
        transcript = "\n".join(lines)[-6000:]

        notes_blocks: list[str] = []
        for uid, name in participants.items():
            notes = self.profiles.get_notes(uid)
            if notes:
                notes_blocks.append(f"{name} ({uid}):\n{notes}")
        notes_section = "\n\n".join(notes_blocks) if notes_blocks else "(aucune)"

        pending = self.store.pending_descriptions(guild_id) if guild_id else []
        pending_section = "\n".join(f"- {p}" for p in pending) if pending else "(aucune)"

        participants_section = "\n".join(f"- {uid}: {name}" for uid, name in participants.items())
        now_paris = datetime.now(PARIS_TZ)
        user_prompt = (
            f"Date actuelle : {now_paris:%A %d/%m/%Y %H:%M} (Paris).\n\n"
            f"PARTICIPANTS:\n{participants_section}\n\n"
            f"NOTES EXISTANTES:\n{notes_section}\n\n"
            f"SUGGESTIONS EN ATTENTE:\n{pending_section}\n\n"
            f"CONVERSATION:\n{transcript}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        completion = await self._get_client().chat(
            messages, model=MODEL_NANO, response_format=_ANALYSIS_SCHEMA
        )
        if not completion.choices:
            return 0
        raw = completion.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Sortie nano non-JSON pour salon {channel_id}")
            return 0

        suggestions = data.get("suggestions") or []
        stored = 0
        for s in suggestions:
            if self._store_suggestion(s, guild_id=guild_id, channel_id=channel_id, participants=participants):
                stored += 1
        if stored:
            logger.info(f"{stored} suggestion(s) enregistrée(s) depuis le salon {channel_id}")
        return stored

    def _store_suggestion(self, s: dict, *, guild_id: int, channel_id: int, participants: dict[int, str]) -> bool:
        if not isinstance(s, dict):
            return False
        kind = s.get("kind")
        if kind not in _VALID_KINDS:
            return False
        try:
            confidence = float(s.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < CONFIDENCE_THRESHOLD:
            return False
        content = (s.get("content") or "").strip()
        if not content:
            return False

        # Résolution de la cible (obligatoire pour les suggestions personnelles)
        target_user_id: Optional[int] = None
        raw_target = str(s.get("target_user_id") or "").strip()
        if raw_target.isdigit():
            tid = int(raw_target)
            if tid in participants:
                target_user_id = tid
        if kind in PERSONAL_KINDS and target_user_id is None:
            return False

        # Anti-spam : plafonds par portée
        if kind in PERSONAL_KINDS:
            if self.store.count_pending(target_user_id=target_user_id, kinds=PERSONAL_KINDS) >= MAX_PENDING_PERSONAL:
                return False
        else:
            if self.store.count_pending(guild_id=guild_id, kinds=EVENT_KINDS) >= MAX_PENDING_EVENTS:
                return False

        when = (s.get("when") or "").strip()
        recurrence = s.get("recurrence") if s.get("recurrence") in _VALID_RECURRENCES else RECURRENCE_NONE
        category = (s.get("category") or "").strip()

        if kind == KIND_PROFILE_UPDATE:
            # Évite de reproposer un fait déjà noté.
            existing = self.profiles.get_notes(target_user_id).lower() if target_user_id else ""
            if content.lower() in existing:
                return False
            payload = {"info": content, "category": category or "perso"}
        elif kind == KIND_PERSONAL_REMINDER:
            payload = {"description": content, "when": when, "recurrence": recurrence}
        else:  # server_event / group_activity
            payload = {"title": content, "when": when}

        new_id = self.store.add(
            kind=kind,
            guild_id=guild_id,
            channel_id=channel_id,
            target_user_id=target_user_id,
            payload=payload,
            source_excerpt=content[:200],
        )
        return new_id is not None

    async def _nl_to_datetime(self, text: str) -> Optional[datetime]:
        """Normalise une expression temporelle française en datetime UTC via nano."""
        text = (text or "").strip()
        if not text:
            return None
        # Court-circuit : si l'utilisateur a déjà tapé de l'ISO.
        direct = parse_when(text)
        if direct is not None:
            return direct
        now_paris = datetime.now(PARIS_TZ)
        messages = [
            {
                "role": "system",
                "content": (
                    "Convertis l'expression temporelle de l'utilisateur en date ISO 8601 "
                    "(format YYYY-MM-DDTHH:MM:SS, fuseau Europe/Paris, sans suffixe de fuseau). "
                    f"Date et heure actuelles : {now_paris:%Y-%m-%dT%H:%M:%S} ({now_paris:%A}). "
                    "Choisis toujours une date dans le futur. Si l'heure n'est pas précisée, utilise 09:00. "
                    "Si c'est impossible à interpréter, renvoie iso=\"\"."
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            completion = await self._get_client().chat(
                messages, model=MODEL_NANO, response_format=_DATETIME_SCHEMA
            )
        except Exception as e:
            logger.warning(f"Normalisation de date échouée : {e}")
            return None
        if not completion.choices:
            return None
        try:
            iso = (json.loads(completion.choices[0].message.content or "{}").get("iso") or "").strip()
        except (json.JSONDecodeError, TypeError):
            return None
        return parse_when(iso)

    # ------------------------------------------------------------------
    # Groupe /watch (modération)
    # ------------------------------------------------------------------

    watch = app_commands.Group(
        name="watch",
        description="Configure la lecture passive des salons par MARIA",
        default_permissions=discord.Permissions(manage_messages=True),
        guild_only=True,
    )

    @watch.command(name="toggle", description="Active ou désactive la lecture passive sur ce salon")
    async def watch_toggle(self, interaction: discord.Interaction) -> None:
        target = self._target_channel(interaction.channel)
        if target is None or not interaction.guild:
            return await interaction.response.send_message("Salon textuel requis.", ephemeral=True)
        watched = self._watched_channels(interaction.guild)
        if target.id in watched:
            watched.discard(target.id)
            self._set_watched_channels(interaction.guild, watched)
            self._buffers.pop(target.id, None)
            self._new_counts.pop(target.id, None)
            await interaction.response.send_message(
                f"Lecture passive **désactivée** sur {target.mention}.", ephemeral=True
            )
        else:
            watched.add(target.id)
            self._set_watched_channels(interaction.guild, watched)
            await interaction.response.send_message(
                f"Lecture passive **activée** sur {target.mention}. "
                f"MARIA proposera des suggestions via `/suggestions`.",
                ephemeral=True,
            )

    @watch.command(name="list", description="Liste les salons surveillés")
    async def watch_list(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await interaction.response.send_message("Pas dans un serveur.", ephemeral=True)
        watched = self._watched_channels(interaction.guild)
        watched_str = ", ".join(f"<#{c}>" for c in watched) if watched else "_aucun_"
        await interaction.response.send_message(
            f"**Salons surveillés :** {watched_str}", ephemeral=True
        )

    @watch.command(name="analyze", description="[Debug] Force l'analyse immédiate de ce salon")
    @app_commands.describe(historique="Nombre de messages récents à analyser (défaut 40)")
    async def watch_analyze(
        self, interaction: discord.Interaction, historique: app_commands.Range[int, 5, 100] = 40
    ) -> None:
        target = self._target_channel(interaction.channel)
        if target is None or not interaction.guild:
            return await interaction.response.send_message("Salon textuel requis.", ephemeral=True)

        await interaction.response.defer(ephemeral=True, thinking=True)

        # Construit un extrait depuis l'historique récent (indépendant du buffer).
        snapshot: list[tuple] = []
        try:
            async for msg in target.history(limit=historique):
                if msg.author.bot:
                    continue
                text = (msg.content or "").strip()
                if text:
                    snapshot.append((msg.author.id, msg.author.display_name, msg.created_at, text))
        except discord.HTTPException as e:
            return await interaction.followup.send(f"Lecture de l'historique impossible : `{e}`", ephemeral=True)

        snapshot.reverse()  # ordre chronologique
        if len(snapshot) < MIN_MESSAGES_TO_ANALYZE:
            return await interaction.followup.send(
                f"Pas assez de messages exploitables ({len(snapshot)}).", ephemeral=True
            )

        try:
            stored = await self._run_analysis(
                guild_id=interaction.guild.id, channel_id=target.id, snapshot=snapshot
            )
        except Exception as e:
            logger.error(f"Analyse forcée échouée ({target.id}) : {e}", exc_info=True)
            return await interaction.followup.send(f"Analyse échouée : `{e}`", ephemeral=True)

        # Le buffer ayant servi de déclencheur naturel est remis à zéro.
        self._new_counts[target.id] = 0
        await interaction.followup.send(
            f"Analyse de {len(snapshot)} message(s) terminée : **{stored}** suggestion(s) ajoutée(s). "
            f"Consulte `/suggestions`.",
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # /suggestions — commande unique (perso + événements pour les modos)
    # ------------------------------------------------------------------

    def _view_context(self, interaction: discord.Interaction) -> ViewContext:
        guild = interaction.guild
        is_mod = bool(
            guild
            and isinstance(interaction.user, discord.Member)
            and interaction.user.guild_permissions.manage_events
        )
        return ViewContext(interaction.user.id, guild.id if guild else 0, is_mod)

    @app_commands.command(
        name="suggestions",
        description="Consulte les suggestions de MARIA (rappels, profil, et événements si tu es modo)",
    )
    async def cmd_suggestions(self, interaction: discord.Interaction) -> None:
        ctx = self._view_context(interaction)
        await interaction.response.send_message(
            view=build_suggestions_view(self, ctx), ephemeral=True
        )

    # --- Suggestions personnelles -------------------------------------

    async def handle_personal_action(
        self, interaction: discord.Interaction, suggestion_id: int, *, accept: bool, ctx: ViewContext
    ) -> None:
        sugg = self.store.get(suggestion_id)
        if not sugg or sugg.status != "pending" or sugg.target_user_id != interaction.user.id:
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        if not accept:
            self.store.set_status(suggestion_id, "rejected", user_id=interaction.user.id)
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        if sugg.kind == KIND_PROFILE_UPDATE:
            info = sugg.payload.get("info", "").strip()
            category = sugg.payload.get("category", "perso").strip() or "perso"
            if info:
                self.profiles.append_notes(interaction.user.id, f"[{category}] {info}")
            self.store.set_status(suggestion_id, "accepted", user_id=interaction.user.id)
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        # personal_reminder : date exploitable -> planifie ; sinon modal de précision.
        if has_valid_date(sugg):
            execute_at = parse_when(sugg.payload.get("when", ""))
            self.rappels.add(
                sugg.channel_id,
                interaction.user.id,
                sugg.payload.get("description", "Rappel"),
                execute_at,
                recurrence=sugg.payload.get("recurrence", RECURRENCE_NONE),
            )
            self.store.set_status(suggestion_id, "accepted", user_id=interaction.user.id)
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        await interaction.response.send_modal(
            ReminderCompleteModal(
                self, ctx, suggestion_id, description=sugg.payload.get("description", "")
            )
        )

    async def finish_reminder_from_modal(
        self, interaction: discord.Interaction, suggestion_id: int,
        *, description: str, when_text: str, ctx: ViewContext,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        sugg = self.store.get(suggestion_id)
        if not sugg or sugg.status != "pending" or sugg.target_user_id != interaction.user.id:
            return await interaction.followup.send("Suggestion déjà traitée.", ephemeral=True)

        execute_at = await self._nl_to_datetime(when_text)
        if execute_at is None or execute_at <= datetime.now(timezone.utc):
            return await interaction.followup.send(
                "Je n'ai pas réussi à comprendre la date. Réessaie (ex : « demain 18h »).",
                ephemeral=True,
            )

        self.rappels.add(
            sugg.channel_id,
            interaction.user.id,
            description or sugg.payload.get("description", "Rappel"),
            execute_at,
            recurrence=sugg.payload.get("recurrence", RECURRENCE_NONE),
        )
        self.store.set_status(suggestion_id, "accepted", user_id=interaction.user.id)
        ts = int(execute_at.timestamp())
        await interaction.followup.send(
            view=build_suggestions_view(self, ctx, header=f"✅ Rappel programmé pour <t:{ts}:f> (<t:{ts}:R>)."),
            ephemeral=True,
        )

    # --- Événements serveur (modos) -> événement Discord natif --------

    async def handle_event_action(
        self, interaction: discord.Interaction, suggestion_id: int, *, accept: bool, ctx: ViewContext
    ) -> None:
        guild = interaction.guild
        if not guild or not ctx.is_mod:
            return await interaction.response.send_message("Action réservée aux modérateurs.", ephemeral=True)
        sugg = self.store.get(suggestion_id)
        if not sugg or sugg.status != "pending" or sugg.guild_id != guild.id:
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        if not accept:
            self.store.set_status(suggestion_id, "rejected")
            return await interaction.response.edit_message(view=build_suggestions_view(self, ctx))

        # Toujours un modal : un événement Discord requiert nom, date et lieu.
        await interaction.response.send_modal(
            EventCompleteModal(
                self, ctx, suggestion_id,
                name=sugg.payload.get("title", sugg.description),
                when_raw=sugg.payload.get("when", ""),
            )
        )

    async def finish_event_from_modal(
        self, interaction: discord.Interaction, suggestion_id: int,
        *, name: str, when_text: str, location: str, ctx: ViewContext,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        guild = interaction.guild
        if not guild:
            return await interaction.followup.send("Action serveur uniquement.", ephemeral=True)
        sugg = self.store.get(suggestion_id)
        if not sugg or sugg.status != "pending" or sugg.guild_id != guild.id:
            return await interaction.followup.send("Suggestion déjà traitée.", ephemeral=True)

        start = await self._nl_to_datetime(when_text)
        if start is None or start <= datetime.now(timezone.utc):
            return await interaction.followup.send(
                "Je n'ai pas réussi à comprendre la date. Réessaie (ex : « samedi 20h »).",
                ephemeral=True,
            )
        end = start + timedelta(hours=EVENT_DEFAULT_DURATION_HOURS)

        try:
            event = await guild.create_scheduled_event(
                name=name[:100] or "Événement",
                start_time=start,
                end_time=end,
                entity_type=discord.EntityType.external,
                privacy_level=discord.PrivacyLevel.guild_only,
                location=(location or "À préciser")[:100],
                description=(sugg.source_excerpt or "")[:1000] or None,
            )
        except discord.Forbidden:
            return await interaction.followup.send(
                "Je n'ai pas la permission **Gérer les événements** sur ce serveur.", ephemeral=True
            )
        except discord.HTTPException as e:
            return await interaction.followup.send(f"Création de l'événement impossible : `{e}`", ephemeral=True)

        self.store.set_status(suggestion_id, "accepted")
        await interaction.followup.send(
            view=build_suggestions_view(self, ctx, header=f"✅ Événement créé : **{event.name}** — {event.url}"),
            ephemeral=True,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Watcher(bot))
