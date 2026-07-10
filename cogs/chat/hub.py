"""Hub personnel — vue LayoutView pour /hub."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from discord.ext import commands

from cogs.chat.config import MODEL_NANO
from common.hub import UserHubStore, hashtag, parse_topics, truncate_lines
from common.rappels import RECURRENCE_NONE, VALID_RECURRENCES, Rappel, RappelStore
from common.timezones import PARIS_TZ, format_french_date
from common.news import brave_news, build_news_summary

try:
    from cogs.meteo.meteo import _emoji as _weather_emoji
except ImportError:
    def _weather_emoji(icon: str) -> str:
        return "🌡️"

logger = logging.getLogger("MARIA.Hub")

REMINDER_MAX_PENDING = 10

_DATETIME_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "datetime_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "execute_at": {
                    "type": ["string", "null"],
                    "description": "Date/heure ISO 8601 (Europe/Paris, sans décalage) ou null si indéterminable",
                },
                "recurrence": {
                    "type": "string",
                    "enum": list(VALID_RECURRENCES),
                    "description": (
                        "'daily' si répété chaque jour ('tous les jours à 8h', 'chaque matin'…), "
                        "'weekly' si répété chaque semaine ('tous les lundis'…), sinon 'none'."
                    ),
                },
            },
            "required": ["execute_at", "recurrence"],
            "additionalProperties": False,
        },
    },
}


async def _parse_natural_datetime(bot: commands.Bot, text: str) -> tuple[Optional[datetime], str]:
    """Interprète une date/heure + récurrence en langage naturel via le modèle nano.

    Ex: 'demain 18h' → (dt, 'none'). 'tous les jours à 8h' → (dt, 'daily').
    """
    chat_cog = bot.get_cog("Chat")
    if chat_cog is None or not hasattr(chat_cog, "gpt_api"):
        return None, RECURRENCE_NONE
    now_str = datetime.now(PARIS_TZ).strftime("%A %d/%m/%Y %H:%M")
    messages = [
        {
            "role": "system",
            "content": (
                f"Nous sommes le {now_str} (fuseau Europe/Paris). Convertis la date/heure donnée par "
                "l'utilisateur en ISO 8601 local (Europe/Paris, sans décalage, ex. '2026-08-15T18:00:00') "
                "pour son PREMIER déclenchement, et détecte une éventuelle récurrence. "
                "Si aucune heure n'est précisée, utilise 09:00. Si la date/heure est indéterminable, "
                "renvoie execute_at=null."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        completion = await chat_cog.gpt_api.client.chat(
            messages, model=MODEL_NANO, response_format=_DATETIME_SCHEMA,
        )
        raw = json.loads(completion.choices[0].message.content or "{}")
        execute_at_str = raw.get("execute_at")
        recurrence = raw.get("recurrence") or RECURRENCE_NONE
        if recurrence not in VALID_RECURRENCES:
            recurrence = RECURRENCE_NONE
        if not execute_at_str:
            return None, recurrence
        dt = datetime.fromisoformat(execute_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=PARIS_TZ)
        return dt.astimezone(timezone.utc), recurrence
    except Exception as e:
        logger.warning(f"Parsing date rappel échoué ({text!r}): {e}")
        return None, RECURRENCE_NONE


@dataclass
class HubDisplayData:
    config_first_name: str = ""
    config_city: str = ""
    config_topics: list[str] = field(default_factory=list)
    weather_line: str = ""
    reminders: list[Rappel] = field(default_factory=list)
    news_text: str = ""
    is_empty: bool = True
    tz_offset: Optional[int] = None  # secondes par rapport à UTC, déduit de la ville


def _local_hour(tz_offset: Optional[int]) -> int:
    """Heure locale déduite de l'offset UTC de la ville (ou Paris par défaut)."""
    if tz_offset is None:
        return datetime.now(PARIS_TZ).hour
    return (datetime.now(timezone.utc) + timedelta(seconds=tz_offset)).hour


def _greeting_title(first_name: str, tz_offset: Optional[int] = None) -> str:
    """'Hub · Bonjour/Bonsoir Prénom' selon l'heure locale, ou titre par défaut si pas de prénom."""
    if not first_name:
        return "Ton hub"
    hour = _local_hour(tz_offset)
    greeting = "Bonjour" if 5 <= hour < 18 else "Bonsoir"
    return f"Hub · {greeting} {first_name}"


def _format_weather(city: str, raw: dict) -> str:
    if "error" in raw:
        return f"-# Météo indisponible : {raw['error']}"
    main = raw.get("main", {})
    temp = round(main.get("temp", 0))
    feels = round(main.get("feels_like", temp))
    weather = (raw.get("weather") or [{}])[0]
    desc = weather.get("description", "")
    icon = weather.get("icon", "01d")
    city_name = raw.get("name") or city
    return f"{_weather_emoji(icon)} **{city_name}** · {temp}°C (ressenti {feels}°C) · {desc}"


def _format_rappel_line(r: Rappel) -> str:
    ts = int(r.execute_at.timestamp())
    desc = r.description[:80] + ("…" if len(r.description) > 80 else "")
    rec = {
        "daily": " <:repeat:1525261027883745342>",
        "weekly": " <:repeat:1525261027883745342>",
    }.get(r.recurrence, "")
    return f"› **Rappel #{r.id}**{rec} · <t:{ts}:R> — {desc}"


async def _fetch_weather(bot: commands.Bot, city: str) -> tuple[str, Optional[int]]:
    """Renvoie (ligne météo formatée, offset UTC en secondes déduit de la ville)."""
    meteo = bot.get_cog("Meteo")
    if meteo is None or not hasattr(meteo, "_fetch_current"):
        return "-# Météo indisponible (cog Meteo absent).", None
    raw = await asyncio.to_thread(meteo._fetch_current, city)
    tz_offset = raw.get("timezone") if "error" not in raw else None
    return _format_weather(city, raw), tz_offset


async def _fetch_news(
    hub_store: UserHubStore, user_id: int, topics: list[str], brave_key: str,
) -> str:
    if not topics:
        return ""
    if not brave_key:
        return "-# Actu indisponible (clé Brave manquante)."

    date_str = datetime.now(PARIS_TZ).strftime("%d/%m/%Y")
    tasks = [asyncio.to_thread(brave_news, brave_key, topic, 3) for topic in topics[:3]]
    results_lists = await asyncio.gather(*tasks)
    all_results: list[dict] = []
    for res in results_lists:
        all_results.extend(res)

    summary = build_news_summary(all_results, date_str)
    if not summary:
        return "-# Aucune actu trouvée pour tes sujets."

    display = summary.split(":\n", 1)[-1] if ":\n" in summary else summary
    hub_store.set_news_cache(user_id, display)
    return display


async def fetch_hub_data(
    bot: commands.Bot,
    user_id: int,
    hub_store: UserHubStore,
    rappels: RappelStore,
    *,
    brave_key: str = "",
    refresh_news: bool = False,
) -> HubDisplayData:
    config = hub_store.get(user_id)
    data = HubDisplayData(
        config_first_name=config.first_name,
        config_city=config.city,
        config_topics=list(config.topics),
        is_empty=config.is_empty,
    )

    data.reminders = rappels.get_user_rappels(user_id)

    fetch_tasks: list[tuple[str, object]] = []
    if config.city:
        fetch_tasks.append(("weather", _fetch_weather(bot, config.city)))
    if config.topics and (refresh_news or hub_store.is_news_stale(user_id, config)):
        fetch_tasks.append(("news", _fetch_news(hub_store, user_id, config.topics, brave_key)))
    elif config.topics and config.news_cache.get("summary"):
        data.news_text = config.news_cache["summary"]

    if fetch_tasks:
        results = await asyncio.gather(*(t[1] for t in fetch_tasks), return_exceptions=True)
        for (name, _), result in zip(fetch_tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"Hub fetch {name} failed: {result}")
                continue
            if name == "weather":
                data.weather_line, data.tz_offset = result
            elif name == "news":
                data.news_text = result

    return data


class ConfigureHubModal(discord.ui.Modal, title="Configurer ton hub"):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
        channel_id: int,
        first_name: str,
        city: str,
        topics_str: str,
        *,
        brave_key: str = "",
    ):
        super().__init__()
        self.bot = bot
        self.hub_store = hub_store
        self.rappels = rappels
        self.user_id = user_id
        self.channel_id = channel_id
        self.brave_key = brave_key
        self.first_name_input = discord.ui.TextInput(
            label="Prénom",
            placeholder="Ex: Alex",
            default=first_name[:50],
            max_length=50,
            required=False,
        )
        self.city_input = discord.ui.TextInput(
            label="Ville",
            placeholder="Ex: Lyon, Paris, Marseille…",
            default=city[:100],
            max_length=100,
            required=False,
        )
        self.topics_input = discord.ui.TextInput(
            label="Sujets d'actu (max 3, virgules ou #)",
            placeholder="Ex: tech, cinéma, PSG",
            default=topics_str[:200],
            max_length=200,
            required=False,
        )
        self.add_item(self.first_name_input)
        self.add_item(self.city_input)
        self.add_item(self.topics_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        self.hub_store.update(
            self.user_id,
            first_name=self.first_name_input.value.strip(),
            city=self.city_input.value.strip(),
            topics=parse_topics(self.topics_input.value),
        )
        # Interaction issue d'un composant (le bouton "Configurer") : defer() sans
        # `thinking=True` produit une DEFERRED_UPDATE_MESSAGE, qui cible le message
        # d'origine (celui qui portait le bouton) — pas une nouvelle réponse ephémère.
        await interaction.response.defer()
        data = await fetch_hub_data(
            self.bot, self.user_id, self.hub_store, self.rappels,
            brave_key=self.brave_key, refresh_news=True,
        )
        view = build_me_hub_layout(
            data, self.hub_store, self.user_id, self.bot, self.rappels, self.brave_key,
            channel_id=self.channel_id,
        )
        await interaction.edit_original_response(view=view)


class _ConfigureHubButton(discord.ui.Button):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
        channel_id: int,
        first_name: str,
        city: str,
        topics_str: str,
        *,
        brave_key: str = "",
    ):
        super().__init__(label="Configurer", style=discord.ButtonStyle.primary)
        self.bot = bot
        self.hub_store = hub_store
        self.rappels = rappels
        self.user_id = user_id
        self.channel_id = channel_id
        self.first_name = first_name
        self.city = city
        self.topics_str = topics_str
        self.brave_key = brave_key

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        await interaction.response.send_modal(
            ConfigureHubModal(
                self.bot, self.hub_store, self.rappels, self.user_id, self.channel_id,
                self.first_name, self.city, self.topics_str, brave_key=self.brave_key,
            )
        )


class AddRappelModal(discord.ui.Modal, title="Ajouter un rappel"):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
        channel_id: int,
        *,
        brave_key: str = "",
    ):
        super().__init__()
        self.bot = bot
        self.hub_store = hub_store
        self.rappels = rappels
        self.user_id = user_id
        self.channel_id = channel_id
        self.brave_key = brave_key
        self.desc_input = discord.ui.TextInput(
            label="Rappel",
            placeholder="Ex: Anniversaire de Sam, appeler le médecin…",
            max_length=200,
        )
        self.when_input = discord.ui.TextInput(
            label="Quand ?",
            placeholder="Ex: demain 18h, tous les jours à 8h, tous les lundis midi",
            max_length=60,
        )
        self.add_item(self.desc_input)
        self.add_item(self.when_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        if self.rappels.count_pending(self.user_id) >= REMINDER_MAX_PENDING:
            return await interaction.response.send_message(
                f"Max {REMINDER_MAX_PENDING} rappels en attente.", ephemeral=True,
            )
        await interaction.response.defer()
        execute_at, recurrence = await _parse_natural_datetime(self.bot, self.when_input.value)
        if not execute_at:
            return await interaction.followup.send(
                "Date pas comprise. Essaie « demain 18h », « lundi prochain » ou « 2026-08-15 ».",
                ephemeral=True,
            )
        self.rappels.add(
            self.channel_id, self.user_id, self.desc_input.value.strip(), execute_at,
            recurrence=recurrence,
        )
        data = await fetch_hub_data(self.bot, self.user_id, self.hub_store, self.rappels, brave_key=self.brave_key)
        view = build_me_hub_layout(
            data, self.hub_store, self.user_id, self.bot, self.rappels, self.brave_key,
            channel_id=self.channel_id,
        )
        await interaction.edit_original_response(view=view)


class _AddRappelButton(discord.ui.Button):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
        channel_id: int,
        *,
        brave_key: str = "",
    ):
        super().__init__(label="+ Rappel", style=discord.ButtonStyle.secondary)
        self.bot = bot
        self.hub_store = hub_store
        self.rappels = rappels
        self.user_id = user_id
        self.channel_id = channel_id
        self.brave_key = brave_key

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        await interaction.response.send_modal(
            AddRappelModal(
                self.bot, self.hub_store, self.rappels, self.user_id, self.channel_id, brave_key=self.brave_key,
            )
        )


class _CancelRappelButton(discord.ui.Button):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
        rappel_id: int,
        channel_id: int,
        *,
        brave_key: str = "",
    ):
        super().__init__(label="✕", style=discord.ButtonStyle.secondary)
        self.bot = bot
        self.hub_store = hub_store
        self.rappels = rappels
        self.user_id = user_id
        self.rappel_id = rappel_id
        self.channel_id = channel_id
        self.brave_key = brave_key

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        self.rappels.cancel(self.rappel_id, self.user_id)
        await interaction.response.defer()
        data = await fetch_hub_data(self.bot, self.user_id, self.hub_store, self.rappels, brave_key=self.brave_key)
        view = build_me_hub_layout(
            data, self.hub_store, self.user_id, self.bot, self.rappels, self.brave_key,
            channel_id=self.channel_id,
        )
        await interaction.edit_original_response(view=view)


def build_me_hub_layout(
    data: HubDisplayData,
    hub_store: UserHubStore,
    user_id: int,
    bot: commands.Bot,
    rappels: RappelStore,
    brave_key: str = "",
    channel_id: int = 0,
) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=180)
    today_str = format_french_date(datetime.now(PARIS_TZ))
    title = _greeting_title(data.config_first_name, data.tz_offset)
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## <:hub:1525259996315652209> {title}"),
        discord.ui.TextDisplay(f"-# {today_str}"),
        discord.ui.Separator(),
    ]

    if data.is_empty and not data.reminders:
        children.append(discord.ui.TextDisplay(
            "-# Ajoute ta ville et tes sujets pour personnaliser ton hub."
        ))
        children.append(discord.ui.Separator())

    if data.weather_line:
        children.append(discord.ui.TextDisplay(f"### Météo\n{data.weather_line}"))
        children.append(discord.ui.Separator())

    # Agenda : tous les rappels (récurrents ou non), triés chronologiquement,
    # chacun avec son propre bouton d'annulation.
    reminders_sorted = sorted(data.reminders, key=lambda r: r.execute_at)
    children.append(discord.ui.TextDisplay("### Agenda"))
    for r in reminders_sorted[:6]:
        children.append(discord.ui.Section(
            discord.ui.TextDisplay(_format_rappel_line(r)),
            accessory=_CancelRappelButton(
                bot, hub_store, rappels, user_id, r.id, channel_id, brave_key=brave_key,
            ),
        ))
    if len(reminders_sorted) > 6:
        children.append(discord.ui.TextDisplay(f"-# +{len(reminders_sorted) - 6} autre(s)"))
    if not reminders_sorted:
        children.append(discord.ui.TextDisplay("-# Rien de prévu."))
    children.append(discord.ui.Section(
        discord.ui.TextDisplay("-# Ajoute un rappel à ton agenda."),
        accessory=_AddRappelButton(bot, hub_store, rappels, user_id, channel_id, brave_key=brave_key),
    ))
    children.append(discord.ui.Separator())

    if data.config_topics:
        topics_label = " ".join(hashtag(t) for t in data.config_topics)
        if data.news_text:
            news_display = truncate_lines(data.news_text, 900)
            children.append(discord.ui.TextDisplay(f"### Actu · {topics_label}\n{news_display}"))
        else:
            children.append(discord.ui.TextDisplay(f"### Actu · {topics_label}\n-# Chargement ou indisponible."))
        children.append(discord.ui.Separator())

    topics_str = ", ".join(data.config_topics)
    topics_display = " ".join(hashtag(t) for t in data.config_topics) if data.config_topics else "—"
    config_line = discord.ui.TextDisplay(
        f"-# Prénom : {data.config_first_name or '—'} · Ville : {data.config_city or '—'} · Sujets : {topics_display}"
    )
    config_section = discord.ui.Section(
        config_line,
        accessory=_ConfigureHubButton(
            bot, hub_store, rappels, user_id, channel_id,
            data.config_first_name, data.config_city, topics_str,
            brave_key=brave_key,
        ),
    )
    children.append(config_section)

    view.add_item(discord.ui.Container(*children))
    return view


async def build_me_hub_view(
    bot: commands.Bot,
    user_id: int,
    hub_store: UserHubStore,
    rappels: RappelStore,
    *,
    brave_key: str = "",
    refresh_news: bool = False,
    channel_id: int = 0,
) -> discord.ui.LayoutView:
    data = await fetch_hub_data(
        bot, user_id, hub_store, rappels,
        brave_key=brave_key,
        refresh_news=refresh_news,
    )
    return build_me_hub_layout(data, hub_store, user_id, bot, rappels, brave_key, channel_id=channel_id)


async def show_me_hub(
    interaction: discord.Interaction,
    hub_store: UserHubStore,
    rappels: RappelStore,
    bot: commands.Bot,
    *,
    brave_key: str = "",
) -> None:
    await interaction.response.defer(ephemeral=True)
    view = await build_me_hub_view(
        bot, interaction.user.id, hub_store, rappels, brave_key=brave_key,
        channel_id=interaction.channel_id or 0,
    )
    await interaction.followup.send(view=view, ephemeral=True)
