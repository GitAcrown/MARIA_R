"""Hub personnel — vue LayoutView pour /me."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

import discord
from discord.ext import commands

from common.hub import UserHubStore, hashtag, parse_topics
from common.rappels import Rappel, RappelStore
from common.timezones import PARIS_TZ, format_french_date
from common.news import brave_news, build_news_summary

try:
    from cogs.meteo.meteo import _emoji as _weather_emoji
except ImportError:
    def _weather_emoji(icon: str) -> str:
        return "🌡️"

logger = logging.getLogger("MARIA.Hub")


@dataclass
class HubDisplayData:
    config_city: str = ""
    config_topics: list[str] = field(default_factory=list)
    weather_line: str = ""
    reminders_lines: list[str] = field(default_factory=list)
    news_text: str = ""
    is_empty: bool = True


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


def _format_reminders(rappels: list[Rappel]) -> list[str]:
    lines: list[str] = []
    for r in rappels[:5]:
        ts = int(r.execute_at.timestamp())
        desc = r.description[:80] + ("…" if len(r.description) > 80 else "")
        rec = {
            "daily": " <:repeat:1525261027883745342>",
            "weekly": " <:repeat:1525261027883745342>",
        }.get(r.recurrence, "")
        lines.append(f"› **#{r.id}**{rec} · <t:{ts}:R> — {desc}")
    if len(rappels) > 5:
        lines.append(f"-# +{len(rappels) - 5} autre(s) rappel(s)")
    return lines


async def _fetch_weather(bot: commands.Bot, city: str) -> str:
    meteo = bot.get_cog("Meteo")
    if meteo is None or not hasattr(meteo, "_fetch_current"):
        return "-# Météo indisponible (cog Meteo absent)."
    raw = await asyncio.to_thread(meteo._fetch_current, city)
    return _format_weather(city, raw)


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
        config_city=config.city,
        config_topics=list(config.topics),
        is_empty=config.is_empty,
    )

    rappel_list = rappels.get_user_rappels(user_id)
    data.reminders_lines = _format_reminders(rappel_list)

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
                data.weather_line = result
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
        self.brave_key = brave_key
        self.city_input = discord.ui.TextInput(
            label="Ville",
            placeholder="Ex: Lyon, Paris, Marseille…",
            default=city[:100],
            max_length=100,
            required=False,
        )
        self.topics_input = discord.ui.TextInput(
            label="Sujets d'actu (virgules ou #)",
            placeholder="Ex: tech, cinéma, PSG",
            default=topics_str[:200],
            max_length=200,
            required=False,
        )
        self.add_item(self.city_input)
        self.add_item(self.topics_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        self.hub_store.update(
            self.user_id,
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
        view = build_me_hub_layout(data, self.hub_store, self.user_id, self.bot, self.rappels, self.brave_key)
        await interaction.edit_original_response(view=view)


class _ConfigureHubButton(discord.ui.Button):
    def __init__(
        self,
        bot: commands.Bot,
        hub_store: UserHubStore,
        rappels: RappelStore,
        user_id: int,
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
        self.city = city
        self.topics_str = topics_str
        self.brave_key = brave_key

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ton hub.", ephemeral=True)
        await interaction.response.send_modal(
            ConfigureHubModal(
                self.bot, self.hub_store, self.rappels, self.user_id,
                self.city, self.topics_str, brave_key=self.brave_key,
            )
        )


def build_me_hub_layout(
    data: HubDisplayData,
    hub_store: UserHubStore,
    user_id: int,
    bot: commands.Bot,
    rappels: RappelStore,
    brave_key: str = "",
) -> discord.ui.LayoutView:
    view = discord.ui.LayoutView(timeout=180)
    today_str = format_french_date(datetime.now(PARIS_TZ))
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay("## <:hub:1525259996315652209> Ton hub"),
        discord.ui.TextDisplay(f"-# {today_str}"),
        discord.ui.Separator(),
    ]

    if data.is_empty and not data.reminders_lines:
        children.append(discord.ui.TextDisplay(
            "-# Ajoute ta ville et tes sujets pour personnaliser ton hub."
        ))
        children.append(discord.ui.Separator())

    if data.weather_line:
        children.append(discord.ui.TextDisplay(f"### Météo\n{data.weather_line}"))
        children.append(discord.ui.Separator())

    if data.reminders_lines:
        body = "\n".join(data.reminders_lines)
        children.append(discord.ui.TextDisplay(f"### Rappels\n{body}"))
        children.append(discord.ui.Separator())
    elif not data.is_empty:
        children.append(discord.ui.TextDisplay("### Rappels\n-# Aucun rappel en attente."))
        children.append(discord.ui.Separator())

    if data.config_topics:
        topics_label = " ".join(hashtag(t) for t in data.config_topics)
        if data.news_text:
            news_display = data.news_text[:900] + ("…" if len(data.news_text) > 900 else "")
            children.append(discord.ui.TextDisplay(f"### Actu · {topics_label}\n{news_display}"))
        else:
            children.append(discord.ui.TextDisplay(f"### Actu · {topics_label}\n-# Chargement ou indisponible."))
        children.append(discord.ui.Separator())

    topics_str = ", ".join(data.config_topics)
    topics_display = " ".join(hashtag(t) for t in data.config_topics) if data.config_topics else "—"
    config_line = discord.ui.TextDisplay(
        f"-# Ville : {data.config_city or '—'} · Sujets : {topics_display}"
    )
    config_section = discord.ui.Section(
        config_line,
        accessory=_ConfigureHubButton(
            bot, hub_store, rappels, user_id, data.config_city, topics_str, brave_key=brave_key,
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
) -> discord.ui.LayoutView:
    data = await fetch_hub_data(
        bot, user_id, hub_store, rappels,
        brave_key=brave_key,
        refresh_news=refresh_news,
    )
    return build_me_hub_layout(data, hub_store, user_id, bot, rappels, brave_key)


async def show_me_hub(
    interaction: discord.Interaction,
    hub_store: UserHubStore,
    rappels: RappelStore,
    bot: commands.Bot,
    *,
    brave_key: str = "",
) -> None:
    await interaction.response.defer(ephemeral=True)
    view = await build_me_hub_view(bot, interaction.user.id, hub_store, rappels, brave_key=brave_key)
    await interaction.followup.send(view=view, ephemeral=True)
