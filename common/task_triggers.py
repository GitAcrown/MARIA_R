"""Triggers Discord event + cache + gate JEV pour tâches kind=event."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from common.task_recipe import (
    looks_like_noise,
    match_message_pattern,
    parse_trigger,
    trigger_terms,
)
from common.tasks import (
    EVENT_COOLDOWN_DEFAULT,
    FIRE_STORM_MAX_PER_HOUR,
    KIND_EVENT,
    SCOPE_GUILD,
    STATUS_ARMED,
    TaskStore,
)

if TYPE_CHECKING:
    import discord
    from common.llm.typesafe_client import MariaTypeSafeClient

logger = logging.getLogger("MARIA.TaskTriggers")

_CACHE_TTL = 30.0
JEV_FIRE_MIN = 0.55


class EventTriggerCache:
    """Cache court des tâches event armées par guild (jamais de SQLite sur la boucle async)."""

    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self._by_guild: dict[int, tuple[float, list]] = {}

    def invalidate(self, guild_id: int | None = None) -> None:
        if guild_id is None:
            self._by_guild.clear()
        else:
            self._by_guild.pop(guild_id, None)

    def _load(self, guild_id: int) -> list:
        now_dt = datetime.now(timezone.utc)
        return [
            t for t in self.store.list_armed_events(guild_id)
            if t.status == STATUS_ARMED
            and (t.expires_at is None or t.expires_at > now_dt)
        ]

    async def armed(self, guild_id: int) -> list:
        now = time.monotonic()
        hit = self._by_guild.get(guild_id)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]
        tasks = await asyncio.to_thread(self._load, guild_id)
        self._by_guild[guild_id] = (now, tasks)
        return tasks


def _channel_allowed(task, channel_id: int) -> bool:
    if task.scope == SCOPE_GUILD:
        return True
    return channel_id in task.channel_ids


def _author_allowed(task, author_id: int) -> bool:
    trig = parse_trigger(task.trigger_json)
    mode = trig.get("author") or "not_self"
    if mode == "any":
        return True
    if mode == "not_self":
        return author_id != task.user_id
    try:
        return author_id == int(mode)
    except (TypeError, ValueError):
        return author_id != task.user_id


def cheap_match(task, message: "discord.Message") -> bool:
    if task.kind != KIND_EVENT:
        return False
    if not _channel_allowed(task, message.channel.id):
        return False
    if not _author_allowed(task, message.author.id):
        return False
    terms = trigger_terms(parse_trigger(task.trigger_json))
    if not terms:
        return False
    text = message.clean_content or message.content or ""
    return any(match_message_pattern(text, term, whole_word=True) for term in terms)


def cooldown_ok(task) -> bool:
    cd = int(task.cooldown_seconds or EVENT_COOLDOWN_DEFAULT)
    if cd <= 0 or task.last_fired_at is None:
        return True
    elapsed = (datetime.now(timezone.utc) - task.last_fired_at).total_seconds()
    return elapsed >= cd


async def jev_should_fire(
    typesafe: Optional["MariaTypeSafeClient"],
    *,
    pattern: str,
    message: str,
    intent: str = "",
) -> bool:
    """Gate anti-bruit. Sans JEV → True (fail-open : le match cheap a déjà filtré).

    `intent` = ce que le membre voulait vraiment (« préviens-moi quand ça joue ranked ») :
    JEV juge le sens, pas seulement la présence du mot.
    """
    if typesafe is None or not getattr(typesafe, "enabled", False):
        return True
    from typesafe_sdk import Noul

    result = await typesafe.system_one(
        {
            "pattern": (pattern or "")[:40],
            "intent": (intent or "")[:200],
            "message": (message or "")[:400],
        },
        {
            "fire": Noul(
                instructions=(
                    "A Discord bot watches keyword `pattern` for a member whose goal is "
                    "`intent` (may be empty). `message` just contained the keyword. "
                    "Should the bot alert that member now? "
                    "true only if the message really matches the member's goal "
                    "(someone actually talking about / proposing the topic). "
                    "false for passing mentions, jokes listing words, quoting the watch itself, "
                    "or an unrelated sense of the word."
                ),
                criteria={
                    "true": "Genuinely relevant to the member's goal; alert is useful",
                    "false": "Noise, unrelated sense, list, or meta about the watch",
                },
            ),
        },
    )
    if result is None:
        return True
    try:
        return float(result.nouls["fire"].noul) >= JEV_FIRE_MIN
    except (KeyError, AttributeError, TypeError, ValueError):
        return True


# --- Détection sémantique (JEV) : garde-fous de coût -------------------------
SEMANTIC_MIN_CHARS = 12
SEMANTIC_CHANNEL_GAP = 4.0       # s min entre deux analyses dans un même salon
SEMANTIC_GUILD_DAILY = 400       # analyses JEV max / serveur / jour
_last_scan: dict[int, float] = {}
_guild_scans: dict[int, tuple[str, int]] = {}


def _semantic_gate(message: "discord.Message") -> bool:
    """Filtres gratuits avant tout appel JEV sémantique."""
    text = (message.clean_content or message.content or "").strip()
    if len(text) < SEMANTIC_MIN_CHARS or looks_like_noise(text):
        return False
    now = time.monotonic()
    chan = message.channel.id
    if now - _last_scan.get(chan, 0.0) < SEMANTIC_CHANNEL_GAP:
        return False
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    gid = message.guild.id
    cur_day, used = _guild_scans.get(gid, (day, 0))
    if cur_day != day:
        used = 0
    if used >= SEMANTIC_GUILD_DAILY:
        return False
    _last_scan[chan] = now
    _guild_scans[gid] = (day, used + 1)
    if len(_last_scan) > 2000:
        for key in sorted(_last_scan, key=_last_scan.get)[:1000]:
            _last_scan.pop(key, None)
    return True


def _in_scope(task, message: "discord.Message") -> bool:
    return (
        task.kind == KIND_EVENT
        and _channel_allowed(task, message.channel.id)
        and _author_allowed(task, message.author.id)
    )


async def _budget_ok(store: TaskStore, task) -> bool:
    if not cooldown_ok(task):
        return False
    if task.max_fires and task.fires_count >= task.max_fires:
        return False
    storm = await asyncio.to_thread(store.fires_last_hour, task.user_id)
    return storm < FIRE_STORM_MAX_PER_HOUR


async def find_firing_tasks(
    cache: EventTriggerCache,
    store: TaskStore,
    message: "discord.Message",
    typesafe: Optional["MariaTypeSafeClient"],
) -> list:
    """Tâches event qui doivent se déclencher pour ce message.

    1. mots-clés (gratuit) puis confirmation JEV du sens ;
    2. écoutes « par sujet » : un seul appel JEV groupé pour tout le salon.
    """
    if not message.guild or message.author.bot:
        return []
    armed = await cache.armed(message.guild.id)
    scoped = [t for t in armed if _in_scope(t, message)]
    if not scoped:
        return []
    text = message.clean_content or message.content or ""
    out: list = []
    done: set[int] = set()

    # 1) mots-clés
    for task in scoped:
        if not cheap_match(task, message):
            continue
        done.add(task.id)
        if not await _budget_ok(store, task):
            continue
        trig = parse_trigger(task.trigger_json)
        # Expression de 2+ mots (« game night ») : déjà assez précise, pas besoin de JEV.
        if any(
            " " in term.strip() and match_message_pattern(text, term, whole_word=True)
            for term in trigger_terms(trig)
        ):
            out.append(task)
            continue
        intent = trig.get("topic") or trig.get("intent") or task.instruction or ""
        if await jev_should_fire(
            typesafe,
            pattern=" / ".join(trigger_terms(trig)),
            message=text,
            intent=intent,
        ):
            out.append(task)

    # 2) sujets compris par JEV (nécessite JEV actif)
    if typesafe is None or not getattr(typesafe, "enabled", False):
        return out
    sem = []
    for task in scoped:
        if task.id in done:
            continue
        if not parse_trigger(task.trigger_json).get("topic"):
            continue
        if await _budget_ok(store, task):
            sem.append(task)
    if sem and _semantic_gate(message):
        topics = {t.id: str(parse_trigger(t.trigger_json).get("topic")) for t in sem}
        hits = await typesafe.match_topics(text, topics)
        out.extend(t for t in sem if t.id in hits)
    return out
