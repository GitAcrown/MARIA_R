"""Hub personnel utilisateur — config structurée (ville, sujets, cache actu)."""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from common.dataio import CogData, DictTableBuilder
from common.timezones import PARIS_TZ

MAX_TOPICS = 3
MAX_AGENDA_EVENTS = 20
NEWS_STALE_AFTER = timedelta(hours=20)


def hashtag(topic: str) -> str:
    """Formate un sujet en hashtag pour l'affichage : 'jeux vidéo' → '#jeuxvidéo'."""
    return "#" + re.sub(r"\s+", "", topic)


def truncate_lines(text: str, max_chars: int) -> str:
    """Tronque `text` en gardant des lignes entières (jamais au milieu d'un lien Markdown)."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    total = 0
    for line in lines:
        added = len(line) + (1 if kept else 0)
        if total + added > max_chars:
            break
        kept.append(line)
        total += added
    if not kept:
        return text[:max_chars] + "…"
    return "\n".join(kept) + "…"


@dataclass
class UserHubConfig:
    first_name: str = ""
    city: str = ""
    topics: list[str] = field(default_factory=list)
    news_cache: dict = field(default_factory=dict)
    agenda: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.first_name and not self.city and not self.topics and not self.agenda

    def prompt_line(self) -> str:
        """Ligne succincte pour injection dans le prompt."""
        parts: list[str] = []
        if self.first_name:
            parts.append(f"prénom : {self.first_name}")
        if self.city:
            parts.append(f"ville : {self.city}")
        if self.topics:
            parts.append("sujets : " + ", ".join(self.topics))
        return " · ".join(parts)


def parse_topics(raw: str) -> list[str]:
    """Parse des sujets séparés par virgules et/ou préfixés #.

    Chaque `#` démarre un nouveau sujet, même sans virgule ('#tech #cinéma' → 2 sujets).
    """
    topics: list[str] = []
    for chunk in re.split(r"[,#]+", raw):
        t = chunk.strip()
        if t and t.lower() not in {x.lower() for x in topics}:
            topics.append(t)
        if len(topics) >= MAX_TOPICS:
            break
    return topics


class UserHubStore:
    """Stockage du hub personnel par user_id Discord."""

    def __init__(self) -> None:
        self._data = CogData("chat")
        self._data.set_builders(
            "global",
            DictTableBuilder("user_profiles"),
        )
        self._db = self._data.get("global")

    def _key(self, user_id: int) -> str:
        return f"hub_{user_id}"

    def get(self, user_id: int) -> UserHubConfig:
        raw = self._db.settings("user_profiles").get(self._key(user_id), default="") or ""
        if not raw:
            return UserHubConfig()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return UserHubConfig()
        return UserHubConfig(
            first_name=(data.get("first_name") or "").strip(),
            city=(data.get("city") or "").strip(),
            topics=list(data.get("topics") or [])[:MAX_TOPICS],
            news_cache=dict(data.get("news_cache") or {}),
            agenda=list(data.get("agenda") or [])[:MAX_AGENDA_EVENTS],
        )

    def save(self, user_id: int, config: UserHubConfig) -> None:
        payload = {
            "first_name": config.first_name.strip(),
            "city": config.city.strip(),
            "topics": config.topics[:MAX_TOPICS],
            "news_cache": config.news_cache,
            "agenda": config.agenda[:MAX_AGENDA_EVENTS],
        }
        self._db.settings("user_profiles").set(self._key(user_id), json.dumps(payload, ensure_ascii=False))

    def update(
        self,
        user_id: int,
        *,
        first_name: Optional[str] = None,
        city: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> UserHubConfig:
        config = self.get(user_id)
        if first_name is not None:
            config.first_name = first_name.strip()
        if city is not None:
            config.city = city.strip()
        if topics is not None:
            config.topics = topics[:MAX_TOPICS]
        self.save(user_id, config)
        return config

    def is_news_stale(self, user_id: int, config: Optional[UserHubConfig] = None) -> bool:
        config = config if config is not None else self.get(user_id)
        if not config.topics:
            return False
        cache = config.news_cache
        if not cache.get("summary"):
            return True
        updated_str = cache.get("updated") or cache.get("date") or ""
        if not updated_str:
            return True
        try:
            updated = datetime.fromisoformat(updated_str)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - updated >= NEWS_STALE_AFTER
        except ValueError:
            return True

    def set_news_cache(self, user_id: int, summary: str) -> None:
        config = self.get(user_id)
        now = datetime.now(timezone.utc)
        config.news_cache = {
            "summary": truncate_lines(summary, 1200),
            "updated": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
        }
        self.save(user_id, config)

    def add_agenda_event(self, user_id: int, title: str, event_date: date) -> UserHubConfig:
        config = self.get(user_id)
        # On élague les événements passés à chaque ajout pour ne pas accumuler indéfiniment.
        config.agenda = [e for e in config.agenda if not _agenda_event_is_past(e)]
        config.agenda.append({
            "id": uuid.uuid4().hex[:8],
            "title": title.strip()[:100],
            "date": event_date.isoformat(),
        })
        config.agenda.sort(key=lambda e: e.get("date", ""))
        config.agenda = config.agenda[:MAX_AGENDA_EVENTS]
        self.save(user_id, config)
        return config

    def remove_agenda_event(self, user_id: int, event_id: str) -> UserHubConfig:
        config = self.get(user_id)
        config.agenda = [e for e in config.agenda if e.get("id") != event_id]
        self.save(user_id, config)
        return config

    def get_upcoming_agenda(self, user_id: int, limit: int = 3) -> list[dict]:
        config = self.get(user_id)
        upcoming = [e for e in config.agenda if not _agenda_event_is_past(e)]
        upcoming.sort(key=lambda e: e.get("date", ""))
        return upcoming[:limit]


def _agenda_event_is_past(event: dict) -> bool:
    try:
        return date.fromisoformat(event.get("date", "")) < datetime.now(PARIS_TZ).date()
    except ValueError:
        return False
