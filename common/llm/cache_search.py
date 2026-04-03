"""IA secondaire gpt-5.4-nano — recherche/compilation de messages hors contexte."""

import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

from openai import AsyncOpenAI

from .client import MODEL_NANO

logger = logging.getLogger("llm.cache_search")

CACHE_SIZE = 500
CACHE_MAX_AGE_HOURS = 48

PROMPT_TEMPLATE = """Requête sur les messages passés d'un salon Discord : {query}

MESSAGES (chronologique — [heure] auteur [→ @cible si reply] : contenu) :
{messages}

Synthèse concise des échanges pertinents.
Indique qui dit quoi, et à qui si c'est une réponse directe. Faits bruts, sans introduction ni conclusion."""


class MessageCache:
    """Cache de messages hors contexte pour la nano."""

    def __init__(self, max_size: int = CACHE_SIZE, max_age_hours: float = CACHE_MAX_AGE_HOURS):
        self._by_channel: dict[int, deque] = {}
        self._max_size = max_size
        self._max_age = timedelta(hours=max_age_hours)

    def push(
        self,
        channel_id: int,
        author: str,
        content: str,
        created_at: datetime,
        *,
        reply_to: Optional[str] = None,
    ) -> None:
        if channel_id not in self._by_channel:
            self._by_channel[channel_id] = deque(maxlen=self._max_size)
        self._by_channel[channel_id].append({
            "author": author,
            "content": content[:500],
            "created_at": created_at,
            "reply_to": reply_to,
        })

    def get_recent(self, channel_id: int, count: int = 50) -> list[dict]:
        now = datetime.now(timezone.utc)
        cut = now - self._max_age
        if channel_id not in self._by_channel:
            return []
        out = []
        for m in list(self._by_channel[channel_id])[-count:]:
            if m["created_at"] > cut:
                out.append(m)
        return out

    @staticmethod
    def format_for_prompt(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            ts = m.get("created_at")
            ts_str = ts.strftime("%H:%M") if ts else ""
            reply = m.get("reply_to")
            reply_str = f" → @{reply}" if reply else ""
            lines.append(f"[{ts_str}] {m.get('author', '?')}{reply_str}: {m.get('content', '')}")
        return "\n".join(lines) if lines else "(aucun message)"


class CacheSearchClient:
    """Client nano pour compiler le cache."""

    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    async def search(
        self,
        query: str,
        messages: list[dict],
    ) -> Optional[str]:
        """Compile les messages pertinents pour la requête."""
        if not messages:
            return None
        formatted = MessageCache.format_for_prompt(messages)
        prompt = PROMPT_TEMPLATE.format(query=query, messages=formatted)
        try:
            r = await self._client.chat.completions.create(
                model=MODEL_NANO,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600,
            )
            text = r.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"CacheSearch: {e}")
            return None
