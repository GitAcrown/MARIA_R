"""Worker mémoire — buffer hybride, flush nano, decay quotidien."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from discord.ext import tasks

from common.memory.agent import extract_memories, parse_user_id
from common.memory.store import (
    CATEGORY_EVENT,
    CATEGORY_SERVER,
    CONFIDENCE_CREATE,
    CONFIDENCE_PENDING,
    CONFIDENCE_UPDATE_DELTA,
    STATUS_ACTIVE,
    STATUS_PENDING,
    VALID_CATEGORIES,
    MemoryStore,
)
from common.memory.vector import VectorStore

logger = logging.getLogger("MARIA.Memory.Worker")


@dataclass
class BufferedMessage:
    author_id: int
    author_name: str
    content: str
    ts: datetime
    reply_to_id: Optional[int] = None
    reply_to_name: Optional[str] = None
    reply_to_content: Optional[str] = None

    def format_line(self) -> str:
        """Ligne lisible pour l'agent nano, avec contexte de reply si présent."""
        stamp = self.ts.strftime("%H:%M")
        head = f"[{stamp}] {self.author_name} ({self.author_id})"
        if self.reply_to_id is not None:
            snippet = (self.reply_to_content or "").replace("\n", " ").strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            head += (
                f" [répond à {self.reply_to_name or '?'} ({self.reply_to_id})"
                f": \"{snippet}\"]"
            )
        return f"{head}: {self.content}"


@dataclass
class ChannelBuffer:
    messages: list[BufferedMessage] = field(default_factory=list)
    flushing: bool = False


class MemoryWorker:
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        llm_client: Any,
        *,
        model: str,
        flush_messages: int = 20,
        flush_minutes: int = 15,
        buffer_cap: int = 40,
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.llm_client = llm_client
        self.model = model
        self.flush_messages = flush_messages
        self.flush_minutes = flush_minutes
        self.buffer_cap = buffer_cap
        self._buffers: dict[tuple[int, int], ChannelBuffer] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self._decay_loop.is_running():
            self._decay_loop.start()
        if not self._timeout_loop.is_running():
            self._timeout_loop.start()

    async def stop(self) -> None:
        if self._decay_loop.is_running():
            self._decay_loop.cancel()
        if self._timeout_loop.is_running():
            self._timeout_loop.cancel()

    def ingest(
        self,
        *,
        guild_id: int,
        channel_id: int,
        author_id: int,
        author_name: str,
        content: str,
        reply_to_id: Optional[int] = None,
        reply_to_name: Optional[str] = None,
        reply_to_content: Optional[str] = None,
    ) -> None:
        text = (content or "").strip()
        if not text:
            return
        key = (guild_id, channel_id)
        buf = self._buffers.setdefault(key, ChannelBuffer())
        buf.messages.append(
            BufferedMessage(
                author_id=author_id,
                author_name=author_name,
                content=text[:500],
                ts=datetime.now(timezone.utc),
                reply_to_id=reply_to_id,
                reply_to_name=reply_to_name,
                reply_to_content=(reply_to_content or "")[:200] or None,
            )
        )
        if len(buf.messages) > self.buffer_cap:
            buf.messages = buf.messages[-self.buffer_cap :]

        if len(buf.messages) >= self.flush_messages and not buf.flushing:
            asyncio.create_task(self._flush_key(key))

    def _should_flush_timeout(self, buf: ChannelBuffer) -> bool:
        if not buf.messages or buf.flushing:
            return False
        oldest = buf.messages[0].ts
        return datetime.now(timezone.utc) - oldest >= timedelta(minutes=self.flush_minutes)

    async def _flush_key(self, key: tuple[int, int]) -> None:
        async with self._lock:
            buf = self._buffers.get(key)
            if buf is None or not buf.messages or buf.flushing:
                return
            buf.flushing = True
            batch = list(buf.messages)
            buf.messages.clear()

        guild_id, channel_id = key
        try:
            await self._process_batch(guild_id, batch)
        except Exception as e:
            logger.warning("Flush mémoire échoué (guild=%s ch=%s): %s", guild_id, channel_id, e)
        finally:
            async with self._lock:
                buf = self._buffers.get(key)
                if buf is not None:
                    buf.flushing = False

    async def _process_batch(self, guild_id: int, batch: list[BufferedMessage]) -> None:
        if not batch:
            return
        user_ids = {m.author_id for m in batch}
        for m in batch:
            if m.reply_to_id is not None:
                user_ids.add(m.reply_to_id)
        existing = self.store.list_for_users(guild_id, user_ids, limit=15)
        batch_text = "\n".join(m.format_line() for m in batch)
        actions = await extract_memories(
            self.llm_client,
            model=self.model,
            batch_text=batch_text,
            existing=existing,
        )
        if not actions:
            return
        for action in actions[:4]:
            await self._apply_action(guild_id, action)

    async def _apply_action(self, guild_id: int, action: dict) -> None:
        kind = (action.get("action") or "").strip()
        content = (action.get("content") or "").strip()
        category = (action.get("category") or "user").strip()
        if category not in VALID_CATEGORIES:
            category = "user"
        target_id = action.get("target_id") or None
        if isinstance(target_id, str) and target_id.lower() in ("null", ""):
            target_id = None
        # `user_id` peut être un membre mentionné mais pas auteur du batch — autorisé.
        user_id = parse_user_id(action.get("user_id"))

        if kind == "create":
            if not content:
                return
            # Sépare strictement perso / collectif.
            if category == "user":
                if user_id is None:
                    return
            elif category == "server":
                user_id = None
            # event : user_id optionnel
            # Perso : tampon pending. Collectif (server/event) : actif tout de suite
            # (petit serveur — on veut remplir /global sans attendre 2 hits).
            if category in (CATEGORY_SERVER, CATEGORY_EVENT):
                mem = self.store.create(
                    category=category,
                    guild_id=guild_id,
                    content=content,
                    user_id=user_id,
                    confidence=CONFIDENCE_CREATE,
                    status=STATUS_ACTIVE,
                )
                self.vectors.upsert(
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )
                logger.info("Mémoire serveur %s: %s", mem.id[:8], mem.content[:60])
                return
            mem = self.store.create(
                category=category,
                guild_id=guild_id,
                content=content,
                user_id=user_id,
                confidence=CONFIDENCE_PENDING,
                status=STATUS_PENDING,
            )
            logger.debug("Mémoire pending %s: %s", mem.id[:8], mem.content[:60])
            return

        if not target_id:
            return
        existing = self.store.get(target_id)
        if existing is None:
            return
        # Souvenirs user = globaux ; server/event restent scopés au guild.
        if existing.category != "user" and existing.guild_id != guild_id:
            return

        if kind in ("update", "merge"):
            was_pending = existing.status == STATUS_PENDING
            new_content = content or existing.content
            mem = self.store.update_content(
                target_id, new_content, confidence_delta=CONFIDENCE_UPDATE_DELTA,
            )
            if mem is None:
                return
            # Promotion pending → active : indexation Chroma seulement à ce moment.
            if mem.status == STATUS_ACTIVE and (was_pending or mem.chroma_id):
                self.vectors.upsert(
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )
            if was_pending and mem.status == STATUS_ACTIVE:
                logger.info("Mémoire promue %s (hits=%s): %s", mem.id[:8], mem.hits, mem.content[:60])
            return

        if kind == "contradict":
            mem = self.store.contradict(target_id)
            if mem is None:
                return
            if mem.status != STATUS_ACTIVE:
                self.vectors.delete(target_id)
            else:
                self.vectors.upsert(
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )

    @tasks.loop(hours=24)
    async def _decay_loop(self) -> None:
        try:
            archived = await asyncio.to_thread(self.store.apply_decay)
            for mid in archived:
                self.vectors.delete(mid)
        except Exception as e:
            logger.warning("Decay mémoire échoué: %s", e)

    @_decay_loop.before_loop
    async def _before_decay(self) -> None:
        await asyncio.sleep(60)

    @tasks.loop(minutes=2)
    async def _timeout_loop(self) -> None:
        """Flush les buffers qui ont dépassé MEMORY_FLUSH_MINUTES."""
        keys: list[tuple[int, int]] = []
        async with self._lock:
            for key, buf in self._buffers.items():
                if self._should_flush_timeout(buf):
                    keys.append(key)
        for key in keys:
            await self._flush_key(key)

    @_timeout_loop.before_loop
    async def _before_timeout(self) -> None:
        await asyncio.sleep(30)
