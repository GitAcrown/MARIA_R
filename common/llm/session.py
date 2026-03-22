"""Session par salon — contexte restreint, lock, tools."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import discord

from .client import MariaLLMClient, MariaOpenAIError
from .context import (
    ConversationContext,
    MessageRecord,
    AssistantRecord,
    ToolCallRecord,
    ToolResponseRecord,
    TextComponent,
    ImageComponent,
    MetadataComponent,
)
from .tools import ToolRegistry
from .attachments import AttachmentCache, process_attachment
from .cache_search import MessageCache, CacheSearchClient

logger = logging.getLogger("llm.session")

USER_FORMAT = "{message.author.name}"
MAX_RECURSION = 8


def _embed_to_text(emb: discord.Embed) -> str:
    """Convertit un embed Discord en texte lisible pour le contexte."""
    lines: list[str] = []
    if emb.author and emb.author.name:
        lines.append(f"[{emb.author.name}]")
    if emb.title:
        title = emb.title
        if emb.url:
            title += f" ({emb.url})"
        lines.append(title)
    if emb.description:
        lines.append(emb.description[:500] + ("…" if len(emb.description) > 500 else ""))
    for field in emb.fields[:6]:
        if field.name and field.value:
            val = str(field.value)
            lines.append(f"{field.name}: {val[:200] + ('…' if len(val) > 200 else '')}")
    if emb.footer and emb.footer.text:
        lines.append(f"({emb.footer.text[:120]})")
    return "\n".join(lines)


class ChannelSession:
    """Session par salon."""

    def __init__(
        self,
        channel_id: int,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        attachment_cache: AttachmentCache,
        message_cache: MessageCache,
        cache_search: Optional[CacheSearchClient],
        developer_prompt_template: Callable[[], str],
        context_window: int = 8192,
        context_age_hours: float = 2,
    ):
        self.channel_id = channel_id
        self.client = client
        self.tool_registry = tool_registry
        self.attachment_cache = attachment_cache
        self.message_cache = message_cache
        self.cache_search = cache_search
        self.developer_prompt_template = developer_prompt_template
        self.context = ConversationContext(
            developer_prompt="",
            context_window=context_window,
            context_age=timedelta(hours=context_age_hours),
        )
        self._lock = asyncio.Lock()
        self.trigger_message: Optional[discord.Message] = None

    async def ingest_message(self, message: discord.Message, is_context_only: bool = False) -> MessageRecord:
        """Ingère un message.
        - is_context_only=True  → uniquement le cache nano (jamais dans la fenêtre principale)
        - is_context_only=False → fenêtre principale + cache nano
        """
        text = message.content or ""
        user_name = USER_FORMAT.format(message=message)

        # ---- Cache nano (tous les messages, toujours) ----
        cache_text = text.strip()
        if not cache_text and message.embeds:
            cache_text = _embed_to_text(message.embeds[0])[:200]
        if cache_text:
            created = message.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            self.message_cache.push(self.channel_id, user_name, cache_text, created)

        # ---- Messages contexte-seul : on s'arrête ici ----
        if is_context_only:
            return MessageRecord(
                role="user",
                components=[],
                created_at=datetime.now(timezone.utc),
                name=user_name,
            )

        # ---- Contexte principal (messages adressés au bot uniquement) ----
        parts: list[ContentComponent] = []

        # Référence (reply)
        if message.reference and message.reference.resolved:
            ref = message.reference.resolved
            ref_author = getattr(ref, "author", None)
            ref_is_bot = getattr(ref_author, "bot", False)
            ref_name = getattr(ref_author, "name", "?") if ref_author else "?"

            ref_lines: list[str] = []
            ref_text = (ref.content or "").strip()
            if ref_text:
                ref_lines.append(ref_text[:400] + ("…" if len(ref_text) > 400 else ""))
            for emb in getattr(ref, "embeds", []):
                t = _embed_to_text(emb)
                if t:
                    ref_lines.append(t[:300])
            preview = " | ".join(ref_lines)[:500] if ref_lines else "(sans texte)"
            label = "ton message" if ref_is_bot else ref_name
            parts.append(TextComponent(f"[Répond à {label} : \"{preview}\"]"))

            for att in getattr(ref, "attachments", []):
                fn = (att.filename or "").lower()
                if (att.content_type or "").startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    parts.append(ImageComponent(att.url, detail="low"))

        # Texte principal
        if text.strip():
            parts.append(TextComponent(f"{user_name}: {message.clean_content}"))
        elif message.embeds or message.stickers or message.attachments:
            parts.append(TextComponent(f"{user_name}:"))

        # URLs d'images dans le texte
        for m in re.finditer(r"https?://[^\s]+", text):
            url = re.sub(r"\?.*$", "", m.group(0))
            if url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                parts.append(ImageComponent(url, detail="auto"))
            elif url.lower().endswith(".gif"):
                parts.append(ImageComponent(f"{url}?format=png" if "?" not in url else f"{url}&format=png", detail="auto"))

        # Embeds
        for emb in message.embeds:
            emb_text = _embed_to_text(emb)
            if emb_text:
                parts.append(TextComponent(f"[EMBED]\n{emb_text[:800]}"))
            if emb.image and emb.image.url:
                url = emb.image.url
                if url.lower().endswith(".gif"):
                    url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                parts.append(ImageComponent(url, detail="high"))
            if emb.thumbnail and emb.thumbnail.url:
                url = emb.thumbnail.url
                if url.lower().endswith(".gif"):
                    url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                parts.append(ImageComponent(url, detail="low"))
            if emb.video and emb.video.url:
                parts.append(TextComponent(f"[VIDEO: {emb.video.url}]"))

        # Stickers
        for st in message.stickers:
            if st.url:
                parts.append(ImageComponent(st.url, detail="auto"))

        # Attachments images
        for att in message.attachments:
            ct = att.content_type or ""
            fn = (att.filename or "").lower()
            if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                url = att.url
                if fn.endswith(".gif"):
                    url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                parts.append(ImageComponent(url, detail="auto"))

        if not parts:
            parts.append(TextComponent(f"{user_name}: (message vide)"))

        record = self.context.add_user_message(components=parts, name=user_name)
        if hasattr(record, "metadata"):
            record.metadata["discord_message"] = message
        return record

    async def process_attachments(self, message: discord.Message) -> list:
        out = []
        for att in message.attachments:
            comps = await process_attachment(att, self.client, self.attachment_cache)
            out.extend(comps)
        return out

    async def run_completion(
        self, trigger_message: Optional[discord.Message] = None
    ) -> AssistantRecord:
        async with self._lock:
            return await self._run(trigger_message, 0)

    async def _run(self, trigger: Optional[discord.Message], depth: int) -> AssistantRecord:
        if depth >= MAX_RECURSION:
            return self.context.add_assistant_message(
                components=[TextComponent("Limite d'outils atteinte. Reformule ta demande.")],
            )

        self.trigger_message = trigger

        # Pièces jointes du trigger
        if trigger:
            comps = await self.process_attachments(trigger)
            if comps:
                recent = self.context.get_recent_messages(1)
                if recent and recent[0].role == "user":
                    recent[0].components.extend(comps)

        self.context.developer_prompt = self.developer_prompt_template()

        messages = self.context.prepare_payload()
        tools = self.tool_registry.get_compiled() if len(self.tool_registry) > 0 else []

        try:
            completion = await self.client.chat(
                messages=messages,
                tools=tools if tools else None,
            )
        except MariaOpenAIError as e:
            if "invalid_image_url" in str(e):
                self.context.filter_images()
                messages = self.context.prepare_payload()
                completion = await self.client.chat(messages=messages, tools=tools if tools else None)
            else:
                raise

        choice = completion.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCallRecord(
                        id=tc.id,
                        function_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )

        components = []
        if msg.content:
            components.append(TextComponent(msg.content))
        else:
            components.append(MetadataComponent("EMPTY"))

        assistant = self.context.add_assistant_message(
            components=components,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

        if tool_calls:
            await self._execute_tools(tool_calls)
            return await self._run(None, depth + 1)

        if not msg.content or not str(msg.content).strip():
            self.context._messages.pop()
            self.context.add_user_message(components=[TextComponent("[SYSTEM] Réponds maintenant.")], name="system")
            return await self._run(None, depth + 1)

        return assistant

    async def _execute_tools(self, tool_calls: list[ToolCallRecord]) -> None:
        for tc in tool_calls:
            tool = self.tool_registry.get(tc.function_name)
            if not tool:
                continue
            try:
                resp = await tool.execute(tc, self)
                self.context.add_message(resp)
            except Exception as e:
                logger.error(f"Outil {tc.function_name}: {e}")
                self.context.add_message(
                    ToolResponseRecord(
                        tool_call_id=tc.id,
                        response_data={"error": str(e)},
                        created_at=datetime.now(timezone.utc),
                    )
                )

    async def run_autonomous_task(self, user_name: str, user_id: int, task_prompt: str) -> AssistantRecord:
        """Tâche autonome isolée — contexte séparé, seule la réponse finale est réinjectée."""
        async with self._lock:
            isolated = ConversationContext(
                developer_prompt=self.developer_prompt_template(),
                context_window=self.context.context_window,
                context_age=self.context.context_age,
            )
            isolated.add_user_message(components=[TextComponent(task_prompt)], name=user_name)
            orig = self.context
            self.context = isolated
            try:
                result = await self._run(None, 0)
                text = result.full_text
            finally:
                self.context = orig
            return self.context.add_assistant_message(
                components=[TextComponent(text)],
                metadata={"autonomous_task": True, "task_owner_id": user_id},
            )

    def forget(self) -> None:
        self.context.clear()

    def get_stats(self) -> dict:
        return {"context_stats": self.context.get_stats()}


class ChannelSessionManager:
    """Gestionnaire de sessions."""

    def __init__(
        self,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        developer_prompt_template: Callable[[], str],
        api_key: str,
        *,
        context_window: int = 8192,
        context_age_hours: float = 2,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.developer_prompt_template = developer_prompt_template
        self.attachment_cache = AttachmentCache()
        self.message_cache = MessageCache()
        self.cache_search = CacheSearchClient(api_key)
        self._sessions: dict[int, ChannelSession] = {}
        self._context_window = context_window
        self._context_age_hours = context_age_hours

    def get_or_create(self, channel: discord.abc.Messageable) -> ChannelSession:
        if channel.id not in self._sessions:
            self._sessions[channel.id] = ChannelSession(
                channel_id=channel.id,
                client=self.client,
                tool_registry=self.tool_registry,
                attachment_cache=self.attachment_cache,
                message_cache=self.message_cache,
                cache_search=self.cache_search,
                developer_prompt_template=self.developer_prompt_template,
                context_window=self._context_window,
                context_age_hours=self._context_age_hours,
            )
        return self._sessions[channel.id]

    def get(self, channel_id: int) -> Optional[ChannelSession]:
        return self._sessions.get(channel_id)
