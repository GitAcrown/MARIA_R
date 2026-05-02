"""Session par salon — contexte complet, lock, tools."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import discord

from .client import MariaLLMClient, MariaOpenAIError, MODEL_NANO
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

logger = logging.getLogger("llm.session")

USER_FORMAT = "{message.author.name}"
MAX_RECURSION = 8

SUMMARIZE_THRESHOLD = 32   # messages avant de déclencher un résumé
SUMMARIZE_CHUNK = 12       # messages les plus anciens condensés en 1 résumé
SUMMARIZE_PROMPT = (
    "Résume en 2-3 phrases ces échanges Discord "
    "(qui dit quoi, faits bruts, sans intro ni conclusion) :\n\n{messages}"
)


def _components_v2_to_parts(
    components: list,
    *,
    _depth: int = 0,
) -> tuple[list[str], list[str]]:
    """Walk a components-v2 tree recursively.
    Returns (text_parts, image_urls).
    Stops at depth 6 to avoid runaway recursion.
    """
    if _depth > 6:
        return [], []

    texts: list[str] = []
    images: list[str] = []

    for comp in components:
        name = type(comp).__name__

        if name == "TextDisplay":
            content = getattr(comp, "content", None) or getattr(comp, "value", None)
            if content:
                texts.append(str(content))

        elif name in ("Container", "Section", "ActionRow"):
            children = (
                getattr(comp, "children", None)
                or getattr(comp, "components", None)
                or []
            )
            sub_texts, sub_imgs = _components_v2_to_parts(children, _depth=_depth + 1)
            texts.extend(sub_texts)
            images.extend(sub_imgs)
            accessory = getattr(comp, "accessory", None)
            if accessory:
                acc_texts, acc_imgs = _components_v2_to_parts([accessory], _depth=_depth + 1)
                texts.extend(acc_texts)
                images.extend(acc_imgs)

        elif name == "MediaGallery":
            for item in getattr(comp, "items", []):
                media = getattr(item, "media", None)
                url = getattr(media, "url", None) if media else None
                if url:
                    images.append(url)

        elif name in ("Thumbnail", "UnfurledMediaItem"):
            media = getattr(comp, "media", None)
            url = getattr(media, "url", None) if media else getattr(comp, "url", None)
            if url:
                images.append(url)

    return texts, images


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
    """Session par salon — tous les messages vont dans le contexte GPT principal."""

    def __init__(
        self,
        channel_id: int,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        attachment_cache: AttachmentCache,
        developer_prompt_template: Callable[[], str],
        context_window: int = 12000,
        context_age_hours: float = 2,
        max_messages: int = 0,
    ):
        self.channel_id = channel_id
        self.client = client
        self.tool_registry = tool_registry
        self.attachment_cache = attachment_cache
        self.developer_prompt_template = developer_prompt_template
        self.context = ConversationContext(
            developer_prompt="",
            context_window=context_window,
            context_age=timedelta(hours=context_age_hours),
            max_messages=max_messages,
        )
        self._lock = asyncio.Lock()
        self.trigger_message: Optional[discord.Message] = None
        # IDs Discord des messages déjà ingérés dans cette session (évite doublons de référence)
        self._ingested_ids: set[int] = set()

    async def ingest_message(self, message: discord.Message, is_context_only: bool = False) -> MessageRecord:
        """Ingère un message dans le contexte GPT principal.

        - is_context_only=False → traitement complet (texte + images + embeds + attachments)
        - is_context_only=True  → texte + référence uniquement (messages de contexte, sans média)
          Si le message n'a pas de texte et is_context_only=True, il est ignoré.
        """
        text = message.content or ""
        user_name = USER_FORMAT.format(message=message)

        # Pour les messages contexte-seul sans texte, ignorer (évite le bruit)
        if is_context_only and not text.strip():
            return MessageRecord(
                role="user",
                components=[],
                created_at=datetime.now(timezone.utc),
                name=user_name,
            )

        parts: list = []

        # --- Référence (reply) ---
        if message.reference and message.reference.resolved:
            ref = message.reference.resolved
            ref_author = getattr(ref, "author", None)
            ref_is_bot = getattr(ref_author, "bot", False)
            ref_name = getattr(ref_author, "name", "?") if ref_author else "?"
            ref_id = getattr(ref, "id", None)
            label = "ton message" if ref_is_bot else ref_name

            if ref_id and ref_id in self._ingested_ids:
                # Message déjà dans le contexte de cette session : pas de doublon
                parts.append(TextComponent(f"[Suite de : {label}]"))
            else:
                # Message hors contexte (avant restart, autre session…)
                ref_text = (ref.content or "").strip()

                if ref_is_bot:
                    # Message du bot : si texte présent → le citer, sinon note générique
                    # Ne jamais dumper les composants v2 (LayoutView) — c'est du markdown illisible
                    if ref_text:
                        parts.append(TextComponent(
                            f"[Répond à {label} : \"{ref_text[:300]}\"]"
                        ))
                    else:
                        parts.append(TextComponent(f"[Répond à la dernière réponse du bot]"))
                else:
                    # Message utilisateur → aperçu complet
                    ref_lines: list[str] = []
                    if ref_text:
                        ref_lines.append(ref_text[:400] + ("…" if len(ref_text) > 400 else ""))
                    if not is_context_only:
                        for emb in getattr(ref, "embeds", []):
                            t = _embed_to_text(emb)
                            if t:
                                ref_lines.append(t[:300])
                        ref_comps = getattr(ref, "components", None)
                        if ref_comps:
                            comp_texts, _ = _components_v2_to_parts(list(ref_comps))
                            if comp_texts:
                                ref_lines.append("\n".join(comp_texts)[:400])
                    if ref_lines:
                        preview = " | ".join(ref_lines)[:500]
                        parts.append(TextComponent(f"[Répond à {label} : \"{preview}\"]"))

            if not is_context_only:
                for att in getattr(ref, "attachments", []):
                    fn = (att.filename or "").lower()
                    if (att.content_type or "").startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        parts.append(ImageComponent(att.url, detail="low"))

        # --- Texte principal ---
        if text.strip():
            parts.append(TextComponent(f"{user_name}: {message.clean_content}"))
        elif not is_context_only and (message.embeds or message.stickers or message.attachments):
            parts.append(TextComponent(f"{user_name}:"))

        # --- Média (uniquement pour les messages adressés au bot) ---
        if not is_context_only:
            # URLs d'images dans le texte
            for m in re.finditer(r"https?://[^\s]+", text):
                url = re.sub(r"\?.*$", "", m.group(0))
                if url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    parts.append(ImageComponent(url, detail="auto"))
                elif url.lower().endswith(".gif"):
                    parts.append(ImageComponent(
                        f"{url}?format=png" if "?" not in url else f"{url}&format=png",
                        detail="auto",
                    ))

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

            # Components v2
            if message.components:
                comp_texts, comp_imgs = _components_v2_to_parts(list(message.components))
                if comp_texts:
                    full = "\n".join(comp_texts)
                    parts.append(TextComponent(f"[LAYOUT]\n{full[:1200]}"))
                for url in comp_imgs[:6]:
                    if url.lower().endswith(".gif"):
                        url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                    parts.append(ImageComponent(url, detail="low"))

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
        self._ingested_ids.add(message.id)
        return record

    async def _maybe_summarize(self) -> None:
        """Condense les SUMMARIZE_CHUNK plus anciens messages en un résumé nano si le contexte est chargé."""
        if len(self.context._messages) < SUMMARIZE_THRESHOLD:
            return
        to_summarize = self.context._messages[:SUMMARIZE_CHUNK]
        remaining = self.context._messages[SUMMARIZE_CHUNK:]

        lines: list[str] = []
        for m in to_summarize:
            name = getattr(m, "name", None) or m.role
            if name == "system":
                continue
            text = m.full_text[:200].strip()
            if text:
                lines.append(f"{name}: {text}")
        if not lines:
            return

        summary = await self.client.summarize(
            SUMMARIZE_PROMPT.format(messages="\n".join(lines))
        )
        if not summary:
            return

        summary_record = MessageRecord(
            role="user",
            components=[TextComponent(f"[Résumé des échanges précédents]\n{summary}")],
            created_at=to_summarize[-1].created_at,
            name="system",
        )
        self.context._messages = [summary_record] + remaining
        logger.debug(f"Contexte résumé : {SUMMARIZE_CHUNK} messages → 1")

    async def run_completion(
        self, trigger_message: Optional[discord.Message] = None, *, model: Optional[str] = None
    ) -> AssistantRecord:
        async with self._lock:
            return await self._run(trigger_message, 0, model=model)

    async def _run(self, trigger: Optional[discord.Message], depth: int, *, model: Optional[str] = None) -> AssistantRecord:
        if depth >= MAX_RECURSION:
            return self.context.add_assistant_message(
                components=[TextComponent("Limite d'outils atteinte. Reformule ta demande.")],
            )

        # Résumé des anciens messages si contexte trop chargé (depth=0 uniquement)
        if depth == 0:
            await self._maybe_summarize()

        self.trigger_message = trigger

        # Pièces jointes du trigger
        if trigger:
            out = []
            for att in trigger.attachments:
                comps = await process_attachment(att, self.client, self.attachment_cache)
                out.extend(comps)
            if out:
                recent = self.context.get_recent_messages(1)
                if recent and recent[0].role == "user":
                    recent[0].components.extend(out)

        self.context.developer_prompt = self.developer_prompt_template()

        messages = self.context.prepare_payload()

        # Injecter une note éphémère (non persistée) pour indiquer le trigger au LLM
        if depth == 0 and trigger:
            author = trigger.author.display_name
            content = trigger.content.strip()
            if content:
                hint = f"[FOCUS] Tu réponds au message de {author} : « {content[:200]} »"
            else:
                hint = f"[FOCUS] Tu réponds à {author}."
            messages = messages + [{"role": "user", "content": hint, "name": "system"}]

        tools = self.tool_registry.get_compiled() if len(self.tool_registry) > 0 else []

        try:
            completion = await self.client.chat(
                messages=messages,
                tools=tools if tools else None,
                model=model,
            )
        except MariaOpenAIError as e:
            if "invalid_image_url" in str(e):
                self.context.filter_images()
                messages = self.context.prepare_payload()
                completion = await self.client.chat(messages=messages, tools=tools if tools else None, model=model)
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
            return await self._run(None, depth + 1, model=model)

        if not msg.content or not str(msg.content).strip():
            self.context._messages.pop()
            self.context.add_user_message(components=[TextComponent("[SYSTEM] Réponds maintenant.")], name="system")
            return await self._run(None, depth + 1, model=model)

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

    def forget(self) -> None:
        self.context.clear()

    def get_stats(self) -> dict:
        return {"context_stats": self.context.get_stats()}


class ChannelSessionManager:
    """Gestionnaire de sessions par salon."""

    def __init__(
        self,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        developer_prompt_template: Callable[[], str],
        *,
        context_window: int = 12000,
        context_age_hours: float = 2,
        max_messages: int = 0,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.developer_prompt_template = developer_prompt_template
        self.attachment_cache = AttachmentCache()
        self._sessions: dict[int, ChannelSession] = {}
        self._context_window = context_window
        self._context_age_hours = context_age_hours
        self._max_messages = max_messages

    def get_or_create(self, channel: discord.abc.Messageable) -> ChannelSession:
        if channel.id not in self._sessions:
            self._sessions[channel.id] = ChannelSession(
                channel_id=channel.id,
                client=self.client,
                tool_registry=self.tool_registry,
                attachment_cache=self.attachment_cache,
                developer_prompt_template=self.developer_prompt_template,
                context_window=self._context_window,
                context_age_hours=self._context_age_hours,
                max_messages=self._max_messages,
            )
        return self._sessions[channel.id]

    def get(self, channel_id: int) -> Optional[ChannelSession]:
        return self._sessions.get(channel_id)
