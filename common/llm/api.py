"""Façade publique de l'API GPT."""

import logging
from typing import Callable, Iterable, Optional, Sequence

import discord

from .client import MariaLLMClient
from .session import ChannelSession, ChannelSessionManager
from .tools import Tool, ToolRegistry
from .context import AssistantRecord, TextComponent


def _tool_response_failed(tr) -> bool:
    rd = getattr(tr, "response_data", None)
    return isinstance(rd, dict) and bool(rd.get("error"))


def _collect_run_artifacts(session, assistant) -> tuple[list, list[dict]]:
    """Réponses d'outils + preuves d'usage (hors appels en erreur)."""
    tool_responses: list = []
    used_tools: list[dict] = []
    found = False
    for m in reversed(session.context.get_messages()):
        if m == assistant:
            found = True
            continue
        if not found:
            continue
        if m.role == "tool":
            tool_responses.insert(0, m)
        elif m.role == "assistant":
            if hasattr(m, "tool_calls") and m.tool_calls:
                failed_ids = {
                    tr.tool_call_id for tr in tool_responses
                    if _tool_response_failed(tr)
                }
                seen_names = {t["name"] for t in used_tools}
                for tc in reversed(m.tool_calls):
                    if getattr(tc, "id", None) in failed_ids:
                        continue
                    if tc.function_name not in seen_names:
                        used_tools.insert(0, {
                            "name": tc.function_name,
                            "args": tc.arguments or {},
                        })
                        seen_names.add(tc.function_name)
        elif m.role == "user":
            if getattr(m, "name", None) == "system":
                continue
            break
    return tool_responses, used_tools

logger = logging.getLogger("llm.api")


class MariaResponse:
    """Réponse d'une complétion."""

    def __init__(
        self,
        text: str,
        assistant_record: AssistantRecord,
        tool_responses: list,
        used_tools: list[dict] | None = None,
    ):
        self.text = text
        self.assistant_record = assistant_record
        self.tool_responses = tool_responses
        # Chaque entrée : {"name": str, "args": dict}
        self.used_tools: list[dict] = used_tools or []


class MariaGptApi:
    """API GPT — point d'entrée unique."""

    def __init__(
        self,
        api_key: str,
        developer_prompt_template: Callable[..., str],
        *,
        completion_model: str = "gpt-5.6-luna",
        transcription_model: str = "gpt-4o-transcribe",
        max_tokens: int = 1536,
        context_window: int = 12000,
        context_age_hours: float = 2,
        max_messages: int = 0,
    ):
        self.client = MariaLLMClient(
            api_key=api_key,
            completion_model=completion_model,
            transcription_model=transcription_model,
            max_tokens=max_tokens,
        )
        self.tool_registry = ToolRegistry()
        self.session_manager = ChannelSessionManager(
            client=self.client,
            tool_registry=self.tool_registry,
            developer_prompt_template=developer_prompt_template,
            context_window=context_window,
            context_age_hours=context_age_hours,
            max_messages=max_messages,
        )

    async def run_completion(
        self,
        channel: discord.abc.Messageable,
        trigger_message: Optional[discord.Message] = None,
        *,
        model: Optional[str] = None,
        prompt_context: Optional[dict] = None,
    ) -> MariaResponse:
        session = self.session_manager.get_or_create(channel)
        assistant = await session.run_completion(
            trigger_message,
            model=model,
            prompt_context=prompt_context,
        )

        tool_responses, used_tools = _collect_run_artifacts(session, assistant)
        return MariaResponse(assistant.full_text, assistant, tool_responses, used_tools)

    async def run_isolated_completion(
        self,
        channel: discord.abc.Messageable,
        user_text: str,
        *,
        trigger_message=None,
        developer_prompt: str,
        allowed_tools: Optional[Sequence[str]] = None,
        prompt_context: Optional[dict] = None,
        model: Optional[str] = None,
    ) -> MariaResponse:
        """Tour LLM hors historique du salon (tâches planifiées)."""
        registry = ToolRegistry()
        if allowed_tools:
            for name in allowed_tools:
                tool = self.tool_registry.get(name)
                if tool is not None:
                    registry.register(tool)
                else:
                    logger.warning("Outil isolé introuvable : %s", name)
        mgr = self.session_manager
        session = ChannelSession(
            channel_id=0,
            client=self.client,
            tool_registry=registry,
            attachment_cache=mgr.attachment_cache,
            developer_prompt_template=lambda ctx: developer_prompt,
            context_window=mgr._context_window,
            context_age_hours=mgr._context_age_hours,
            max_messages=mgr._max_messages,
        )
        session.context.add_user_message(
            components=[TextComponent(user_text)],
            name="system",
        )
        assistant = await session.run_completion(
            trigger_message,
            model=model,
            prompt_context=prompt_context,
            skip_focus=True,
        )
        tool_responses, used_tools = _collect_run_artifacts(session, assistant)
        return MariaResponse(assistant.full_text, assistant, tool_responses, used_tools)

    async def record_assistant_post(
        self,
        channel: discord.abc.Messageable,
        text: str,
        *,
        discord_messages: Optional[Sequence[discord.Message]] = None,
        system_notes: Optional[Sequence[str]] = None,
    ) -> None:
        """Intègre un message déjà posté (tâche) dans l'historique du salon."""
        session = self.session_manager.get_or_create(channel)
        body = (text or "").strip()
        async with session._lock:
            record = session.context.add_assistant_message(
                components=[TextComponent(body)],
            )
            for msg in discord_messages or ():
                session._remember_ingested(msg.id, record)
                if hasattr(record, "metadata"):
                    record.metadata["discord_message"] = msg
            for note in system_notes or ():
                cleaned = (note or "").strip()
                if not cleaned:
                    continue
                session.context.add_user_message(
                    components=[TextComponent(f"[SYSTEM] {cleaned}")],
                    name="system",
                )
                session.record_system_note(cleaned)

    async def inject_context_note_async(self, channel: discord.abc.Messageable, note: str) -> None:
        """Injecte une note système en acquérant le lock de session.
        Garantit que la note est visible pour le prochain run_completion.
        Enregistre aussi la note dans le journal récent pour le [FOCUS]."""
        from .context import TextComponent
        session = self.session_manager.get_or_create(channel)
        async with session._lock:
            session.context.add_user_message(
                components=[TextComponent(f"[SYSTEM] {note}")],
                name="system",
            )
            session.record_system_note(note)

    def update_tools(self, tools: Iterable[Tool]) -> None:
        self.tool_registry.clear()
        self.tool_registry.register_multiple(*tools)

    async def close(self) -> None:
        await self.client.close()
