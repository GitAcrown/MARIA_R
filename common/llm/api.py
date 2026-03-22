"""Façade publique de l'API GPT."""

from datetime import timedelta
from typing import Callable, Iterable, Optional

import discord

from .client import MariaLLMClient
from .session import ChannelSessionManager, ChannelSession
from .tools import Tool, ToolRegistry
from .context import AssistantRecord, MessageRecord


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

    @property
    def has_tools(self) -> bool:
        return bool(self.used_tools)


class MariaGptApi:
    """API GPT — point d'entrée unique."""

    def __init__(
        self,
        api_key: str,
        developer_prompt_template: Callable[[], str],
        *,
        completion_model: str = "gpt-5.4-mini",
        transcription_model: str = "gpt-4o-transcribe",
        max_tokens: int = 1024,
        context_window: int = 8192,
        context_age_hours: float = 2,
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
            api_key=api_key,
            context_window=context_window,
            context_age_hours=context_age_hours,
        )

    async def ingest_message(
        self, channel: discord.abc.Messageable, message: discord.Message, is_context_only: bool = False
    ) -> None:
        session = self.session_manager.get_or_create(channel)
        await session.ingest_message(message, is_context_only)

    async def run_completion(
        self,
        channel: discord.abc.Messageable,
        trigger_message: Optional[discord.Message] = None,
    ) -> MariaResponse:
        session = self.session_manager.get_or_create(channel)
        assistant = await session.run_completion(trigger_message)

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
                    seen_names = {t["name"] for t in used_tools}
                    for tc in reversed(m.tool_calls):
                        if tc.function_name not in seen_names:
                            used_tools.insert(0, {"name": tc.function_name, "args": tc.arguments or {}})
                            seen_names.add(tc.function_name)
            elif m.role == "user":
                break

        return MariaResponse(assistant.full_text, assistant, tool_responses, used_tools)

    async def run_autonomous_task(
        self,
        channel: discord.abc.Messageable,
        user_name: str,
        user_id: int,
        task_prompt: str,
    ) -> MariaResponse:
        session = self.session_manager.get_or_create(channel)
        assistant = await session.run_autonomous_task(user_name, user_id, task_prompt)
        return MariaResponse(assistant.full_text, assistant, [])

    async def forget(self, channel: discord.abc.Messageable) -> None:
        session = self.session_manager.get(channel.id)
        if session:
            session.forget()

    def add_tools(self, *tools: Tool) -> None:
        self.tool_registry.register_multiple(*tools)

    def remove_tool(self, name: str) -> None:
        self.tool_registry.unregister(name)

    def update_tools(self, tools: Iterable[Tool]) -> None:
        self.tool_registry.clear()
        self.tool_registry.register_multiple(*tools)

    async def close(self) -> None:
        await self.client.close()
