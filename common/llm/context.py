"""Contexte de conversation — fenêtre restreinte, trim simple."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import tiktoken

logger = logging.getLogger("llm.context")

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Fenêtre restreinte par défaut
DEFAULT_WINDOW = 8192
DEFAULT_AGE = timedelta(hours=2)


@dataclass
class ContentComponent:
    """Composant de contenu (texte ou image)."""

    type: Literal["text", "image_url"]
    data: dict
    token_count: int = 0

    def to_payload(self) -> dict:
        return self.data


class TextComponent(ContentComponent):
    def __init__(self, text: str):
        super().__init__(
            type="text",
            data={"type": "text", "text": text},
            token_count=len(TOKENIZER.encode(text)),
        )


class ImageComponent(ContentComponent):
    def __init__(self, url: str, detail: Literal["low", "high", "auto"] = "auto"):
        super().__init__(
            type="image_url",
            data={"type": "image_url", "image_url": {"url": url, "detail": detail}},
            token_count=250,
        )


class MetadataComponent(ContentComponent):
    """Métadonnée affichée comme texte."""

    def __init__(self, title: str, **meta):
        text = f"<{title.upper()}"
        if meta:
            text += " " + " ".join(f"{k}={v}" for k, v in meta.items())
        text += ">"
        super().__init__(
            type="text",
            data={"type": "text", "text": text},
            token_count=len(TOKENIZER.encode(text)),
        )


@dataclass
class MessageRecord:
    """Message dans l'historique."""

    role: Literal["user", "assistant", "developer", "tool"]
    components: list[ContentComponent]
    created_at: datetime
    name: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def token_count(self) -> int:
        return sum(c.token_count for c in self.components)

    @property
    def full_text(self) -> str:
        return "".join(
            c.data.get("text", "")
            for c in self.components
            if c.type == "text" and "text" in c.data
        )

    def to_payload(self) -> dict:
        p = {
            "role": self.role,
            "content": [c.to_payload() for c in self.components],
        }
        if self.name:
            p["name"] = self.name
        return p


@dataclass
class ToolCallRecord:
    id: str
    function_name: str
    arguments: dict

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


class AssistantRecord(MessageRecord):
    """Message assistant avec tool calls optionnels."""

    def __init__(
        self,
        components: list[ContentComponent],
        created_at: datetime,
        tool_calls: Optional[list["ToolCallRecord"]] = None,
        finish_reason: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            role="assistant",
            components=components,
            created_at=created_at,
            **kwargs,
        )
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason

    def to_payload(self) -> dict:
        if self.tool_calls:
            return {
                "role": "assistant",
                "tool_calls": [t.to_payload() for t in self.tool_calls],
                "content": None,
            }
        return super().to_payload()


class ToolResponseRecord(MessageRecord):
    """Réponse d'outil."""

    def __init__(
        self,
        tool_call_id: str,
        response_data: dict,
        created_at: datetime,
        **kwargs,
    ):
        super().__init__(
            role="tool",
            components=[TextComponent(json.dumps(response_data, ensure_ascii=False))],
            created_at=created_at,
            **kwargs,
        )
        self.tool_call_id = tool_call_id
        self.response_data = response_data

    def to_payload(self) -> dict:
        return {
            "role": "tool",
            "content": json.dumps(self.response_data, ensure_ascii=False),
            "tool_call_id": self.tool_call_id,
        }


class ConversationContext:
    """Contexte restreint — trim par tokens, âge et nombre de messages."""

    def __init__(
        self,
        developer_prompt: str,
        *,
        context_window: int = DEFAULT_WINDOW,
        context_age: timedelta = DEFAULT_AGE,
        max_messages: int = 0,
    ):
        self.developer_prompt = developer_prompt
        self.context_window = context_window
        self.context_age = context_age
        self.max_messages = max_messages  # 0 = pas de limite
        self._messages: list[MessageRecord] = []
        self._needs_trim = False

    def add_message(self, msg: MessageRecord) -> None:
        self._messages.append(msg)
        self._needs_trim = True

    def add_user_message(
        self,
        components: list[ContentComponent],
        name: str = "user",
        discord_message=None,
        **meta,
    ) -> MessageRecord:
        r = MessageRecord(
            role="user",
            components=components,
            created_at=datetime.now(timezone.utc),
            name=name,
            metadata=meta,
        )
        if discord_message is not None:
            r.metadata["discord_message"] = discord_message
        self.add_message(r)
        return r

    def add_assistant_message(
        self,
        components: list[ContentComponent],
        tool_calls: Optional[list[ToolCallRecord]] = None,
        finish_reason: Optional[str] = None,
        **meta,
    ) -> AssistantRecord:
        r = AssistantRecord(
            components=components,
            created_at=datetime.now(timezone.utc),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            metadata=meta,
        )
        self.add_message(r)
        return r

    def add_tool_response(self, tool_call_id: str, response_data: dict) -> ToolResponseRecord:
        r = ToolResponseRecord(
            tool_call_id=tool_call_id,
            response_data=response_data,
            created_at=datetime.now(timezone.utc),
        )
        self.add_message(r)
        return r

    def get_recent_messages(self, count: int) -> list[MessageRecord]:
        return self._messages[-count:] if count > 0 else []

    def get_messages(self) -> list[MessageRecord]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._needs_trim = False

    def trim(self) -> None:
        """Supprime messages trop vieux, hors fenêtre tokens, ou hors plafond de messages."""
        now = datetime.now(timezone.utc)
        self._messages = [m for m in self._messages if now - m.created_at < self.context_age]
        total = 0
        kept: list[MessageRecord] = []
        for m in reversed(self._messages):
            if self.context_window > 0 and total + m.token_count > self.context_window:
                break
            kept.insert(0, m)
            total += m.token_count
        # Plafond de messages (garde les plus récents)
        if self.max_messages > 0 and len(kept) > self.max_messages:
            kept = kept[-self.max_messages:]
        self._messages = kept
        self._needs_trim = False

    def prepare_payload(self) -> list[dict]:
        if self._needs_trim:
            self.trim()
        dev = MessageRecord(
            role="developer",
            components=[TextComponent(self.developer_prompt)],
            created_at=datetime.now(timezone.utc),
        )
        return [dev.to_payload()] + [m.to_payload() for m in self._messages]

    def get_stats(self) -> dict:
        total = sum(m.token_count for m in self._messages)
        return {
            "total_messages": len(self._messages),
            "total_tokens": total,
            "window_usage_pct": (total / self.context_window * 100) if self.context_window else 0,
            "context_window": self.context_window,
        }

    def filter_images(self) -> None:
        """Retire les images (pour retry après invalid_image_url)."""
        for m in self._messages:
            m.components = [c for c in m.components if c.type != "image_url"]
