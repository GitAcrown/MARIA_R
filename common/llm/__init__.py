"""common.llm — Couche LLM minimale pour MARIA."""

from .api import MariaGptApi, MariaResponse
from .client import MariaLLMClient, MariaLLMError, MariaOpenAIError
from .context import (
    ContentComponent,
    TextComponent,
    ImageComponent,
    MetadataComponent,
    MessageRecord,
    AssistantRecord,
    ToolCallRecord,
    ToolResponseRecord,
)
from .session import resolve_message_reference
from .tools import Tool, ToolRegistry

__all__ = [
    "MariaGptApi",
    "MariaResponse",
    "MariaLLMClient",
    "MariaLLMError",
    "MariaOpenAIError",
    "ContentComponent",
    "TextComponent",
    "ImageComponent",
    "MetadataComponent",
    "MessageRecord",
    "AssistantRecord",
    "ToolCallRecord",
    "ToolResponseRecord",
    "Tool",
    "ToolRegistry",
    "resolve_message_reference",
]
