"""Outils — registre et exécution."""

import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Union, Awaitable

from .context import ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("llm.tools")


class Tool:
    """Outil OpenAI function calling."""

    def __init__(
        self,
        name: str,
        description: str,
        properties: dict,
        function: Union[Callable, Callable[..., Awaitable]],
    ):
        self.name = name
        self.description = description
        self.properties = properties
        self.function = function
        self._required = list(properties.keys())

    async def execute(
        self, tool_call: ToolCallRecord, context_data: Any = None
    ) -> ToolResponseRecord:
        try:
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(tool_call, context_data)
            else:
                result = self.function(tool_call, context_data)
            if isinstance(result, ToolResponseRecord):
                return result
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data=result if isinstance(result, dict) else {"result": result},
                created_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Outil {self.name}: {e}")
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={"error": str(e)},
                created_at=datetime.now(timezone.utc),
            )

    def to_openai_dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self._required,
                    "additionalProperties": False,
                },
            },
        }


class ToolRegistry:
    """Registre d'outils."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._cache: list[dict] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cache = None

    def register_multiple(self, *tools: Tool) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_compiled(self) -> list[dict]:
        if self._cache is None:
            self._cache = [t.to_openai_dict() for t in self._tools.values()]
        return self._cache

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._cache = None

    def clear(self) -> None:
        self._tools.clear()
        self._cache = None

    def __len__(self) -> int:
        return len(self._tools)
