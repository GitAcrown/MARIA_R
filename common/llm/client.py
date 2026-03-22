"""Client OpenAI — wrapper minimal gpt-5.4-mini / nano."""

import logging
from typing import Any, Optional

from openai import AsyncOpenAI
import openai

logger = logging.getLogger("llm.client")

# Modèles
MODEL_MAIN = "gpt-5.4-mini"
MODEL_NANO = "gpt-5.4-nano"
MODEL_TRANSCRIBE = "gpt-4o-transcribe"


class MariaLLMError(Exception):
    """Erreur LLM."""

    pass


class MariaOpenAIError(MariaLLMError):
    """Erreur API OpenAI."""

    pass


class MariaLLMClient:
    """Client unique pour API OpenAI — complétion, transcription."""

    def __init__(
        self,
        api_key: str,
        *,
        completion_model: str = MODEL_MAIN,
        transcription_model: str = MODEL_TRANSCRIBE,
        max_tokens: int = 1024,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self.completion_model = completion_model
        self.transcription_model = transcription_model
        self.max_tokens = max_tokens
        self._stats = {"completions": 0, "transcriptions": 0, "errors": 0}

    async def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Complétion chat."""
        kwargs = {
            "model": model or self.completion_model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True

        try:
            out = await self._client.chat.completions.create(**kwargs)
            self._stats["completions"] += 1
            return out
        except (openai.BadRequestError, openai.OpenAIError) as e:
            self._stats["errors"] += 1
            raise MariaOpenAIError(str(e)) from e

    async def transcribe(self, audio_file, *, model: Optional[str] = None) -> str:
        """Transcription audio."""
        try:
            t = await self._client.audio.transcriptions.create(
                model=model or self.transcription_model,
                file=audio_file,
            )
            self._stats["transcriptions"] += 1
            return t.text
        except (openai.BadRequestError, openai.OpenAIError) as e:
            self._stats["errors"] += 1
            raise MariaOpenAIError(str(e)) from e

    def get_stats(self) -> dict:
        return self._stats.copy()

    async def close(self) -> None:
        await self._client.close()
