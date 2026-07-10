"""Client OpenAI — wrapper minimal gpt-5.6-luna / nano."""

import logging
from typing import Any, Optional

from openai import AsyncOpenAI
import openai

logger = logging.getLogger("llm.client")

# Modèles
MODEL_MAIN = "gpt-5.6-luna"
MODEL_TRANSCRIBE = "gpt-4o-transcribe"

# Réseau — timeout par requête et nombre de tentatives.
# Le SDK OpenAI relance automatiquement sur 429, 5xx et erreurs réseau/timeout.
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 2


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
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        self.completion_model = completion_model
        self.transcription_model = transcription_model
        self.max_tokens = max_tokens

    async def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> Any:
        """Complétion chat.

        `response_format` (optionnel) est transmis tel quel à l'API pour forcer une
        sortie structurée (ex. ``{"type": "json_object"}`` ou un json_schema strict).
        Sans valeur, le comportement est inchangé.
        """
        kwargs = {
            "model": model or self.completion_model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        if response_format:
            kwargs["response_format"] = response_format

        try:
            return await self._client.chat.completions.create(**kwargs)
        except (openai.BadRequestError, openai.OpenAIError) as e:
            raise MariaOpenAIError(str(e)) from e

    async def transcribe(self, audio_file, *, model: Optional[str] = None) -> str:
        """Transcription audio."""
        try:
            t = await self._client.audio.transcriptions.create(
                model=model or self.transcription_model,
                file=audio_file,
            )
            return t.text
        except (openai.BadRequestError, openai.OpenAIError) as e:
            raise MariaOpenAIError(str(e)) from e

    async def close(self) -> None:
        await self._client.close()
