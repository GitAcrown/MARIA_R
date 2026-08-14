"""Client OpenAI — wrapper minimal gpt-5.6-luna."""

import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional

from openai import AsyncOpenAI
import openai

logger = logging.getLogger("llm.client")

# Modèles
MODEL_MAIN = "gpt-5.6-luna"
MODEL_TRANSCRIBE = "gpt-4o-transcribe"
# Modèle de repli si MODEL_MAIN renvoie une erreur de permissions (401) — ex. accès
# au modèle pas encore activé sur l'organisation/clé API.
MODEL_FALLBACK = "gpt-5.4-mini"

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

    def _chat_kwargs(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": model or self.completion_model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self.max_tokens,
            # gpt-5.x : sans ça, le raisonnement peut manger tout max_completion_tokens
            # (sortie content vide) — notamment /moi, /global, résumés, etc.
            # Les function tools exigent aussi reasoning_effort='none'.
            "reasoning_effort": "none",
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        if response_format:
            kwargs["response_format"] = response_format
        return kwargs

    async def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        tools: Optional[list] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        on_text_reset: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Any:
        """Complétion chat.

        `response_format` (optionnel) est transmis tel quel à l'API pour forcer une
        sortie structurée (ex. ``{"type": "json_object"}`` ou un json_schema strict).
        Sans valeur, le comportement est inchangé.

        Si `stream=True`, consomme le flux OpenAI et renvoie un objet compatible
        (choices[0].message / finish_reason). `on_text_delta` est appelé au fil
        de l'eau tant qu'aucun tool call n'apparaît ; `on_text_reset` si des
        tools arrivent après du texte déjà streamé. En cas d'échec du stream,
        repli automatique sur une complétion synchrone.
        """
        kwargs = self._chat_kwargs(
            messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if stream:
            try:
                return await self._create_stream(kwargs, on_text_delta, on_text_reset)
            except MariaOpenAIError as e:
                logger.warning("Stream OpenAI échoué (%s) — repli synchrone.", e)

        try:
            return await self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            # 401 "insufficient permissions" typique d'un modèle pas encore
            # accessible sur le compte/projet — on retente une fois avec un modèle de repli.
            if kwargs["model"] == MODEL_FALLBACK:
                raise MariaOpenAIError(str(e)) from e
            self._log_model_fallback(kwargs["model"], e)
            kwargs["model"] = MODEL_FALLBACK
            try:
                return await self._client.chat.completions.create(**kwargs)
            except (openai.BadRequestError, openai.OpenAIError) as e2:
                raise MariaOpenAIError(str(e2)) from e2
        except (openai.BadRequestError, openai.OpenAIError) as e:
            raise MariaOpenAIError(str(e)) from e

    def _log_model_fallback(self, refused: str, err: Exception) -> None:
        logger.error(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!  REPLI MODÈLE — INFO ESSENTIELLE                       !!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!  Modèle demandé refusé (401 permissions)               !!\n"
            "!!  demandé : %-44s !!\n"
            "!!  repli   : %-44s !!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "Détail API : %s",
            refused,
            MODEL_FALLBACK,
            err,
        )

    async def _create_stream(
        self,
        kwargs: dict,
        on_text_delta: Optional[Callable[[str], Awaitable[None]]],
        on_text_reset: Optional[Callable[[], Awaitable[None]]],
    ) -> Any:
        stream_kwargs = {**kwargs, "stream": True}
        try:
            stream = await self._client.chat.completions.create(**stream_kwargs)
            return await self._consume_stream(stream, on_text_delta, on_text_reset)
        except openai.AuthenticationError as e:
            if stream_kwargs["model"] == MODEL_FALLBACK:
                raise MariaOpenAIError(str(e)) from e
            self._log_model_fallback(stream_kwargs["model"], e)
            stream_kwargs["model"] = MODEL_FALLBACK
            try:
                stream = await self._client.chat.completions.create(**stream_kwargs)
                return await self._consume_stream(stream, on_text_delta, on_text_reset)
            except (openai.BadRequestError, openai.OpenAIError) as e2:
                raise MariaOpenAIError(str(e2)) from e2
        except (openai.BadRequestError, openai.OpenAIError) as e:
            raise MariaOpenAIError(str(e)) from e

    async def _consume_stream(
        self,
        stream,
        on_text_delta: Optional[Callable[[str], Awaitable[None]]],
        on_text_reset: Optional[Callable[[], Awaitable[None]]],
    ) -> Any:
        content = ""
        tools: dict[int, dict[str, str]] = {}
        finish_reason: Optional[str] = None
        streamed = False
        reset_done = False
        async for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if getattr(delta, "tool_calls", None):
                if streamed and not reset_done and on_text_reset is not None:
                    reset_done = True
                    try:
                        await on_text_reset()
                    except Exception:
                        logger.exception("on_text_reset")
                    streamed = False
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tools.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
            piece = getattr(delta, "content", None)
            if piece and not tools:
                content += piece
                if on_text_delta is not None:
                    try:
                        await on_text_delta(content)
                        streamed = True
                    except Exception:
                        logger.exception("on_text_delta")
        ordered = [tools[i] for i in sorted(tools)]
        fake_tcs = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in ordered
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason or "stop",
                    message=SimpleNamespace(
                        content=content or None,
                        tool_calls=fake_tcs or None,
                    ),
                )
            ]
        )

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
