"""Pièces jointes — images, audio (transcription), fichiers texte. Pas de vidéo."""

import io
import logging
from pathlib import Path
from typing import Optional

import discord

from .context import ContentComponent, TextComponent, ImageComponent, MetadataComponent
from .client import MariaLLMClient

logger = logging.getLogger("llm.attachments")

TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)
MAX_TEXT_SIZE = 1024 * 1024  # 1 Mo
MAX_TEXT_CHARS = 80000
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}


class AttachmentCache:
    """Cache transcriptions audio."""

    def __init__(self, max_size: int = 20):
        self._cache: dict[str, str] = {}
        self._max = max_size
        self._order: list[str] = []

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self._cache[key] = value
        while len(self._cache) > self._max and self._order:
            k = self._order.pop(0)
            self._cache.pop(k, None)

    def get_stats(self) -> dict:
        return {"size": len(self._cache)}


def _is_audio(attachment: discord.Attachment) -> bool:
    ct = attachment.content_type or ""
    return ct.startswith("audio/") or Path(attachment.filename or "").suffix.lower() in AUDIO_EXT


def _is_image(attachment: discord.Attachment) -> bool:
    ct = attachment.content_type or ""
    fn = (attachment.filename or "").lower()
    return (
        ct.startswith("image/")
        or fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))
    )


def _is_text_file(attachment: discord.Attachment) -> bool:
    fn = (attachment.filename or "").lower()
    return fn.endswith(
        (".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv", ".log")
    )


def _image_url(att: discord.Attachment) -> str:
    """URL image — fix pour CDN Discord (éviter invalid_image_url)."""
    url = att.url
    if (att.filename or "").lower().endswith(".gif"):
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}format=png"
    return url


async def _process_audio(
    attachment: discord.Attachment, client: MariaLLMClient, cache: AttachmentCache
) -> list[ContentComponent]:
    key = attachment.url
    cached = cache.get(key)
    if cached:
        return [
            MetadataComponent("AUDIO", filename=attachment.filename, transcript=cached, url=key)
        ]

    try:
        buf = io.BytesIO()
        buf.name = attachment.filename
        await attachment.save(buf, seek_begin=True)
        transcript = await client.transcribe(buf)
        cache.set(key, transcript)
        return [
            MetadataComponent("AUDIO", filename=attachment.filename, transcript=transcript, url=key)
        ]
    except Exception as e:
        logger.error(f"Transcription audio: {e}")
        return [
            MetadataComponent(
                "AUDIO", filename=attachment.filename, error="TRANSCRIPTION_FAILED", url=key
            )
        ]


async def _process_text_file(attachment: discord.Attachment) -> list[ContentComponent]:
    if attachment.size and attachment.size > MAX_TEXT_SIZE:
        return [
            MetadataComponent("TEXT_FILE", filename=attachment.filename, error="FILE_TOO_LARGE")
        ]
    try:
        raw = await attachment.read()
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return [MetadataComponent("TEXT_FILE", filename=attachment.filename, error="ENCODING")]

        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS] + "\n... [TRONQUÉ]"
        ext = Path(attachment.filename or "x").suffix.lstrip(".") or "txt"
        return [
            MetadataComponent("TEXT_FILE", filename=attachment.filename),
            TextComponent(f"```{ext}\n{content}\n```"),
        ]
    except Exception as e:
        logger.error(f"Fichier texte: {e}")
        return [MetadataComponent("TEXT_FILE", filename=attachment.filename, error=str(e))]


async def process_attachment(
    attachment: discord.Attachment,
    client: MariaLLMClient,
    cache: AttachmentCache,
) -> list[ContentComponent]:
    """Dispatche selon le type : audio → transcription, image → URL, texte → lecture."""
    if _is_audio(attachment):
        return await _process_audio(attachment, client, cache)
    if _is_image(attachment):
        return [ImageComponent(_image_url(attachment), detail="auto")]
    if _is_text_file(attachment):
        return await _process_text_file(attachment)
    return []
