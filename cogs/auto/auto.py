"""Cog Auto — transcription audio à la demande (réaction 📜) et automatique."""

import io
import logging
import time

import discord
from discord.ext import commands

from common.llm import MariaLLMClient

logger = logging.getLogger("MARIA.Auto")

AUTO_TRANSCRIBE_MAX_SECS = 120


class Auto(commands.Cog):
    """Transcription audio — réaction 📜 sur message audio pour transcrire."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._client: MariaLLMClient | None = None
        self._cache: dict[str, str] = {}
        self._expiration: dict[str, float] = {}
        self.EXPIRY_SEC = 300

    def _get_client(self) -> MariaLLMClient:
        if self._client is None:
            self._client = MariaLLMClient(
                api_key=self.bot.config["OPENAI_API_KEY"],
                transcription_model="gpt-4o-transcribe",
            )
        return self._client

    def _is_audio(self, att: discord.Attachment) -> bool:
        ct = att.content_type or ""
        ext = (att.filename or "").lower()
        return ct.startswith("audio/") or ext.endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac"))

    def _is_voice_message(self, message: discord.Message) -> discord.Attachment | None:
        """Retourne la pièce jointe si le message est un message vocal Discord (< 2 min).
        Le flag voice_message correspond au bit 13 de MessageFlags."""
        if not (message.flags.value & (1 << 13)):
            return None
        for att in message.attachments:
            if not self._is_audio(att):
                continue
            duration = getattr(att, "duration_secs", None)
            # Si duration_secs est absent, on fait confiance au flag voice_message
            if duration is None or duration <= AUTO_TRANSCRIBE_MAX_SECS:
                return att
        return None

    def _auto_transcribe_enabled(self, channel) -> bool:
        chat_cog = self.bot.get_cog("Chat")
        if not chat_cog or not hasattr(chat_cog, "_channel_config"):
            return False
        return bool(chat_cog._channel_config(channel).get("auto_transcribe", False))

    async def _do_transcribe(
        self,
        att: discord.Attachment,
        reply_to: discord.Message,
        requester_name: str | None = None,
    ) -> None:
        key = att.url
        if key in self._cache and time.time() < self._expiration.get(key, 0):
            transcript = self._cache[key]
            logger.debug(f"Transcription depuis cache : {att.filename}")
        else:
            logger.info(f"Transcription de {att.filename!r} ({getattr(att, 'duration_secs', '?')}s) dans #{reply_to.channel}")
            try:
                async with reply_to.channel.typing():
                    buf = io.BytesIO()
                    buf.name = att.filename
                    await att.save(buf, seek_begin=True)
                    size = buf.seek(0, 2)
                    buf.seek(0)
                    if size == 0:
                        logger.error(f"Buffer vide après att.save() pour {att.filename!r} — URL : {att.url}")
                        await reply_to.channel.send("Erreur de transcription : fichier vide.", delete_after=15)
                        return
                    logger.debug(f"Fichier téléchargé ({size} octets), envoi à l'API...")
                    transcript = await self._get_client().transcribe(buf)
                    logger.info(f"Transcription réussie : {len(transcript)} caractères")
                self._cache[key] = transcript
                self._expiration[key] = time.time() + self.EXPIRY_SEC
            except Exception as e:
                logger.error(f"Erreur transcription : {e}", exc_info=True)
                try:
                    await reply_to.channel.send(f"Erreur de transcription : `{e}`", delete_after=15)
                except Exception:
                    pass
                return
        if len(transcript) > 1900:
            transcript = transcript[:1900] + "..."
        suffix = f"\n-# Transcription demandée par {requester_name}" if requester_name else "\n-# Transcription automatique"
        await reply_to.reply(f"> {transcript}{suffix}", mention_author=False)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        voice_att = self._is_voice_message(message)
        if voice_att:
            if self._auto_transcribe_enabled(message.channel):
                logger.info(f"Message vocal détecté dans #{message.channel} — transcription auto")
                await self._do_transcribe(voice_att, message)
                return  # pas de réaction si on transcrit automatiquement
            logger.debug(f"Message vocal dans #{message.channel} — auto-transcription désactivée")

        has_audio = any(self._is_audio(a) for a in message.attachments)
        if has_audio:
            await message.add_reaction("📜")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if user.bot or str(reaction.emoji) != "📜":
            return
        msg = reaction.message
        audio_att = next((a for a in msg.attachments if self._is_audio(a)), None)
        if not audio_att:
            return
        await reaction.remove(user)
        await self._do_transcribe(audio_att, msg, requester_name=user.display_name)

    async def cog_unload(self):
        if self._client:
            await self._client.close()


async def setup(bot):
    await bot.add_cog(Auto(bot))
