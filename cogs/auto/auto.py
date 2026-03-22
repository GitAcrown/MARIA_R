"""Cog Auto — transcription audio à la demande (réaction 📜)."""

import io
import time

import discord
from discord.ext import commands

from common.llm import MariaLLMClient


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

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        has_audio = any(self._is_audio(a) for a in message.attachments)
        if has_audio:
            await message.add_reaction("📜")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if user.bot or str(reaction.emoji) != "📜":
            return
        msg = reaction.message
        has_audio = False
        audio_att = None
        for a in msg.attachments:
            if self._is_audio(a):
                has_audio = True
                audio_att = a
                break
        if not has_audio or not audio_att:
            return
        key = audio_att.url
        if key in self._cache and time.time() < self._expiration.get(key, 0):
            transcript = self._cache[key]
        else:
            try:
                async with msg.channel.typing():
                    buf = io.BytesIO()
                    buf.name = audio_att.filename
                    await audio_att.save(buf, seek_begin=True)
                    transcript = await self._get_client().transcribe(buf)
                self._cache[key] = transcript
                self._expiration[key] = time.time() + self.EXPIRY_SEC
            except Exception as e:
                await msg.channel.send(f"Erreur de transcription : `{e}`", delete_after=15)
                await reaction.remove(user)
                return
        await reaction.remove(user)
        if len(transcript) > 1900:
            transcript = transcript[:1900] + "..."
        await msg.reply(f"> *{transcript}*\n-# Transcription demandée par {user.display_name}", mention_author=False)

    async def cog_unload(self):
        if self._client:
            await self._client.close()


async def setup(bot):
    await bot.add_cog(Auto(bot))
