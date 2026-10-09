"""Une transcription vocale reste dans l'historique du salon, au nom de l'auteur."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from common.llm.session import ChannelSession


def _voice(mid: int = 7):
    return SimpleNamespace(
        id=mid,
        content="",
        clean_content="",
        reference=None,
        embeds=[],
        components=[],
        stickers=[],
        attachments=[],
        created_at=datetime(2026, 10, 9, 17, 32, tzinfo=timezone.utc),
        author=SimpleNamespace(name="lea", id=42),
        flags=SimpleNamespace(value=1 << 13),
        guild=None,
        _state=None,
    )


def _session() -> ChannelSession:
    return ChannelSession(
        channel_id=1,
        client=object(),
        tool_registry=object(),
        attachment_cache=object(),
        developer_prompt_template=lambda ctx: "",
    )


class VoiceHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_lands_on_the_author_line(self):
        session = _session()
        await session.attach_voice_transcript(_voice(), "on se dispute depuis vingt minutes")
        lines = [m.full_text for m in session.context.get_messages()]
        self.assertEqual(len(lines), 1)
        self.assertIn("lea (42)", lines[0])
        self.assertIn("(vocal) on se dispute depuis vingt minutes", lines[0])
        self.assertNotIn("[SYSTEM]", lines[0])
        self.assertNotIn("[vocal]", lines[0])

    async def test_transcript_replaces_a_bare_vocal_line(self):
        session = _session()
        msg = _voice()
        session._ingest_locked(msg, False, resolved_ref=None)
        before = session.context.get_messages()[0].full_text
        self.assertIn("[vocal]", before)
        await session.attach_voice_transcript(msg, "le film est nul")
        lines = [m.full_text for m in session.context.get_messages()]
        self.assertEqual(len(lines), 1)
        self.assertIn("(vocal) le film est nul", lines[0])
        self.assertNotIn("[vocal]", lines[0])


if __name__ == "__main__":
    unittest.main()
