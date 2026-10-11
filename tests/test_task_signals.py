"""Signaux Discord des tâches et sous-texte -# des notifications."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from common.task_recipe import (
    build_discord_trigger,
    event_label,
    natural_summary,
    normalize_discord_signal,
    signal_phrase,
)
from common.task_triggers import cheap_match, signal_fact, signal_matches, voice_transitions
from common.tasks import KIND_AT, KIND_EVENT
from cogs.chat.chat import _strip_task_subtext, _task_subtext


def _task(**kwargs):
    base = dict(
        kind=KIND_EVENT,
        user_id=1,
        scope="guild",
        channel_ids=[],
        instruction="Préviens-moi",
        recipe_json={},
        schedule_kind="once",
        max_fires=1,
        fires_count=0,
        cooldown_seconds=0,
        last_fired_at=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class VoiceTransitionTests(unittest.TestCase):
    def test_mute_is_not_a_signal(self):
        self.assertEqual(voice_transitions(4, 4, False, False), [])

    def test_join_and_leave(self):
        self.assertEqual(voice_transitions(0, 4, False, False), [("voice_join", 4)])
        self.assertEqual(voice_transitions(4, 0, False, False), [("voice_leave", 4)])

    def test_move_is_a_join_not_a_leave(self):
        self.assertEqual(voice_transitions(4, 9, False, False), [("voice_join", 9)])

    def test_go_live(self):
        self.assertEqual(
            voice_transitions(4, 4, False, True),
            [("voice_stream", 4)],
        )
        self.assertEqual(voice_transitions(4, 4, True, True), [])

    def test_join_and_stream_together(self):
        self.assertEqual(
            voice_transitions(0, 4, False, True),
            [("voice_join", 4), ("voice_stream", 4)],
        )


class SignalMatchTests(unittest.TestCase):
    def test_specific_user_and_channel(self):
        task = _task(trigger_json=build_discord_trigger(
            signal="voice_join", user_id=42, voice_channel_id=7,
        ))
        self.assertTrue(signal_matches(task, signal="voice_join", user_id=42, channel_id=7))
        self.assertFalse(signal_matches(task, signal="voice_join", user_id=43, channel_id=7))
        self.assertFalse(signal_matches(task, signal="voice_join", user_id=42, channel_id=8))
        self.assertFalse(signal_matches(task, signal="voice_leave", user_id=42, channel_id=7))

    def test_any_channel_when_unset(self):
        task = _task(trigger_json=build_discord_trigger(signal="voice_stream", user_id=42))
        self.assertTrue(signal_matches(task, signal="voice_stream", user_id=42, channel_id=3))

    def test_member_join_ignores_channel(self):
        task = _task(trigger_json=build_discord_trigger(signal="member_join", user_id=42))
        self.assertTrue(signal_matches(task, signal="member_join", user_id=42, channel_id=None))

    def test_self_target(self):
        trig = build_discord_trigger(signal="voice_join", user_id=1)
        trig["user_id"] = "self"
        task = _task(user_id=1, trigger_json=trig)
        self.assertTrue(signal_matches(task, signal="voice_join", user_id=1, channel_id=2))
        self.assertFalse(signal_matches(task, signal="voice_join", user_id=9, channel_id=2))

    def test_message_scan_ignores_discord_signals(self):
        task = _task(trigger_json=build_discord_trigger(signal="voice_join", user_id=7))
        msg = SimpleNamespace(
            channel=SimpleNamespace(id=1),
            author=SimpleNamespace(id=2),
            clean_content="voice_join:7",
            content="voice_join:7",
        )
        self.assertFalse(cheap_match(task, msg))


class SignalCopyTests(unittest.TestCase):
    def test_aliases_and_phrase(self):
        self.assertEqual(normalize_discord_signal("go-live"), "voice_stream")
        self.assertEqual(normalize_discord_signal("bonjour"), "")
        trig = build_discord_trigger(signal="voice_stream", user_id=42, voice_channel_id=3)
        self.assertEqual(signal_phrase(trig, owner_id=1), "<@42> lance un stream (<#3>)")
        self.assertEqual(signal_phrase(trig, owner_id=42), "tu lances un stream (<#3>)")
        self.assertEqual(event_label(trig), "lance un stream")
        task = _task(user_id=1, max_fires=1, trigger_json=trig)
        text = natural_summary(task)
        self.assertIn("une fois", text)
        self.assertIn("<@42>", text)
        self.assertNotIn("voice_stream:42", text)

    def test_fact_lines(self):
        self.assertEqual(signal_fact("voice_join", "Léa", "Général"), "Léa a rejoint Général")
        self.assertEqual(signal_fact("voice_leave", "Léa", "Général"), "Léa a quitté Général")
        self.assertEqual(
            signal_fact("voice_stream", "Léa", "Général"),
            "Léa a lancé un stream dans Général",
        )
        self.assertEqual(signal_fact("member_join", "Léa", "Srv"), "Léa a rejoint le serveur")


class TaskSubtextTests(unittest.TestCase):
    def test_clock_alert_always_has_subtext(self):
        when = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
        task = SimpleNamespace(kind=KIND_AT, execute_at=when)
        line = _task_subtext(task, "")
        self.assertTrue(line.startswith("-# **Programmé**"))
        self.assertIn(f"<t:{int(when.timestamp())}:f>", line)

    def test_event_detail_uses_listen_label(self):
        task = SimpleNamespace(kind=KIND_EVENT, execute_at=datetime.now(timezone.utc))
        line = _task_subtext(task, "Léa a rejoint Général")
        self.assertEqual(line, "-# **Écoute** · Léa a rejoint Général")

    def test_strip_keeps_other_subtext(self):
        raw = "résumé\n-# **Programmé** · hier\n-# autre note"
        self.assertEqual(_strip_task_subtext(raw), "résumé\n-# autre note")


if __name__ == "__main__":
    unittest.main()
