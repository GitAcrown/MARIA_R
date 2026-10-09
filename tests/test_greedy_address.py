"""Filtre greedy, fatigue par personne, historique allégé. Sans Discord ni JEV."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from common.attention import (
    FATIGUE_BUMP,
    FATIGUE_EXHAUSTED,
    FATIGUE_HALF_LIFE,
    FATIGUE_TIRED,
    MemberFatigue,
    SocialFocus,
    mention_becomes_emoji,
)
from common.greedy_address import (
    GREEDY_IGNORE_CONFIDENCE,
    finalize_greedy_choice,
    middle_is_address,
    name_hit_kind,
)
from common.llm.context import (
    CONTEXT_KEEP_RECENT,
    ConversationContext,
    TextComponent,
    ToolResponseRecord,
)
from common.llm.session import SYSTEM_NOTE_HISTORY_CAP
from common.llm.typesafe_client import REACT_VERDICT_CONFIDENCE
from cogs.chat.chat import DEV_PROMPT_BASE, _TASKS_CHAT_PROMPT
from cogs.chat.config import CONTEXT_AGE_HOURS


def _route(content: str, name: str = "Maria") -> str:
    kind = name_hit_kind(content, name)
    if kind == "edge":
        return "edge"
    if kind == "middle" and middle_is_address(content):
        return "fast"
    if kind in ("middle", "list"):
        return "jev"
    return "absent"


class AddressTests(unittest.TestCase):
    def test_edge_answers_without_jev(self) -> None:
        for text in (
            "Maria ?",
            "Maria tu penses quoi",
            "Maria devrait venir",
            "t'en penses quoi Maria",
            "hey Maria",
            "Maria.",
        ):
            self.assertEqual(_route(text), "edge", text)

    def test_middle_question_is_fast(self) -> None:
        self.assertEqual(_route("au fait Maria c'est quoi le TCP"), "fast")
        self.assertEqual(_route("dis Maria c'est quoi X"), "edge")

    def test_name_drop_stays_on_jev(self) -> None:
        self.assertEqual(_route("Maria et Bob"), "jev")
        self.assertEqual(_route("Maria, Léa"), "jev")
        self.assertEqual(_route("faut que Maria fasse ça"), "jev")

    def test_ignore_needs_high_confidence(self) -> None:
        low = GREEDY_IGNORE_CONFIDENCE - 0.01
        self.assertEqual(
            finalize_greedy_choice("ignore", low, question=False, react_min=REACT_VERDICT_CONFIDENCE),
            "respond",
        )
        self.assertEqual(
            finalize_greedy_choice(
                "ignore", GREEDY_IGNORE_CONFIDENCE, question=False, react_min=REACT_VERDICT_CONFIDENCE,
            ),
            "ignore",
        )

    def test_react_does_not_swallow_a_question(self) -> None:
        self.assertEqual(
            finalize_greedy_choice("react", 0.9, question=True, react_min=REACT_VERDICT_CONFIDENCE),
            "respond",
        )
        self.assertEqual(
            finalize_greedy_choice("react", 0.2, question=False, react_min=REACT_VERDICT_CONFIDENCE),
            "ignore",
        )


class FatigueTests(unittest.TestCase):
    def test_many_people_do_not_stack(self) -> None:
        fat = MemberFatigue()
        for user in range(10):
            fat.bump(1, user)
            self.assertLess(fat.value(1, user), FATIGUE_TIRED)

    def test_quarter_hour_of_one_person_stays_under_tired(self) -> None:
        fat = MemberFatigue()
        clock = [0.0]
        fat._clock = lambda: clock[0]
        for _ in range(10):
            fat.bump(1, 7)
            clock[0] += 90
        self.assertLess(fat.value(1, 7), FATIGUE_TIRED)
        self.assertEqual(FATIGUE_HALF_LIFE, 8 * 60)
        self.assertEqual(FATIGUE_BUMP, 0.4)

    def test_exhausted_is_personal(self) -> None:
        fat = MemberFatigue()
        fat._clock = lambda: 1000.0
        for _ in range(20):
            fat.bump(1, 7)
        self.assertGreaterEqual(fat.value(1, 7) + 1e-6, FATIGUE_EXHAUSTED)
        self.assertEqual(fat.value(1, 8), 0.0)
        self.assertFalse(mention_becomes_emoji(fat.value(1, 8), hot=False, question=False))
        self.assertTrue(mention_becomes_emoji(fat.value(1, 7), hot=False, question=False))
        self.assertFalse(mention_becomes_emoji(fat.value(1, 7), hot=False, question=True))

    def test_followup_question_holds_until_exhausted(self) -> None:
        focus = SocialFocus()
        kept = focus.soften_followup(
            "respond",
            attention=0.2,
            chain_depth=3,
            fatigue=FATIGUE_TIRED,
            is_addressee=True,
            is_question=True,
        )
        self.assertEqual(kept, "respond")
        dropped = focus.soften_followup(
            "respond",
            attention=0.2,
            chain_depth=3,
            fatigue=FATIGUE_EXHAUSTED,
            is_addressee=True,
            is_question=True,
        )
        self.assertNotEqual(dropped, "respond")


class HistoryTests(unittest.TestCase):
    def test_noise_is_not_folded_into_the_summary(self) -> None:
        ctx = ConversationContext("dev", context_window=0)
        now = datetime.now(timezone.utc)
        chatter = ctx.add_user_message(
            [TextComponent("[contexte] [12:00] Bob: on parle d'autre chose pendant un bon moment")],
            name="bob_1",
            context_only=True,
        )
        note = ctx.add_user_message(
            [TextComponent("[SYSTEM] Réaction 😂 sur le message de Bob.")],
            name="system",
        )
        tool = ToolResponseRecord(
            "call-1",
            {"_tool": "search_web", "results": [{"title": "x" * 400}]},
            now,
        )
        ctx._fold_evicted([chatter, note, tool])
        self.assertEqual(ctx.session_summary, "")

    def test_tool_history_keeps_a_short_line(self) -> None:
        tool = ToolResponseRecord(
            "call-1",
            {"_tool": "search_web", "results": [{"blob": "x" * 800}]},
            datetime.now(timezone.utc),
        )
        tool.compact_for_history()
        content = tool.to_payload()["content"]
        self.assertLessEqual(len(content), 160)
        self.assertNotIn("xxxx", content)
        self.assertEqual(SYSTEM_NOTE_HISTORY_CAP, 160)
        self.assertEqual(CONTEXT_AGE_HOURS, 1)
        self.assertEqual(CONTEXT_KEEP_RECENT, 4)

    def test_base_prompt_has_no_task_dsl_or_style_draw(self) -> None:
        self.assertNotIn("schedule_task", DEV_PROMPT_BASE)
        self.assertNotIn("style_ctx", DEV_PROMPT_BASE)
        self.assertIn("schedule_task", _TASKS_CHAT_PROMPT)
        rendered = DEV_PROMPT_BASE.format(
            bot_name="Maria",
            model="m",
            weekday="Friday",
            datetime="2026-10-09 19:00",
            channel_ctx="",
            self_ctx="",
            profile_ctx="",
            memory_ctx="",
            session_ctx="",
            capability_ctx="",
            poll_ctx="",
            tasks_ctx="",
            silence_ctx="",
        )
        self.assertNotIn("schedule_task", rendered)
        with_tasks = DEV_PROMPT_BASE.format(
            bot_name="Maria",
            model="m",
            weekday="Friday",
            datetime="2026-10-09 19:00",
            channel_ctx="",
            self_ctx="",
            profile_ctx="",
            memory_ctx="",
            session_ctx="",
            capability_ctx="",
            poll_ctx="",
            tasks_ctx=_TASKS_CHAT_PROMPT,
            silence_ctx="",
        )
        self.assertIn("schedule_task", with_tasks)


if __name__ == "__main__":
    unittest.main()
