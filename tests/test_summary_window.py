"""Fenêtre du résumé de salon. « depuis 16h » = 16:00, pas les 24 dernières heures."""

from __future__ import annotations

import unittest
from datetime import datetime

from common.timezones import PARIS_TZ
from cogs.chat.tools_summary import (
    _DEFAULT_LIMIT,
    _WINDOW_CAP_DAY,
    _WINDOW_CAP_SHORT,
    build_channel_summary_tools,
    resolve_summary_window,
)

_NOW = datetime(2026, 10, 9, 19, 32, tzinfo=PARIS_TZ)


def _window(text: str, **kwargs):
    return resolve_summary_window(text, now=_NOW, **kwargs)


def _paris(moment: datetime) -> datetime:
    return moment.astimezone(PARIS_TZ)


class SummaryWindowTests(unittest.TestCase):
    def test_depuis_16h_is_clock_not_last_24h(self):
        window = _window("un résumé depuis 16h", hours=24)
        self.assertEqual(window.source, "clock")
        self.assertIsNotNone(window.after)
        start = _paris(window.after)
        self.assertEqual((start.hour, start.minute, start.date()), (16, 0, _NOW.date()))
        span_h = (_NOW - start).total_seconds() / 3600
        self.assertLess(span_h, 5)
        self.assertGreaterEqual(window.limit, 2000)
        self.assertEqual(window.limit, _WINDOW_CAP_SHORT)

    def test_16h30_and_a_partir_de(self):
        for text in ("résumé depuis 16h30", "à partir de 16:00"):
            window = _window(text, hours=24)
            self.assertEqual(window.source, "clock", text)
            start = _paris(window.after)
            self.assertEqual(start.hour, 16, text)

    def test_future_clock_uses_yesterday(self):
        morning = datetime(2026, 10, 9, 15, 0, tzinfo=PARIS_TZ)
        window = resolve_summary_window("depuis 16h", now=morning, hours=24)
        start = _paris(window.after)
        self.assertEqual(start.date().day, 8)
        self.assertEqual((start.hour, start.minute), (16, 0))

    def test_heures_word_is_duration(self):
        window = _window("résumé depuis 16 heures", hours=24)
        self.assertEqual(window.source, "duration")
        start = _paris(window.after)
        self.assertEqual((start.hour, start.minute), (3, 32))
        self.assertEqual(window.limit, _WINDOW_CAP_DAY)

    def test_short_h_is_duration(self):
        window = _window("depuis 2h", hours=24)
        self.assertEqual(window.source, "duration")
        start = _paris(window.after)
        self.assertEqual((start.hour, start.minute), (17, 32))
        self.assertEqual(window.limit, _WINDOW_CAP_SHORT)

    def test_last_n_hours(self):
        window = _window("les 3 dernières heures")
        self.assertEqual(window.source, "duration")
        span_h = (_NOW - _paris(window.after)).total_seconds() / 3600
        self.assertAlmostEqual(span_h, 3, places=2)

    def test_today_is_midnight_not_rolling_24h(self):
        window = _window("résumé d'aujourd'hui", hours=24)
        self.assertEqual(window.source, "period")
        start = _paris(window.after)
        self.assertEqual((start.hour, start.minute, start.date()), (0, 0, _NOW.date()))
        self.assertGreaterEqual(window.limit, 2000)

    def test_bare_summary_ignores_hours_24(self):
        window = _window("résume le salon", hours=24, limit=60)
        self.assertEqual(window.source, "recent")
        self.assertIsNone(window.after)
        self.assertGreaterEqual(window.limit, 180)
        self.assertEqual(window.limit, _DEFAULT_LIMIT)

    def test_modest_model_hours_without_text(self):
        window = _window("", hours=4)
        self.assertEqual(window.source, "duration")
        span_h = (_NOW - _paris(window.after)).total_seconds() / 3600
        self.assertAlmostEqual(span_h, 4, places=2)

    def test_since_arg_when_text_has_no_clock(self):
        window = _window("les décisions", since="16:00", hours=24)
        self.assertEqual(window.source, "since")
        start = _paris(window.after)
        self.assertEqual((start.hour, start.minute), (16, 0))

    def test_schema_does_not_ask_for_a_full_day(self):
        tool = build_channel_summary_tools(object(), model="x")[0]
        self.assertNotIn("toute la journée", tool.description)
        self.assertNotIn("24", tool.properties["hours"]["description"])
        self.assertIn("since", tool.properties)


if __name__ == "__main__":
    unittest.main()
