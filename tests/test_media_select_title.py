"""Le select des résultats média prend la place du titre de l'œuvre."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import discord

import common.media_hub  # noqa: F401  — enregistre le rendu
from common.dyn_widgets import _Record, render_record


def _record(kind: str, hits: list[dict]) -> _Record:
    return _Record(
        id="abcd1234",
        kind="media_hub",
        payload={"kind": kind, "hits": hits},
        commentary="",
        selected=0,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        channel_id=0,
        message_id=0,
        stripped=False,
    )


def _texts(item) -> list[str]:
    found: list[str] = []
    if isinstance(item, discord.ui.TextDisplay):
        found.append(item.content)
    for child in getattr(item, "children", []) or []:
        found.extend(_texts(child))
    return found


def _movies() -> list[dict]:
    titles = ["Insidious", "Insidious 2", "Insidious 3", "Insidious 4"]
    return [
        {
            "id": i + 1,
            "media_type": "movie",
            "title": title,
            "release_date": "2011-04-01",
            "overview": "Une famille.",
            "vote_average": 6.8,
            "vote_count": 100,
            "genres": [{"name": "Horreur"}, {"name": "Thriller"}],
        }
        for i, title in enumerate(titles)
    ]


class MediaSelectTitleTests(unittest.TestCase):
    def test_select_replaces_film_title(self):
        view = render_record(_record("tmdb", _movies()), live=True)
        self.assertIsNotNone(view)
        card = view.children[0]
        self.assertIsInstance(card, discord.ui.Container)
        self.assertIsInstance(card.children[0], discord.ui.ActionRow)
        self.assertFalse(any(isinstance(item, discord.ui.ActionRow) for item in view.children))
        blob = "\n".join(_texts(card))
        self.assertNotIn("##", blob)
        self.assertNotIn("Insidious", blob)
        self.assertIn("Horreur", blob)
        select = card.children[0].children[0].item
        self.assertEqual(select.options[0].label, "Insidious (2011)")
        self.assertTrue(select.options[0].default)

    def test_title_returns_when_the_select_is_gone(self):
        view = render_record(_record("tmdb", _movies()), live=False)
        card = view.children[0]
        self.assertIsInstance(card, discord.ui.Container)
        self.assertNotIsInstance(card.children[0], discord.ui.ActionRow)
        self.assertIn("##", "\n".join(_texts(card)))
        self.assertIn("Insidious", "\n".join(_texts(card)))

    def test_few_results_keep_buttons_and_title(self):
        view = render_record(_record("tmdb", _movies()[:2]), live=True)
        self.assertIsInstance(view.children[0], discord.ui.ActionRow)
        card = view.children[-1]
        self.assertIn("##", "\n".join(_texts(card)))

    def test_select_replaces_game_and_track_titles(self):
        games = [
            {
                "steam_appid": i,
                "name": f"Jeu {i}",
                "genres": [{"description": "Action"}],
                "short_description": "Un jeu.",
            }
            for i in range(1, 5)
        ]
        tracks = [
            {
                "id": str(i),
                "name": f"Morceau {i}",
                "artists": [{"name": "Artiste"}],
                "album": {"name": "Album", "release_date": "2020-01-01"},
            }
            for i in range(1, 5)
        ]
        for kind, hits, gone, kept in (
            ("steam", games, "Jeu 1", "Action"),
            ("spotify", tracks, "Morceau 1", "Artiste"),
        ):
            view = render_record(_record(kind, hits), live=True)
            card = view.children[0]
            self.assertIsInstance(card.children[0], discord.ui.ActionRow, kind)
            blob = "\n".join(_texts(card))
            self.assertNotIn("##", blob, kind)
            self.assertNotIn(gone, blob, kind)
            self.assertIn(kept, blob, kind)


if __name__ == "__main__":
    unittest.main()
