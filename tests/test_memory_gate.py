"""Le tchat consulte la mémoire. Seul un outil live sûr la saute."""

from __future__ import annotations

import unittest

from cogs.chat.chat import should_skip_memory_rag


class MemoryGateTests(unittest.TestCase):
    def test_chat_consults_memory(self):
        self.assertFalse(should_skip_memory_rag("none", 0.9, "tu préfères quoi comme films"))
        self.assertFalse(should_skip_memory_rag(None, 0.0, "salut"))

    def test_live_tool_skips_memory(self):
        self.assertTrue(should_skip_memory_rag("weather", 0.8, "météo à Lyon"))
        self.assertFalse(should_skip_memory_rag("weather", 0.2, "météo à Lyon"))

    def test_explicit_recall_always_consults(self):
        self.assertFalse(
            should_skip_memory_rag("weather", 0.9, "tu te souviens de la météo qu'il aime"),
        )


if __name__ == "__main__":
    unittest.main()
