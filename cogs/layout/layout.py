"""Cog Layout — rendu de tableaux ASCII (tabulate) en codeblock pour l'IA."""

import logging
from datetime import datetime, timezone

from discord.ext import commands

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Layout")

_MAX_COLS = 8
_MAX_ROWS = 20
_MAX_CELL = 40


def _render_table(headers: list, rows: list) -> str:
    """Génère un tableau ASCII dans un codeblock Discord."""
    if not rows and not headers:
        return ""

    headers = [str(h)[:_MAX_CELL] for h in (headers or [])[:_MAX_COLS]]
    rows    = [[str(c)[:_MAX_CELL] for c in row[:_MAX_COLS]] for row in rows[:_MAX_ROWS]]

    if _HAS_TABULATE:
        table = _tabulate(rows, headers=headers, tablefmt="simple")
    else:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))
        sep  = "  ".join("-" * w for w in col_widths)
        head = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        body = "\n".join(
            "  ".join((cell if i < len(row) else "").ljust(col_widths[i]) for i, cell in enumerate(row))
            for row in rows
        )
        table = f"{head}\n{sep}\n{body}" if headers else body

    return f"```\n{table}\n```"


class Layout(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _tool_render_table(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        headers = tc.arguments.get("headers") or []
        rows    = tc.arguments.get("rows") or []
        if not rows:
            return ToolResponseRecord(tc.id, {"error": "Aucune ligne fournie."}, datetime.now(timezone.utc))
        table = _render_table(headers, rows)
        if not table:
            return ToolResponseRecord(tc.id, {"error": "Tableau vide."}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "table": table,
            "note":  "Colle ce bloc tel quel dans ta réponse, sans le modifier.",
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="render_table",
                description=(
                    "Met en forme un tableau aligné (tabulate) renvoyé dans un codeblock prêt à coller. "
                    "À utiliser dès qu'une réponse contient un tableau, pour un rendu propre et lisible. "
                    "Fournir headers (colonnes) et rows (liste de lignes, chaque ligne = liste de cellules)."
                ),
                properties={
                    "headers": {
                        "type":        "array",
                        "description": "Noms des colonnes.",
                        "items":       {"type": "string"},
                    },
                    "rows": {
                        "type":        "array",
                        "description": "Lignes du tableau (liste de listes de chaînes).",
                        "items": {
                            "type":  "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                function=self._tool_render_table,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Layout(bot))
