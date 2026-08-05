"""Cog Layout — rendu de tableaux ASCII (tabulate) et de widgets libres pour l'IA."""

from datetime import datetime, timezone
from typing import Optional

import discord
from discord.ext import commands

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.widget_catalog import WIDGET_CANON_EXAMPLES, WIDGET_SPEC_SCHEMA, render_free_widget
from common.widgets import register_widget, unregister_widget

_MAX_COLS = 8
_MAX_ROWS = 20
_MAX_CELL = 40


def build_render_widget_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget libre — rend le spec produit par l'outil render_widget."""
    if not isinstance(data, dict) or "error" in data:
        return None
    return render_free_widget(data.get("spec"), commentary=commentary)


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

    def _tool_render_widget(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        spec = tc.arguments.get("spec")
        if not isinstance(spec, dict):
            return ToolResponseRecord(tc.id, {"error": "spec manquant ou invalide."}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "_tool":        "render_widget",
            "_llm_summary": "Widget affiché dans le salon.",
            "spec":         spec,
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
            Tool(
                name="render_widget",
                description=(
                    "Compose et affiche un widget visuel libre (blocs : text, separator, stat_row, "
                    "thumbnail, gallery, footer). RARE — seulement contenu dense à garder sous les yeux "
                    "(recette complète, tutoriel multi-étapes, classement/comparatif substantiel). Jamais "
                    "pour une liste courte, des tips, un avis/blague/une phrase (markdown en tchat), ni pour "
                    "remplacer un widget dédié (météo/film/jeu/foot/rappels). "
                    "thumbnail/gallery : uniquement une URL déjà fiable en contexte (avatar Discord, "
                    "pochette d'un outil dédié) — jamais via search_images (images web souvent cassées sur "
                    "Discord, anti-hotlink/liens temporaires) ; sans URL fiable, ignore l'image. "
                    "Reste sobre, footer sourcé si pertinent.\n\n"
                    + WIDGET_CANON_EXAMPLES
                ),
                properties={
                    "spec": {**WIDGET_SPEC_SCHEMA, "description": "Structure du widget à afficher."},
                },
                function=self._tool_render_widget,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Layout(bot))
    register_widget("render_widget", build_render_widget_view)


async def teardown(bot: commands.Bot) -> None:
    unregister_widget("render_widget")
