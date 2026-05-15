"""Cog Layout — permet à l'IA de générer des LayoutView personnalisés via JSON."""

import logging
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

logger = logging.getLogger("MARIA.Layout")

_MAX_BLOCKS  = 12
_MAX_CONTENT = 600
_MAX_COLS    = 8
_MAX_ROWS    = 20


# ---------------------------------------------------------------------------
# Rendu tableau
# ---------------------------------------------------------------------------

def _render_table(headers: list, rows: list) -> str:
    """Génère un tableau ASCII dans un codeblock Discord."""
    if not rows and not headers:
        return ""

    # Sanitize
    headers = [str(h)[:40] for h in (headers or [])[:_MAX_COLS]]
    rows    = [[str(c)[:40] for c in row[:_MAX_COLS]] for row in rows[:_MAX_ROWS]]

    if _HAS_TABULATE:
        table = _tabulate(rows, headers=headers, tablefmt="simple")
    else:
        # Fallback manuel si tabulate absent
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


# ---------------------------------------------------------------------------
# Builder LayoutView
# ---------------------------------------------------------------------------

def build_custom_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit un LayoutView depuis la structure JSON fournie par l'IA.
    Le commentary passé en paramètre est ignoré : l'IA le fournit dans data["commentary"].
    """
    blocks = data.get("blocks")
    if not blocks:
        return None

    header_text = (data.get("commentary") or "").strip()

    children: list = []
    for block in blocks[:_MAX_BLOCKS]:
        btype   = (block.get("type") or "text").strip()
        content = (block.get("content") or "").strip()

        if btype == "separator":
            children.append(discord.ui.Separator())

        elif btype == "section":
            thumb_url  = (block.get("thumbnail_url") or "").strip()
            text_block = discord.ui.TextDisplay(content[:_MAX_CONTENT] if content else "\u200b")
            try:
                if thumb_url.startswith("https://"):
                    thumb = discord.ui.Thumbnail(discord.ui.UnfurledMediaItem(url=thumb_url))
                    children.append(discord.ui.Section(text_block, accessory=thumb))
                else:
                    children.append(text_block)
            except Exception:
                children.append(text_block)

        elif btype == "table":
            headers = block.get("headers") or []
            rows    = block.get("rows") or []
            table_text = _render_table(headers, rows)
            if table_text:
                children.append(discord.ui.TextDisplay(table_text))

        else:  # text
            if content:
                children.append(discord.ui.TextDisplay(content[:_MAX_CONTENT]))

    if not children:
        return None

    view = discord.ui.LayoutView(timeout=None)
    if header_text:
        view.add_item(discord.ui.TextDisplay(header_text))
        view.add_item(discord.ui.Separator())
    view.add_item(discord.ui.Container(*children))
    return view


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Layout(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _tool_create_layout(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        commentary = (tc.arguments.get("commentary") or "").strip()
        blocks     = tc.arguments.get("blocks") or []

        if not blocks:
            return ToolResponseRecord(
                tc.id, {"error": "Aucun bloc fourni."}, datetime.now(timezone.utc)
            )

        llm_summary = f"Widget personnalisé affiché. {commentary[:200]}" if commentary else "Widget personnalisé affiché."

        return ToolResponseRecord(tc.id, {
            "_tool":        "create_layout",
            "_llm_summary": llm_summary,
            "commentary":   commentary,
            "blocks":       blocks,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="create_layout",
                description=(
                    "Affiche un widget visuel personnalisé (carte, fiche, comparaison, tableau, liste structurée). "
                    "Utilise uniquement quand la mise en forme visuelle apporte vraiment par rapport à du texte brut. "
                    "Pas pour les réponses conversationnelles simples. "
                    "commentary : ta réponse/intro affichée en tête du widget. "
                    "Blocs disponibles : "
                    "text (markdown Discord : **gras**, ## titre, -# petit texte) ; "
                    "separator (ligne de séparation) ; "
                    "section (texte + image thumbnail optionnelle) ; "
                    "table (tableau formaté automatiquement — fournir headers + rows, PAS de markdown table manuel)."
                ),
                properties={
                    "commentary": {
                        "type":        "string",
                        "description": "Réponse ou intro de l'IA, affichée au-dessus du widget. Peut être vide.",
                    },
                    "blocks": {
                        "type":        "array",
                        "description": "Blocs du widget dans l'ordre d'affichage (max 12).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type":        "string",
                                    "enum":        ["text", "separator", "section", "table"],
                                    "description": "text: markdown. separator: ligne. section: texte+thumbnail. table: tableau ASCII auto.",
                                },
                                "content": {
                                    "type":        ["string", "null"],
                                    "description": "Texte markdown pour text/section. null pour separator et table.",
                                },
                                "thumbnail_url": {
                                    "type":        ["string", "null"],
                                    "description": "URL https image pour section. null sinon.",
                                },
                                "headers": {
                                    "type":        ["array", "null"],
                                    "description": "Noms de colonnes pour type table. null sinon.",
                                    "items":       {"type": "string"},
                                },
                                "rows": {
                                    "type":        ["array", "null"],
                                    "description": "Lignes du tableau pour type table (liste de listes de strings). null sinon.",
                                    "items": {
                                        "type":  "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "required":             ["type", "content", "thumbnail_url", "headers", "rows"],
                            "additionalProperties": False,
                        },
                    },
                },
                function=self._tool_create_layout,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Layout(bot))
