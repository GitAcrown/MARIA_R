"""Cog Layout — permet à l'IA de générer des LayoutView personnalisés via JSON."""

import logging
from datetime import datetime, timezone
from typing import Optional

import discord
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger("MARIA.Layout")

_MAX_BLOCKS  = 12
_MAX_CONTENT = 600


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
                    "Affiche un widget visuel personnalisé (carte, fiche, comparaison, liste structurée). "
                    "Utilise uniquement quand la mise en forme visuelle apporte vraiment par rapport à du texte brut "
                    "(ex : fiche avec image, comparaison côte à côte, récapitulatif multi-champs). "
                    "Pas pour les simples réponses conversationnelles. "
                    "commentary : ta réponse/intro affichée en tête du widget. "
                    "blocks : liste de blocs dans l'ordre — "
                    "text (markdown Discord : **gras**, ## titre, -# petit), "
                    "separator (ligne de séparation), "
                    "section (texte + image thumbnail optionnelle)."
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
                                    "enum":        ["text", "separator", "section"],
                                    "description": "text: bloc texte markdown. separator: ligne. section: texte + thumbnail.",
                                },
                                "content": {
                                    "type":        ["string", "null"],
                                    "description": "Texte markdown pour type text/section. null pour separator.",
                                },
                                "thumbnail_url": {
                                    "type":        ["string", "null"],
                                    "description": "URL https d'une image pour type section. null sinon.",
                                },
                            },
                            "required":             ["type", "content", "thumbnail_url"],
                            "additionalProperties": False,
                        },
                    },
                },
                function=self._tool_create_layout,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Layout(bot))
