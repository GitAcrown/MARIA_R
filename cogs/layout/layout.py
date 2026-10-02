"""Cog Layout — rendu de tableaux ASCII (tabulate) et de widgets libres pour l'IA."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import discord
from discord import app_commands
from discord.ext import commands

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

from common.bookmarks import attach_bookmark_button, list_for_user
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.widget_catalog import WIDGET_CANON_EXAMPLES, WIDGET_SPEC_SCHEMA, render_free_widget
from common.widgets import register_widget, unregister_widget
from cogs.layout.views import BookmarksView

_MAX_COLS = 8
_MAX_ROWS = 20
_MAX_CELL = 40


def build_render_widget_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Builder du widget libre — rend le spec produit par l'outil render_widget."""
    if not isinstance(data, dict) or "error" in data:
        return None
    view = render_free_widget(data.get("spec"), commentary=commentary)
    if view is None:
        return None
    attach_bookmark_button(view, data.get("spec"), commentary=commentary)
    return view


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "oui" if value else "non"
    return str(value).replace("\n", " ").strip()[:_MAX_CELL]


def _normalize_table(
    headers: Any, rows: Any,
) -> tuple[list[str], list[list[str]]]:
    """Accepte les formes bancales que le modèle envoie souvent (liste plate, dicts…)."""
    if headers is None:
        headers = []
    if isinstance(headers, str):
        headers = [headers]
    if not isinstance(headers, (list, tuple)):
        headers = [headers]
    headers = [_cell(h) for h in list(headers)[:_MAX_COLS]]

    if rows is None:
        rows = []
    if isinstance(rows, str):
        rows = [[rows]]
    if not isinstance(rows, (list, tuple)):
        rows = [[rows]]

    raw = list(rows)
    # Liste plate de scalaires → une ligne (si ça matche les headers) ou une colonne.
    if raw and not any(isinstance(r, (list, tuple, dict)) for r in raw):
        if headers and len(raw) == len(headers):
            raw = [raw]
        else:
            raw = [[c] for c in raw]

    out: list[list[str]] = []
    for row in raw[:_MAX_ROWS]:
        if isinstance(row, dict):
            if headers:
                out.append([_cell(row.get(h, "")) for h in headers])
            else:
                # Première ligne dict : les clés deviennent les headers.
                if not headers:
                    headers = [_cell(k) for k in list(row.keys())[:_MAX_COLS]]
                out.append([_cell(row.get(h, "")) for h in headers])
        elif isinstance(row, (list, tuple)):
            out.append([_cell(c) for c in list(row)[:_MAX_COLS]])
        else:
            out.append([_cell(row)])

    ncols = max(len(headers), max((len(r) for r in out), default=0))
    if ncols == 0:
        return [], []
    if not headers:
        headers = [f"Col{i + 1}" for i in range(ncols)]
    headers = (headers + [""] * ncols)[:ncols]
    out = [(r + [""] * ncols)[:ncols] for r in out]
    return headers, out


def _render_table(headers: list, rows: list) -> str:
    """Génère un tableau ASCII dans un codeblock Discord."""
    headers, rows = _normalize_table(headers, rows)
    if not rows:
        return ""

    if _HAS_TABULATE:
        table = _tabulate(rows, headers=headers, tablefmt="simple")
    else:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        sep = "  ".join("-" * w for w in col_widths)
        head = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        body = "\n".join(
            "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
            for row in rows
        )
        table = f"{head}\n{sep}\n{body}"

    # Discord : rester sous 1900 pour laisser de la marge au commentaire.
    if len(table) > 1900:
        table = table[:1897].rstrip() + "…"
    return f"```\n{table}\n```"


def _spec_llm_summary(spec: dict) -> str:
    bits: list[str] = []
    title = (spec.get("title") or "").strip()
    if title:
        bits.append(title)
    for block in spec.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            bits.append((block.get("content") or "").strip()[:400])
        elif kind == "footer":
            bits.append((block.get("text") or "").strip())
        elif kind == "stat_row":
            items = block.get("items") or block.get("stats") or []
            bits.append(" · ".join(str(x) for x in items[:8]))
    text = " | ".join(x for x in bits if x)
    return (text[:1500] if text else "Widget affiché dans le salon.")


class Layout(commands.Cog):

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    def _tool_render_table(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        headers = tc.arguments.get("headers") or []
        rows = tc.arguments.get("rows") or []
        table = _render_table(headers, rows)
        if not table:
            return ToolResponseRecord(
                tc.id, {"error": "Aucune ligne utilisable."}, datetime.now(timezone.utc),
            )
        return ToolResponseRecord(tc.id, {
            "_tool": "render_table",
            "_llm_summary": (
                "Tableau affiché dans le salon. "
                "Commente en une phrase max, ou ne dis rien. Ne recolle pas le tableau."
            ),
            "table": table,
        }, datetime.now(timezone.utc))

    def _tool_render_widget(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        spec = tc.arguments.get("spec")
        if not isinstance(spec, dict):
            return ToolResponseRecord(tc.id, {"error": "spec manquant ou invalide."}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "_tool":        "render_widget",
            "_llm_summary": _spec_llm_summary(spec),
            "spec":         spec,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="render_table",
                description=(
                    "Affiche un tableau aligné (codeblock) directement dans le salon. "
                    "À utiliser dès qu'une réponse mérite un tableau lisible. "
                    "Fournir headers (colonnes) et rows (liste de lignes = listes de cellules). "
                    "Ne pas recopier le tableau dans le message : il est posté automatiquement."
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
                optional_props=["headers"],
                function=self._tool_render_table,
            ),
            Tool(
                name="render_widget",
                description=(
                    "Compose un widget visuel libre (blocs : text, separator, stat_row, "
                    "thumbnail, gallery, footer). Défaut = tchat, pas cet outil. "
                    "À appeler seulement pour une recette complète, un tuto multi-étapes, "
                    "un comparatif dense, ou une demande explicite de fiche/layout. "
                    "Le widget EST alors la réponse (contenu complet dans spec). "
                    "Jamais pour une question directe, un avis, une définition, une liste courte, "
                    "ni pour remplacer un widget dédié (météo/film/jeu/foot/tâches/transports). "
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

    @app_commands.command(name="signets", description="Tes fiches enregistrées")
    async def cmd_bookmarks(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        items = await asyncio.to_thread(list_for_user, interaction.user.id)
        await interaction.followup.send(
            view=BookmarksView(interaction.user.id, items),
            ephemeral=True,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Layout(bot))
    register_widget("render_widget", build_render_widget_view)


async def teardown(bot: commands.Bot) -> None:
    unregister_widget("render_widget")
