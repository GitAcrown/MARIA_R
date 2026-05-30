"""Cog Utils — outils texte et calcul."""

import re
from collections import Counter
from datetime import datetime, timezone

from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

try:
    import numexpr
    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False


class Utils(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # -- Calcul --------------------------------------------------------------

    def _tool_math(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        expr = (tc.arguments.get("expression") or "").strip()
        if not expr:
            return ToolResponseRecord(tc.id, {"error": "Expression manquante"}, datetime.now(timezone.utc))
        if not NUMEXPR_AVAILABLE:
            return ToolResponseRecord(tc.id, {"error": "Module numexpr indisponible"}, datetime.now(timezone.utc))
        try:
            result = numexpr.evaluate(expr)
            if hasattr(result, "item"):
                result = result.item()
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return ToolResponseRecord(tc.id, {"result": result}, datetime.now(timezone.utc))
        except Exception as e:
            return ToolResponseRecord(tc.id, {"error": str(e)}, datetime.now(timezone.utc))

    # -- Comptage ------------------------------------------------------------

    def _tool_count(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        text   = tc.arguments.get("text") or ""
        needle = tc.arguments.get("needle") or ""
        if not text or not needle:
            return ToolResponseRecord(tc.id, {"error": "text et needle requis"}, datetime.now(timezone.utc))
        case_sensitive = tc.arguments.get("case_sensitive") is True
        haystack = text if case_sensitive else text.lower()
        pattern  = needle if case_sensitive else needle.lower()
        count = haystack.count(pattern)
        return ToolResponseRecord(tc.id, {
            "needle": needle, "count": count,
            "note": f'"{needle}" apparaît {count} fois.',
        }, datetime.now(timezone.utc))

    # -- Stats texte ---------------------------------------------------------

    def _tool_text_stats(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        text = tc.arguments.get("text") or ""
        if not text:
            return ToolResponseRecord(tc.id, {"error": "Texte manquant"}, datetime.now(timezone.utc))
        words     = re.findall(r"\w+", text)
        sentences = re.split(r"[.!?…]+", text)
        sentences = [s for s in sentences if s.strip()]
        lines     = [l for l in text.splitlines() if l.strip()]
        freq      = Counter(w.lower() for w in words)
        top5      = freq.most_common(5)
        return ToolResponseRecord(tc.id, {
            "chars":     len(text),
            "chars_no_spaces": len(text.replace(" ", "")),
            "words":     len(words),
            "sentences": len(sentences),
            "lines":     len(lines),
            "top_words": [{"word": w, "count": c} for w, c in top5],
        }, datetime.now(timezone.utc))

    # -- Tri de liste --------------------------------------------------------

    def _tool_sort_list(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        items_raw = tc.arguments.get("items") or ""
        order     = (tc.arguments.get("order") or "asc").lower()
        by        = (tc.arguments.get("by") or "alpha").lower()
        if not items_raw:
            return ToolResponseRecord(tc.id, {"error": "Liste vide"}, datetime.now(timezone.utc))

        items = [i.strip() for i in re.split(r"[,;\n]+", items_raw) if i.strip()]
        reverse = order == "desc"

        if by == "length":
            items.sort(key=len, reverse=reverse)
        elif by == "numeric":
            try:
                items.sort(key=lambda x: float(x.replace(",", ".")), reverse=reverse)
            except ValueError:
                items.sort(key=str.lower, reverse=reverse)
        else:
            items.sort(key=str.lower, reverse=reverse)

        return ToolResponseRecord(tc.id, {
            "sorted": items,
            "count":  len(items),
        }, datetime.now(timezone.utc))

    # -- Outils enregistrés --------------------------------------------------

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="math_eval",
                description="Évalue une expression mathématique (+, -, *, /, **, sqrt, etc.).",
                properties={"expression": {"type": "string", "description": "Expression à évaluer"}},
                function=self._tool_math,
            ),
            Tool(
                name="count_occurrences",
                description=(
                    "Compte le nombre de fois qu'une chaîne apparaît dans un texte. "
                    "Utile pour 'combien de fois il dit X', compter des mots ou caractères précis."
                ),
                properties={
                    "text":           {"type": "string", "description": "Texte dans lequel chercher"},
                    "needle":         {"type": "string", "description": "Chaîne à compter"},
                    "case_sensitive": {"type": "boolean", "description": "Sensible à la casse (défaut false)"},
                },
                function=self._tool_count,
            ),
            Tool(
                name="text_stats",
                description=(
                    "Analyse un texte : nombre de caractères, mots, phrases, lignes, "
                    "et mots les plus fréquents. Utile pour des stats rapides sur un message ou document."
                ),
                properties={
                    "text": {"type": "string", "description": "Texte à analyser"},
                },
                function=self._tool_text_stats,
            ),
            Tool(
                name="sort_list",
                description=(
                    "Trie une liste d'éléments séparés par des virgules, points-virgules ou sauts de ligne. "
                    "Utile pour classer des noms, des chiffres, ou comparer des longueurs."
                ),
                properties={
                    "items": {"type": "string", "description": "Éléments séparés par des virgules ou sauts de ligne"},
                    "order": {"type": "string", "enum": ["asc", "desc"], "description": "Ordre croissant ou décroissant"},
                    "by":    {"type": "string", "enum": ["alpha", "length", "numeric"], "description": "Critère de tri"},
                },
                function=self._tool_sort_list,
            ),
        ]


async def setup(bot):
    await bot.add_cog(Utils(bot))
