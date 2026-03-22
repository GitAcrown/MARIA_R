"""Cog Utils — calcul mathématique."""

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

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="math_eval",
                description="Évalue une expression mathématique (+, -, *, /, **, sqrt, etc.).",
                properties={"expression": {"type": "string", "description": "Expression à évaluer"}},
                function=self._tool_math,
            )
        ]


async def setup(bot):
    await bot.add_cog(Utils(bot))
