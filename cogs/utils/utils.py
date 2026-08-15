"""Cog Utils — exécution Python restreinte (calculs, petits scripts)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

_SANDBOX = Path(__file__).resolve().parent / "sandbox.py"
_TIMEOUT = 3.5


def _run_sandbox(code: str) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(_SANDBOX)],
            input=code.encode("utf-8"),
            capture_output=True,
            timeout=_TIMEOUT,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"Exécution trop longue (>{int(_TIMEOUT)} s)",
            "hint": "Simplifie le script : pas de boucles infinies ni de gros calculs.",
        }
    except OSError as e:
        return {"error": f"Sandbox indisponible : {e}", "hint": "Réessaie plus tard."}

    raw = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if proc.returncode != 0 and not raw:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        return {
            "error": err[:500] or f"Sandbox exit {proc.returncode}",
            "hint": "Envoie du Python valide (math, datetime, print).",
        }
    if not raw:
        return {
            "error": "Sandbox sans résultat",
            "hint": "Termine par une expression ou un print().",
        }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Réponse sandbox illisible", "hint": "Réessaie avec un script plus simple."}
    return data if isinstance(data, dict) else {"error": "Réponse sandbox invalide"}


class Utils(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _tool_python(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        code = (tc.arguments.get("code") or "").strip()
        if not code:
            return ToolResponseRecord(
                tc.id,
                {
                    "error": "Code manquant",
                    "hint": "Envoie du Python, ex. `math.factorial(12)` ou un court script.",
                },
                datetime.now(timezone.utc),
            )
        data = await asyncio.to_thread(_run_sandbox, code)
        return ToolResponseRecord(tc.id, data, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="run_python",
                description=(
                    "Exécute un court programme Python pour calculer, convertir, "
                    "manipuler des dates ou faire des stats. "
                    "Modules déjà importés : math, statistics, datetime, decimal, json, re. "
                    "print() et la dernière expression sont renvoyés. "
                    "Pas de fichiers, pas de réseau, pas d'import, timeout 3 s. "
                    "INTERDIT : du langage naturel — uniquement du Python valide."
                ),
                properties={
                    "code": {
                        "type": "string",
                        "description": (
                            "Code Python (expression ou petit script). "
                            "Ex. `math.sqrt(2)` · `from datetime import date` est INTERDIT, "
                            "datetime est déjà disponible : `datetime.date.today()`."
                        ),
                    },
                },
                function=self._tool_python,
            ),
        ]


async def setup(bot):
    await bot.add_cog(Utils(bot))
