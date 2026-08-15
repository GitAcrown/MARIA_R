"""Exécution Python restreinte : AST whitelist, builtins limités, timeout via subprocess."""

from __future__ import annotations

import ast
import datetime as dt
import decimal
import json
import math
import statistics
import sys
from io import StringIO
from typing import Any

_MAX_SOURCE = 2000
_TIMEOUT_SEC = 3

_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "__import__", "memoryview", "exit", "quit", "help", "copyright",
    "credits", "license", "classmethod", "staticmethod", "property",
    "super", "type",
})

_FORBIDDEN_AST: tuple[type, ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Match,
) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


class _SandboxError(Exception):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def _is_dunder(name: str) -> bool:
    return name.startswith("_") and name != "_result"


class _Checker(ast.NodeVisitor):
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr):
            raise _SandboxError(
                f"Attribut interdit : .{node.attr}",
                "Pas d'accès aux attributs internes (_xxx).",
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if _is_dunder(node.id):
            raise _SandboxError(
                f"Nom interdit : {node.id}",
                "Pas de noms commençant par _.",
            )
        if node.id in _FORBIDDEN_NAMES:
            raise _SandboxError(
                f"Nom interdit : {node.id}",
                "Builtins dangereux bloqués (open, eval, getattr…).",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
            raise _SandboxError(
                f"Appel interdit : {func.id}()",
                "Utilise math / datetime / print, pas open/eval/getattr.",
            )
        self.generic_visit(node)


def _validate(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise _SandboxError(
            f"Syntaxe Python invalide : {e.msg}",
            "Envoie du Python valide, ex. `math.factorial(12)` ou `print(2+2)`.",
        ) from e
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST):
            raise _SandboxError(
                f"Construction interdite : {type(node).__name__}",
                "Pas d'import, de classe, d'async ni de with. math/datetime sont déjà dispo.",
            )
    _Checker().visit(tree)
    return tree


def _capture_last_expr(tree: ast.Module) -> ast.Module:
    """Si le script finit par une expression, la stocke dans `_result`."""
    if not tree.body:
        return tree
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())],
            value=last.value,
        )
        ast.fix_missing_locations(tree)
    return tree


def _safe_globals() -> dict[str, Any]:
    import re as _re

    builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "reversed": reversed, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "frozenset": frozenset,
        "str": str, "int": int, "float": float, "bool": bool, "bytes": bytes,
        "print": print, "isinstance": isinstance, "pow": pow, "divmod": divmod,
        "all": all, "any": any, "map": map, "filter": filter, "repr": repr,
        "hex": hex, "oct": oct, "bin": bin, "chr": chr, "ord": ord,
        "format": format, "True": True, "False": False, "None": None,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "ZeroDivisionError": ZeroDivisionError, "ArithmeticError": ArithmeticError,
        "KeyError": KeyError, "IndexError": IndexError, "StopIteration": StopIteration,
    }
    return {
        "__builtins__": builtins,
        "math": math,
        "statistics": statistics,
        "datetime": dt,
        "decimal": decimal,
        "json": json,
        "re": _re,
        "timedelta": dt.timedelta,
        "timezone": dt.timezone,
        "date": dt.date,
        "time": dt.time,
    }


def execute(source: str) -> dict[str, Any]:
    """Exécute `source` et renvoie {ok, stdout, result} ou {error, hint}."""
    code = (source or "").strip()
    if not code:
        return {
            "error": "Code vide",
            "hint": "Envoie une expression ou un court script Python (math, dates, stats).",
        }
    if len(code) > _MAX_SOURCE:
        return {
            "error": f"Code trop long ({len(code)} > {_MAX_SOURCE} caractères)",
            "hint": "Coupe le script, ou enchaîne plusieurs appels run_python.",
        }
    try:
        tree = _capture_last_expr(_validate(code))
        compiled = compile(tree, "<maria>", "exec")
    except _SandboxError as e:
        return {"error": str(e), "hint": e.hint}
    except Exception as e:
        return {
            "error": f"Compilation impossible : {e}",
            "hint": "Simplifie le script (math, datetime, print).",
        }

    g = _safe_globals()
    g["_result"] = None
    buf = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        exec(compiled, g, g)  # noqa: S102 — sandbox AST déjà filtré
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "hint": "Corrige l'erreur, ou découpe le calcul en étapes plus simples.",
        }
    finally:
        sys.stdout = old_stdout

    stdout = buf.getvalue()
    result = g.get("_result")
    payload: dict[str, Any] = {"ok": True}
    if stdout:
        payload["stdout"] = stdout[:4000]
    if result is not None:
        try:
            json.dumps(result)
            payload["result"] = result
        except (TypeError, ValueError):
            payload["result"] = repr(result)[:2000]
    if "stdout" not in payload and "result" not in payload:
        payload["result"] = None
        payload["hint"] = "Rien à afficher : utilise print(...) ou termine par une expression."
    return payload


def main() -> None:
    raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
    out = execute(raw)
    sys.stdout.buffer.write(
        json.dumps(out, ensure_ascii=False, default=str).encode("utf-8")
    )


if __name__ == "__main__":
    main()
