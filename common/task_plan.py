"""Plan d'une tâche : déclencheur (ailleurs) + action + condition + suite.

Le déclencheur (horloge, écoute, boucle) vit sur la tâche. Ici, seulement
ce qui se passe quand il part :

1. condition `before` — filtre, avant l'action ;
2. action primaire optionnelle (lire une page, chercher, prompt, poster) ;
3. condition `after` — juge le résultat de l'action ;
4. action secondaire optionnelle, uniquement si la condition tient.

Sans condition, la secondaire part à chaque déclenchement.
Condition fausse ou invérifiable → silence, aucune secondaire.
"""

from __future__ import annotations

from typing import Any, Optional

from common.tasks import KIND_WATCH, WATCH_INTERVAL_MIN_MINUTES

PRIMARY_TYPES = ("read_url", "web_search", "prompt", "post")
_OPS = ("lt", "lte", "gt", "change", "lt_prev", "gt_prev")
_REL_OPS = ("lt_prev", "gt_prev")
_OP_SYMBOL = {"lt": "<", "lte": "≤", "gt": ">", "change": "Δ", "lt_prev": "<", "gt_prev": ">"}
_OP_WORD = {
    "lt": "sous",
    "lte": "à ou sous",
    "gt": "au-dessus de",
    "change": "écarté de",
    "lt_prev": "plus bas que le dernier relevé",
    "gt_prev": "plus haut que le dernier relevé",
}


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _mode(mode: str) -> str:
    return mode if mode in ("generate", "verbatim", "ping_only") else "generate"


def _threshold_value(cond: dict) -> float:
    try:
        return float(cond.get("threshold") or 0)
    except (TypeError, ValueError):
        return 0.0


def condition_label(cond: Optional[dict]) -> str:
    if not isinstance(cond, dict):
        return ""
    if cond.get("type") == "price":
        op = str(cond.get("op") or "lt")
        margin = _threshold_value(cond)
        if op == "lt_prev":
            return f"prix < dernier relevé de {margin:g} €" if margin > 0 else "prix < dernier relevé"
        if op == "gt_prev":
            return f"prix > dernier relevé de {margin:g} €" if margin > 0 else "prix > dernier relevé"
        try:
            thr = f"{float(cond.get('threshold')):g}"
        except (TypeError, ValueError):
            thr = "?"
        symbol = _OP_SYMBOL.get(op, "<")
        return f"prix {symbol} {thr} €"
    return _clip(str(cond.get("text") or "condition"), 80)


def compile_plan(
    *,
    instruction: str = "",
    mode: str = "generate",
    primary: str = "",
    primary_prompt: str = "",
    url: str = "",
    query: str = "",
    anchor: str = "",
    var_key: str = "",
    condition: str = "",
    condition_when: str = "",
    op: str = "",
    threshold: Optional[float] = None,
    dedup: bool = False,
    ping: bool = True,
) -> tuple[Optional[dict], str]:
    """(plan, erreur). plan None = rappel simple, sans pipeline (comportement historique)."""
    say = (instruction or "").strip()
    prompt_say = (primary_prompt or "").strip()
    primary_s = (primary or "").strip().lower()
    cond_text = _clip(condition, 200)
    when = (condition_when or "").strip().lower()
    url = (url or "").strip()
    query = _clip(query, 200)
    mode_s = _mode(mode)
    if primary_s and primary_s not in PRIMARY_TYPES:
        return None, "Action primaire inconnue (read_url, web_search, prompt, post)."
    if when and when not in ("before", "after"):
        return None, "condition_when : before (avant l'action) ou after (sur son résultat)."

    op_s = (op or "").strip().lower()
    relative = op_s in _REL_OPS
    has_price = threshold is not None or relative
    if not primary_s:
        if url:
            primary_s = "read_url"
        elif query:
            primary_s = "web_search"

    if has_price and not url and primary_s != "web_search" and not query:
        return None, (
            "Seuil de prix sans page. Passe url (page produit) "
            "ou query + primary=web_search."
        )
    if has_price and not primary_s:
        primary_s = "read_url" if url else "web_search"

    if not op_s:
        op_s = "lt"
    if op_s not in _OPS:
        op_s = "lt"

    cond: Optional[dict] = None
    if has_price:
        if relative and threshold is None:
            thr = 0.0
        else:
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                return None, "Seuil de prix invalide."
        # Le chiffre n'existe qu'une fois la page lue : la condition est toujours après.
        label = condition_label({
            "type": "price", "op": op_s, "threshold": thr,
        })
        cond = {
            "when": "after",
            "type": "price",
            "op": op_s,
            "threshold": thr,
            "text": cond_text or label,
            "dedup": bool(dedup),
        }
    elif cond_text:
        cond = {
            "when": when or "",
            "type": "judge",
            "text": cond_text,
            "dedup": False,
        }

    if cond and not cond["when"]:
        cond["when"] = "after" if primary_s in ("read_url", "web_search", "prompt") else "before"

    # Poster puis juger le post n'a pas de sens : le message est la suite, le filtre est avant.
    if cond and cond["when"] == "after" and primary_s == "post":
        primary_s = ""
        cond["when"] = "before"
    if cond and cond["when"] == "after" and not primary_s:
        return None, (
            "Condition sur le résultat, mais aucune action avant "
            "(read_url, web_search ou prompt)."
        )

    if primary_s == "read_url" and not url:
        return None, "read_url exige une url."
    if primary_s == "web_search" and not query:
        return None, "web_search exige query."

    # Rappel simple : le message part à chaque fois. Pas de plan stocké.
    if cond is None and primary_s in ("", "post", "prompt"):
        return None, ""

    primary_obj: Optional[dict] = None
    if primary_s == "read_url":
        primary_obj = {
            "type": "read_url",
            "url": url[:400],
            "anchor": (anchor or "").strip()[:40],
            "var_key": (var_key or "").strip()[:80],
        }
    elif primary_s == "web_search":
        primary_obj = {"type": "web_search", "query": query}
    elif primary_s == "prompt":
        primary_obj = {"type": "prompt", "say": prompt_say or say}
    elif primary_s == "post":
        primary_obj = {"type": "post", "say": say, "mode": mode_s, "ping": bool(ping)}

    relay = False
    secondary_say = say
    secondary_mode = mode_s
    if primary_s == "prompt" and cond is not None:
        if prompt_say:
            secondary_say = say
        else:
            # Une seule consigne : elle pilote le prompt, on publie sa réponse si la condition tient.
            relay = True
            secondary_say = ""
            secondary_mode = "verbatim"
    if primary_s == "prompt" and not (primary_obj or {}).get("say"):
        return None, "primary=prompt exige primary_prompt ou instruction."

    needs_secondary = cond is not None or primary_s in ("read_url", "web_search", "prompt")
    secondary: Optional[dict] = None
    if needs_secondary and primary_s != "post":
        secondary = {
            "type": "post",
            "say": secondary_say,
            "mode": secondary_mode,
            "ping": bool(ping),
            "relay": relay,
        }
    elif primary_s == "post" and cond is not None and cond["when"] == "before":
        secondary = {
            "type": "post",
            "say": say,
            "mode": mode_s,
            "ping": bool(ping),
            "relay": False,
        }
        primary_obj = None

    if primary_obj is None and secondary is None:
        return None, ""

    return {
        "primary": primary_obj,
        "condition": cond,
        "secondary": secondary,
    }, ""


def _synthesize_watch(trigger: dict, recipe: dict, instruction: str) -> dict:
    try:
        thr = float(trigger.get("threshold"))
    except (TypeError, ValueError):
        thr = 0.0
    op_s = str(trigger.get("op") or "lt")
    if op_s not in _OPS:
        op_s = "lt"
    mode = _mode(str(recipe.get("mode") or "generate"))
    say = (recipe.get("say") or instruction or "").strip()
    return {
        "primary": {
            "type": "read_url",
            "url": str(trigger.get("url") or "").strip()[:400],
            "anchor": str(trigger.get("anchor") or "").strip()[:40],
            "var_key": str(trigger.get("var_key") or "").strip()[:80],
        },
        "condition": {
            "when": "after",
            "type": "price",
            "op": op_s,
            "threshold": thr,
            "text": f"prix {_OP_SYMBOL[op_s]} {thr:g} €",
            "dedup": True,
        },
        "secondary": {
            "type": "post",
            "say": say,
            "mode": mode,
            "ping": bool(recipe.get("ping", True)),
            "relay": False,
        },
    }


def plan_of(*, kind: str, instruction: str, trigger: dict, recipe: dict) -> dict:
    """Plan stocké, veille historique, ou `{legacy: True}` (message à chaque déclenchement)."""
    stored = recipe.get("plan") if isinstance(recipe, dict) else None
    if isinstance(stored, dict) and (
        stored.get("primary") or stored.get("condition") or stored.get("secondary")
    ):
        return stored
    if kind == KIND_WATCH and isinstance(trigger, dict) and (trigger.get("url") or "").strip():
        return _synthesize_watch(trigger, recipe if isinstance(recipe, dict) else {}, instruction)
    return {"legacy": True}


def plan_of_task(task) -> dict:
    recipe = getattr(task, "recipe", None)
    trigger = getattr(task, "trigger", None)
    return plan_of(
        kind=getattr(task, "kind", "") or "",
        instruction=getattr(task, "instruction", "") or "",
        trigger=trigger if isinstance(trigger, dict) else {},
        recipe=recipe if isinstance(recipe, dict) else {},
    )


def describe_plan(plan: dict, *, when: str, price: Optional[float] = None) -> str:
    """Phrase naturelle du pipeline, pour la carte et le modèle."""
    if not isinstance(plan, dict) or plan.get("legacy"):
        return ""
    primary = plan.get("primary") or {}
    cond = plan.get("condition") if isinstance(plan.get("condition"), dict) else None
    bits: list[str] = []
    ptype = primary.get("type")
    if ptype == "read_url":
        bits.append("je lis la page")
    elif ptype == "web_search":
        bits.append(f"je cherche « {_clip(str(primary.get('query') or 'le sujet'), 60)} »")
    elif ptype == "prompt":
        bits.append("je prépare l'info")
    head = (when or "À ce moment").rstrip(".")
    if cond and cond.get("type") == "price":
        try:
            thr = f"{float(cond.get('threshold')):g}"
        except (TypeError, ValueError):
            thr = "?"
        op_name = str(cond.get("op") or "lt")
        now = f" (actuellement {price:.2f} €)" if price is not None else ""
        action = "je te ping" if bits else "je te parle"
        middle = (", ".join(bits) + ", ") if bits else ""
        if op_name in _REL_OPS:
            word = _OP_WORD[op_name]
            margin = _threshold_value(cond)
            extra = f" d'au moins {margin:g} €" if margin > 0 else ""
            return (
                f"{head}, {middle}{action} seulement si le prix est {word}{extra}{now}. "
                "Sinon je ne dis rien."
            )
        word = _OP_WORD.get(op_name, "sous")
        return f"{head}, {middle}{action} seulement si le prix est {word} {thr} €{now}. Sinon je ne dis rien."
    if cond:
        text = _clip(str(cond.get("text") or "la condition"), 80)
        place = "avant" if cond.get("when") == "before" else "après"
        middle = (", ".join(bits) + ", ") if bits else ""
        return (
            f"{head}, {middle}je ne parle que si « {text} » ({place} l'action). "
            "Sinon je ne dis rien."
        )
    if bits:
        return f"{head}, {' et '.join(bits)}, et je te dis le résultat."
    return f"{head}, je te dis le résultat."


def pseudo_lines(plan: dict, *, indent: str = "  ") -> list[str]:
    """Lignes de pseudo-code du pipeline (sans le QUAND, qui dépend du déclencheur)."""
    if not isinstance(plan, dict) or plan.get("legacy"):
        return []
    primary = plan.get("primary") or {}
    cond = plan.get("condition") if isinstance(plan.get("condition"), dict) else None
    secondary = plan.get("secondary") if isinstance(plan.get("secondary"), dict) else None
    when = (cond or {}).get("when") or "after"
    inner = indent + "  "

    def _primary(at: str) -> list[str]:
        ptype = primary.get("type")
        if ptype == "read_url":
            url = _clip(str(primary.get("url") or "URL"), 60)
            return [f"{at}LIRE {url}"]
        if ptype == "web_search":
            return [f'{at}CHERCHER "{_clip(str(primary.get("query") or ""), 60)}"']
        if ptype == "prompt":
            say = _clip(str(primary.get("say") or "la consigne"), 80)
            return [f'{at}ANALYSER "{say}"']
        if ptype == "post":
            return _post_lines(primary, at)
        return []

    def _post_lines(action: dict, at: str) -> list[str]:
        if action.get("relay"):
            return [f"{at}DIRE le résultat"]
        mode = action.get("mode") or "generate"
        say = _clip(str(action.get("say") or ""), 80)
        if mode == "verbatim" and say:
            return [f'{at}DIRE "{say}"']
        if action.get("ping", True):
            return [f"{at}PING moi"]
        if say:
            return [f'{at}DIRE "{say}"']
        return [f"{at}DIRE le résultat"]

    def _si(at: str) -> str:
        label = condition_label(cond)
        if cond and cond.get("type") == "price":
            return f"{at}SI {label}"
        return f'{at}SI "{label}"'

    lines: list[str] = []
    if cond and when == "before":
        lines.append(_si(indent))
        lines.extend(_primary(inner))
        if secondary:
            lines.extend(_post_lines(secondary, inner))
        lines.append(f"{indent}SINON silence")
        return lines

    lines.extend(_primary(indent))
    if cond:
        lines.append(_si(indent))
        if secondary:
            lines.extend(_post_lines(secondary, inner))
        elif primary.get("type") == "post":
            pass
        lines.append(f"{indent}SINON silence")
        return lines
    if secondary and primary.get("type") != "post":
        lines.extend(_post_lines(secondary, indent))
    return lines


def watch_every_label(trigger: dict) -> str:
    try:
        interval = int((trigger or {}).get("interval_minutes") or WATCH_INTERVAL_MIN_MINUTES)
    except (TypeError, ValueError):
        interval = WATCH_INTERVAL_MIN_MINUTES
    if interval >= 60 and interval % 60 == 0:
        return f"toutes les {interval // 60}h"
    return f"toutes les {interval} min"
