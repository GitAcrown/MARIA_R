"""Veille URL / prix — parse heuristique (+ arbitrage JEV si ambigu) + comparaison seuil."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("MARIA.TaskWatch")

_PRICE_RE = re.compile(
    r"(?:€\s*)(\d{1,3}(?:[ \u202f\u00a0.]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"|(\d{1,3}(?:[ \u202f\u00a0.]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(?:€|eur\b|euros?\b)",
    re.IGNORECASE,
)

# Contexte qui fait préférer / écarter un montant.
_GOOD_CTX = (
    "prix", "price", "à partir de", "a partir de", "seulement", "now", "maintenant",
    "offre", "promo", "ajouter au panier", "add to cart", "acheter", "buy",
)
_BAD_CTX = (
    "livraison", "frais", "port", "shipping", "expédition", "expedition",
    "économisez", "economisez", "économie", "remise de", "save ", "−", "mensualité",
    "par mois", "/mois", "x fois", "paiement en", "garantie", "abonnement",
)


def _to_float(raw: str) -> Optional[float]:
    s = (raw or "").strip().replace("\u202f", "").replace("\u00a0", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        # 1.299,90 → 1299.90 ; 1,299.90 → 1299.90
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    elif s.count(".") > 1 or (s.count(".") == 1 and len(s.split(".")[-1]) == 3):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def price_candidates(text: str, anchor: str = "") -> list[dict]:
    """Montants en € classés (meilleur d'abord) : valeur, score, extrait, ancre.

    Score = +2 contexte favorable, -3 défavorable, +5 si l'ancre mémorisée à la
    création précède le montant ; à score égal, la 1re occurrence gagne.
    """
    text = text or ""
    anchor = (anchor or "").strip().casefold()
    out: list[dict] = []
    for idx, m in enumerate(_PRICE_RE.finditer(text)):
        val = _to_float(m.group(1) or m.group(2) or "")
        if val is None or val <= 0 or val > 1_000_000:
            continue
        ctx = text[max(0, m.start() - 24):m.end() + 18].casefold()
        score = 0
        if any(k in ctx for k in _GOOD_CTX):
            score += 2
        if any(k in ctx for k in _BAD_CTX):
            score -= 3
        if val < 1:
            score -= 2
        before = text[max(0, m.start() - 40):m.start()].casefold()
        if anchor and anchor in " ".join(before.split()):
            score += 5
        out.append({
            "value": val,
            "score": score,
            "pos": idx,
            "snippet": " ".join(text[max(0, m.start() - 40):m.end() + 18].split()),
            # Mots juste avant le montant, sans le montant voisin (« … 4,99 € Prix » → « Prix »).
            "anchor": " ".join(
                re.split(r"[€\d]", text[max(0, m.start() - 18):m.start()])[-1].split()
            ),
        })
    out.sort(key=lambda c: (-c["score"], c["pos"]))
    return out


def extract_price_eur(text: str, anchor: str = "") -> Optional[float]:
    """Prix principal plausible d'une page (heuristique pure, sans IA)."""
    cands = price_candidates(text, anchor)
    return cands[0]["value"] if cands else None


async def pick_price(text: str, typesafe=None) -> tuple[Optional[float], str]:
    """(prix, ancre). Heuristique d'abord ; JEV arbitre seulement si c'est ambigu."""
    cands = price_candidates(text)
    if not cands:
        return None, ""
    best = cands[0]
    distinct: list[dict] = []
    for c in cands:
        if all(abs(c["value"] - d["value"]) > 0.001 for d in distinct):
            distinct.append(c)
    ambiguous = len(distinct) > 1 and distinct[0]["score"] - distinct[1]["score"] < 2
    if ambiguous and typesafe is not None:
        idx = await typesafe.pick_main_price([c["snippet"] for c in distinct[:4]])
        if idx is not None and idx < len(distinct):
            best = distinct[idx]
    anchor = best["anchor"] if len(re.sub(r"\W", "", best["anchor"])) >= 4 else ""
    return best["value"], anchor


def condition_met(
    *,
    price: Optional[float],
    op: str,
    threshold: float,
    previous: Optional[float],
) -> bool:
    if price is None:
        return False
    if op == "lt":
        return price < threshold
    if op == "lte":
        return price <= threshold
    if op == "gt":
        return price > threshold
    if op == "change":
        return previous is not None and abs(price - previous) >= threshold
    return price < threshold
