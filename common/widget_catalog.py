"""Catalogue de blocs pour les widgets composés librement par l'IA (`render_widget`).

Les widgets « canon » (météo, TMDB, Steam, foot, rappels) gardent leur mise en page
figée côté code — ce catalogue ne sert qu'aux widgets ad hoc, pour les cas sans
outil dédié (comparatif, classement, carte de membre…). Catalogue volontairement
fermé (pas de HTML/CSS libre) pour rester dans le cadre sûr de Discord Components v2.
"""

from __future__ import annotations

from typing import Optional

import discord

from common.discord_ui import section_with_thumbnail

_MAX_BLOCKS = 12
_MAX_STAT_ITEMS = 6
_MAX_GALLERY = 4
_MAX_TEXT = 800


def _text_block(content: str) -> discord.ui.TextDisplay:
    return discord.ui.TextDisplay(str(content)[:_MAX_TEXT])


def _stat_row_block(items) -> Optional[discord.ui.TextDisplay]:
    parts: list[str] = []
    for item in (items or [])[:_MAX_STAT_ITEMS]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        value = str(item.get("value") or "").strip()
        if not label and not value:
            continue
        parts.append(f"**{label}** {value}".strip() if label else value)
    if not parts:
        return None
    return discord.ui.TextDisplay("  ·  ".join(parts))


def render_free_widget(spec: Optional[dict], commentary: str = "") -> Optional[discord.ui.LayoutView]:
    """Construit un `LayoutView` à partir d'un spec libre (catalogue restreint).

    `spec` = {"title": str|None, "emoji": str|None, "blocks": [bloc, ...]}
    Types de bloc : text, separator, stat_row, thumbnail, gallery, footer.
    Renvoie None si le spec est vide/invalide (fallback texte côté appelant).
    """
    if not isinstance(spec, dict):
        return None
    blocks = spec.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None

    title = (spec.get("title") or "").strip()
    emoji = (spec.get("emoji") or "").strip()

    children: list = []
    if title:
        head = f"## {emoji} {title}".strip() if emoji else f"## {title}"
        children.append(_text_block(head))
        children.append(discord.ui.Separator())

    gallery_urls: list[str] = []
    for raw in blocks[:_MAX_BLOCKS]:
        if not isinstance(raw, dict):
            continue
        btype = (raw.get("type") or "").strip()
        if btype == "text":
            content = (raw.get("content") or "").strip()
            if content:
                children.append(_text_block(content))
        elif btype == "separator":
            children.append(discord.ui.Separator())
        elif btype == "stat_row":
            block = _stat_row_block(raw.get("items"))
            if block:
                children.append(block)
        elif btype == "thumbnail":
            url = (raw.get("url") or "").strip()
            text = (raw.get("text") or "").strip()
            if url and text:
                children.append(section_with_thumbnail(_text_block(text), url))
            elif text:
                children.append(_text_block(text))
        elif btype == "gallery":
            urls = [
                u for u in (raw.get("urls") or [])
                if isinstance(u, str) and u.startswith(("http://", "https://"))
            ]
            if urls:
                gallery_urls = urls[:_MAX_GALLERY]
        elif btype == "footer":
            text = (raw.get("text") or "").strip()
            if text:
                children.append(discord.ui.Separator())
                children.append(discord.ui.TextDisplay(f"-# {text}"))

    if not children and not gallery_urls:
        return None

    view = discord.ui.LayoutView(timeout=None)
    if commentary:
        view.add_item(discord.ui.TextDisplay(commentary))
        view.add_item(discord.ui.Separator())
    if children:
        view.add_item(discord.ui.Container(*children))
    if gallery_urls:
        gallery = discord.ui.MediaGallery()
        added = 0
        for url in gallery_urls:
            try:
                gallery.add_item(media=url)
                added += 1
            except Exception:
                continue
        if added:
            view.add_item(gallery)

    return view if view.children else None


# ---------------------------------------------------------------------------
# Schéma JSON (OpenAI strict function calling) — exposé aux outils LLM.
# ---------------------------------------------------------------------------

_BLOCK_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["text", "separator", "stat_row", "thumbnail", "gallery", "footer"],
            "description": "Nature du bloc.",
        },
        "content": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Texte libre (markdown Discord limité). Uniquement pour type=text.",
        },
        "items": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                },
                {"type": "null"},
            ],
            "description": "Paires label/valeur (max 6). Uniquement pour type=stat_row.",
        },
        "url": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "URL d'image. Uniquement pour type=thumbnail.",
        },
        "text": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Texte associé : à côté de la vignette (thumbnail) ou en petit (footer).",
        },
        "urls": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ],
            "description": "URLs d'images (max 4). Uniquement pour type=gallery.",
        },
    },
    "required": ["type", "content", "items", "url", "text", "urls"],
    "additionalProperties": False,
}

WIDGET_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Titre affiché en tête du widget (optionnel).",
        },
        "emoji": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Un seul emoji affiché avant le titre (optionnel).",
        },
        "blocks": {
            "type": "array",
            "description": "Contenu du widget, dans l'ordre d'affichage. Max 12 blocs.",
            "items": _BLOCK_ITEM_SCHEMA,
        },
    },
    "required": ["title", "emoji", "blocks"],
    "additionalProperties": False,
}

# Exemples canon (few-shot) — à réutiliser tels quels dans la description des outils
# qui exposent ce catalogue, pour garder un style cohérent malgré la composition libre.
WIDGET_CANON_EXAMPLES = """Exemples canon (respecte ce style : titres courts, footer sourcé) :

1) Classement / stats :
{"title": "Classement", "emoji": "🏆", "blocks": [
  {"type": "stat_row", "items": [{"label": "1er", "value": "Alice — 42 pts"}, {"label": "2e", "value": "Bob — 35 pts"}]},
  {"type": "footer", "text": "Basé sur les scores du salon"}
]}

2) Comparatif A vs B :
{"title": "PS5 vs Xbox Series X", "emoji": "🎮", "blocks": [
  {"type": "text", "content": "**PS5** : exclusivités fortes, DualSense."},
  {"type": "separator"},
  {"type": "text", "content": "**Xbox Series X** : Game Pass, rétrocompatibilité."},
  {"type": "footer", "text": "Avis MARIA"}
]}

3) Fiche personne avec vignette :
{"title": "Carte de membre", "emoji": "🪪", "blocks": [
  {"type": "thumbnail", "url": "https://exemple.com/avatar.png", "text": "Membre actif, aime les RPG."},
  {"type": "footer", "text": "D'après la mémoire de MARIA"}
]}

4) Recette / tutoriel :
{"title": "Pâtes carbonara", "emoji": "🍝", "blocks": [
  {"type": "stat_row", "items": [{"label": "Temps", "value": "25 min"}, {"label": "Pers.", "value": "2"}]},
  {"type": "text", "content": "**Ingrédients**\\n• 200 g spaghetti\\n• 100 g guanciale\\n• 2 jaunes\\n• 40 g pecorino"},
  {"type": "separator"},
  {"type": "text", "content": "**Étapes**\\n1. Dorer le guanciale.\\n2. Cuire les pâtes.\\n3. Lier hors du feu avec jaunes + pecorino."},
  {"type": "footer", "text": "Recette classique"}
]}"""
