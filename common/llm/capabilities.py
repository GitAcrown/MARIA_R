"""Consignes temporaires selon ce que contient la demande (lien, image, fichier…).

Ajoutées au developer prompt pour CE tour seulement — le prompt de base
reste mince. Catalogue fermé : on n'ajoute une ligne que si le trigger
ou le message cité matche vraiment.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import discord

from .attachments import _is_audio, _is_image, _is_text_file

_URL_RE = re.compile(r"https?://[^\s<>\]\)]+", re.I)
_YT_RE = re.compile(
    r"(?:https?://)?(?:(?:www|m)\.)?(?:youtube\.com/(?:watch\?|shorts/|live/|embed/)|youtu\.be/)",
    re.I,
)
_YT_HOSTS = frozenset({
    "youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com",
})
_SKIP_HOSTS = frozenset({
    "cdn.discordapp.com", "media.discordapp.net",
    "discord.com", "discordapp.com", "discord.gg",
})
_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_VID_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi")
_LAYOUT_RE = re.compile(
    r"\b(?:fiches?|layouts?)\b"
    r"|\b(?:recettes?|recipes?|ingr[ée]dients?)\b"
    r"|\b(?:tutoriels?|tuto(?:riel)?s?)\b"
    r"|\bcomparatif"
    r"|comment (?:cuisiner|pr[ée]parer|faire)\b"
    r"|[ée]tape par [ée]tape|mode d['']emploi",
    re.I,
)
_TRANSPORT_RE = re.compile(
    r"\b(?:m[eé]tro|rer|tram|bus|transilien|sncf|train|trains?|"
    r"gare|trajet|itin[eé]raire|correspondance|ratp|"
    r"ligne|arr[êe]t|station|quai|trafic|retard)\b"
    r"|comment (?:on |je |tu |nous |vous |y )?(?:va|vais|allez|aller)\b"
    r"|(?:pour|faut|dois|doit|on doit|je dois) aller\b"
    r"|(?:on va|je vais|tu vas|on y va) (?:à|au|aux|en)\b"
    r"|depuis chez\b"
    r"|pour (?:me |te |se )?rendre\b"
    r"|chemin (?:pour|vers|jusqu)\b",
    re.I,
)
_FOOTBALL_RE = re.compile(
    r"\b(?:foot(?:ball)?|match(?:s)?|score|scores?|ligue|champions?|"
    r"psg|\bom\b|\bol\b|asse|équipe|equipe|buteurs?|buts?|"
    r"hors[- ]jeu|classement|c1|c3|l1|ligue 1|"
    r"coupe du monde|euro|penalty|p[ée]nalty|mi[- ]temps|"
    r"r[ée]sultat|r[ée]sultats|championnat)\b",
    re.I,
)
_SUMMARY_RE = re.compile(
    r"\b(?:r[eé]sum[eé]|r[eé]cap(?:itul(?:e|er|atif)?)?|"
    r"synth[eè]se|r[eé]capitule|"
    r"derniers messages|c'[eé]tait quoi|"
    r"ce qui s['']est dit|t'as (?:suivi|lu)|catch[- ]up)\b",
    re.I,
)
_IMAGES_SEARCH_RE = re.compile(
    r"\b(?:montre(?:[- ]moi)?|images?|photos?|illustration|"
    r"visuel|pics?|[àa] quoi (?:[çc]a )?ressemble)\b",
    re.I,
)
_WEB_PAGE_RE = re.compile(
    r"\b(?:articles?|liens?|urls?|ce site|cette page|page web|"
    r"lis (?:le |la |cet |cette )?(?:page|article|lien|site))\b",
    re.I,
)
_YT_WORD_RE = re.compile(r"\b(?:youtube|youtu\.be|\byt\b|shorts)\b", re.I)

# Outils envoyés seulement si le flag correspondant est présent.
# Tout le reste (météo, search_web, mémoire, etc.) reste toujours exposé.
_GATED_TOOLS: dict[str, str] = {
    "read_youtube": "youtube",
    "read_web_page": "web",
    "search_images": "images_search",
    "get_transport": "transport",
    "get_football": "football",
    "summarize_channel": "summary",
    "render_widget": "layout",
}

_HINTS: tuple[tuple[str, str], ...] = (
    ("youtube", "- YouTube : read_youtube (sous-titres auteur ou auto). Pas l'image ni le son. Pas de summarize_channel. Si l'outil échoue / pas de sous-titres → dis-le, n'invente pas."),
    ("video_file", "- Vidéo jointe : tu ne peux pas la lire."),
    ("image", "- Image : tu la vois — prends-la en compte."),
    ("web", "- Lien web : read_web_page avant d'en parler. N'invente pas le contenu."),
    ("audio", "- Audio : base-toi sur la transcription fournie, ne prétends pas l'avoir écouté."),
    ("text_file", "- Fichier texte : le contenu est déjà dans le message."),
    ("file", "- Fichier : tu ne l'ouvres pas (nom seulement, pas le contenu)."),
    ("layout", "- Layout : seulement recette complète, tuto multi-étapes, comparatif dense, ou demande explicite de fiche/layout → render_widget. Question directe, avis, définition, « comment je fais » en deux phrases → tchat, pas de widget."),
)


def _host(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _path_ext(url: str) -> str:
    try:
        path = urlparse(url).path or ""
    except ValueError:
        return ""
    dot = path.rfind(".")
    return path[dot:].lower() if dot >= 0 else ""


def _urls_in(msg: discord.Message) -> list[str]:
    found = _URL_RE.findall(msg.content or "")
    for emb in msg.embeds or []:
        if emb.url:
            found.append(emb.url)
        if emb.video and emb.video.url:
            found.append(emb.video.url)
    return found


def _message_text(msg: discord.Message) -> str:
    """Texte + titres d'embeds — pour matcher large, pas seulement le clean_content."""
    parts = [getattr(msg, "clean_content", None) or msg.content or ""]
    for emb in msg.embeds or []:
        if emb.title:
            parts.append(emb.title)
        if emb.description:
            parts.append(emb.description[:240])
    return "\n".join(parts)


def _is_youtube_message(msg: discord.Message) -> bool:
    blob = msg.content or ""
    for emb in msg.embeds or []:
        blob += f" {emb.url or ''} {emb.title or ''}"
        blob += f" {getattr(emb.provider, 'name', None) or ''}"
        if emb.video and emb.video.url:
            blob += f" {emb.video.url}"
    if _YT_RE.search(blob) or _YT_WORD_RE.search(blob):
        return True
    return any(_host(u) in _YT_HOSTS for u in _urls_in(msg))


def _has_real_image(msg: discord.Message) -> bool:
    """Image vraiment jointe / collée — pas la miniature d'un embed YouTube."""
    for att in msg.attachments or []:
        if _is_image(att):
            return True
    if msg.stickers:
        return True
    for m in _URL_RE.finditer(msg.content or ""):
        if _path_ext(m.group(0)) in _IMG_EXT:
            return True
    for emb in msg.embeds or []:
        provider = (getattr(emb.provider, "name", None) or "").lower()
        if "youtube" in provider:
            continue
        if emb.image and emb.image.url:
            return True
    return False


def _is_video_att(att: discord.Attachment) -> bool:
    ct = att.content_type or ""
    fn = (att.filename or "").lower()
    return ct.startswith("video/") or fn.endswith(_VID_EXT)


def collect_capability_flags(*messages: discord.Message | None) -> set[str]:
    flags: set[str] = set()
    for msg in messages:
        if msg is None:
            continue
        if _is_youtube_message(msg):
            flags.add("youtube")
        if _has_real_image(msg):
            flags.add("image")
        text = _message_text(msg)
        if _LAYOUT_RE.search(text):
            flags.add("layout")
        if _TRANSPORT_RE.search(text):
            flags.add("transport")
        if _FOOTBALL_RE.search(text):
            flags.add("football")
        if _SUMMARY_RE.search(text):
            flags.add("summary")
        if _IMAGES_SEARCH_RE.search(text):
            flags.add("images_search")
        if _WEB_PAGE_RE.search(text):
            flags.add("web")
        for url in _urls_in(msg):
            host = _host(url)
            ext = _path_ext(url)
            if host in _YT_HOSTS or ext in _IMG_EXT or host in _SKIP_HOSTS:
                continue
            if ext in _VID_EXT:
                flags.add("video_file")
                continue
            flags.add("web")
        for att in msg.attachments or []:
            if _is_image(att):
                continue
            if _is_audio(att):
                flags.add("audio")
            elif _is_text_file(att):
                flags.add("text_file")
            elif _is_video_att(att):
                flags.add("video_file")
            else:
                flags.add("file")
    return flags


def build_capability_ctx(*messages: discord.Message | None) -> str:
    flags = collect_capability_flags(*messages)
    if not flags:
        return ""
    lines = [text for key, text in _HINTS if key in flags]
    if not lines:
        return ""
    return "\nDEMANDE (ce tour seulement) :\n" + "\n".join(lines) + "\n"


def select_tool_names(all_names: list[str], flags: set[str]) -> list[str]:
    """Garde tous les outils sauf une poignée de spécialisés hors-sujet.

    Un outil inconnu / non listé reste inclus (sûr pour les ajouts futurs).
    """
    return [
        name for name in all_names
        if _GATED_TOOLS.get(name) is None or _GATED_TOOLS[name] in flags
    ]
