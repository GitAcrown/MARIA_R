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
    r"\b(?:fiche\s+)?layout\b"
    r"|\b(?:une?\s+)?(?:recettes?|recipes?)\b"
    r"|\b(?:tutoriels?|tuto(?:riel)?s?)\b"
    r"|\bcomparatif(?:s|s\s+dense)?\b"
    r"|comment (?:cuisiner|pr[ée]parer)\b",
    re.I,
)

_HINTS: tuple[tuple[str, str], ...] = (
    ("youtube", "- YouTube : tu ne peux pas regarder ni résumer la vidéo. Dis-le. Pas de summarize_channel pour ça."),
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


def _is_youtube_message(msg: discord.Message) -> bool:
    blob = msg.content or ""
    for emb in msg.embeds or []:
        blob += f" {emb.url or ''} {emb.title or ''}"
        blob += f" {getattr(emb.provider, 'name', None) or ''}"
        if emb.video and emb.video.url:
            blob += f" {emb.video.url}"
    if _YT_RE.search(blob) or "youtube" in blob.lower():
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
        text = getattr(msg, "clean_content", None) or msg.content or ""
        if _LAYOUT_RE.search(text):
            flags.add("layout")
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
