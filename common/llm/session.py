"""Session par salon — contexte complet, lock, tools."""

import asyncio
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import discord

from common.timezones import PARIS_TZ as _PARIS_TZ
from common.widgets import has_widget

from .client import MariaLLMClient, MariaOpenAIError
from .context import (
    ConversationContext,
    MessageRecord,
    AssistantRecord,
    ToolCallRecord,
    ToolResponseRecord,
    TextComponent,
    ImageComponent,
    MetadataComponent,
)
from .tools import ToolRegistry
from .attachments import AttachmentCache, process_attachment
from .capabilities import (
    GROUNDING_TOOL_NAMES,
    build_capability_ctx,
    collect_capability_flags,
    momentum_flags,
    select_tool_names,
    should_force_tool,
)

logger = logging.getLogger("llm.session")

# Champ OpenAI `messages[].name` : pattern ^[^\s<|\\/>]+$ (pas d'espaces ni |<>\/).
# Le pseudo+id lisible va dans le *contenu* du message, pas dans ce champ.
USER_FORMAT = "{message.author.name}"
MAX_RECURSION = 8

# Fenêtre de "momentum" pour le gating d'outils : un outil de recherche appelé
# récemment garde son flag ouvert quelques tours (question de suivi sur un titre
# propre sans mot-clé, deuxième recherche web qui enchaîne sur la première…).
_MOMENTUM_MESSAGES = 20
_MOMENTUM_WINDOW = timedelta(minutes=10)

# Fuites de tokens / placeholders connus côté modèle — filet de sécurité, pas un
# vrai fix côté modèle. VEVENT = jargon iCal d'outils internes OpenAI ;
# `[](widget)` = le modèle invente parfois un lien markdown pour « pointer »
# vers le LayoutView Discord qu'il vient d'appeler.
_LEAKED_TOKEN_RE = re.compile(
    r"\s*BEGIN:VEVENT.*?END:VEVENT\s*"
    r"|\s*:?\bVEVENT\b\s*"
    r"|\s*!?\[[^\]]*\]\(\s*widget\s*\)\s*"
    r"|\s*\[widget\]\s*",
    re.IGNORECASE | re.DOTALL,
)
# Scripts « exotiques » parfois collés en fin de réponse par gpt-5.6-luna
# (cyrillique / CJK / hébreu / arabe…). On ne touche QUE un court blob final
# isolé après une phrase majoritairement latine — une vraie réponse en russe
# (corps déjà cyrillique) n'est pas concernée.
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F"  # cyrillique
    r"\u0590-\u05FF"  # hébreu
    r"\u0600-\u06FF\u0750-\u077F"  # arabe
    r"\u0900-\u097F"  # dévanâgarî
    r"\u3040-\u30FF\u3400-\u9FFF\uF900-\uFAFF"  # CJK / kana
    r"\uAC00-\uD7AF]+"  # hangul
)
_TRAILING_FOREIGN_JUNK_RE = re.compile(
    r"(?P<body>.*[.!?…])"  # phrase déjà terminée
    r"(?P<close>[\"'»”)\]*_`]*)"  # fermetures markdown / guillemets
    r"(?P<trail>\s+" + _FOREIGN_SCRIPT_RE.pattern + r")\s*$",
    re.DOTALL,
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]")


def _strip_trailing_foreign_junk(text: str) -> str:
    """Retire un mini-blob de script non-latin collé APRÈS une phrase latine.

    Garde intact :
    - une réponse majoritairement dans ce script (ex. vrai russe) ;
    - un mot étranger intégré avant la fin de phrase (« ça se dit привет. »).
    """
    m = _TRAILING_FOREIGN_JUNK_RE.match(text)
    if not m:
        return text
    body = m.group("body") + m.group("close")
    trail = m.group("trail")
    # Blob trop long → probablement du vrai contenu, pas une fuite.
    foreign_chars = _FOREIGN_SCRIPT_RE.findall(trail)
    trail_len = sum(len(x) for x in foreign_chars)
    if trail_len < 2 or trail_len > 24:
        return text
    latin = len(_LATIN_LETTER_RE.findall(body))
    foreign_in_body = sum(len(x) for x in _FOREIGN_SCRIPT_RE.findall(body))
    if latin < 8:
        return text
    # Corps déjà bilingue / non-latin → on ne touche pas.
    if foreign_in_body > 0 and foreign_in_body >= max(3, latin // 10):
        return text
    return body.rstrip()


def _strip_leaked_tokens(text: str) -> str:
    # Remplacer par un espace (pas une chaîne vide) pour ne pas coller les mots
    # entourant le fragment retiré ; on nettoie ensuite les espaces doublés.
    cleaned = _LEAKED_TOKEN_RE.sub(" ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return _strip_trailing_foreign_junk(cleaned)


INGESTED_IDS_MAX = 500
FOCUS_SNIPPET = 280
FOCUS_CONTENT = 240
BOT_REPLY_LAYOUT_CAP = 800
ARTIFACT_CAP = 800
HINT_PART_CAP = 220
HINT_MAX_PARTS = 3
SYSTEM_NOTE_HISTORY_CAP = 400

_VOICE_FLAG = 1 << 13


@dataclass
class WorkingArtifacts:
    """Contexte de travail du salon : résumé, widget, vocal, onglet — hors RAG."""

    summary: str = ""
    widget: str = ""
    transcript: str = ""
    tab: str = ""

    def record(self, kind: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if len(cleaned) > ARTIFACT_CAP:
            cleaned = cleaned[:ARTIFACT_CAP].rstrip() + "…"
        key = (kind or "").strip().lower()
        if key == "summary":
            self.summary = cleaned
        elif key == "transcript":
            self.transcript = cleaned
        elif key == "tab":
            self.tab = cleaned[:400]
        else:
            self.widget = cleaned

    def infer(self, note: str) -> None:
        raw = (note or "").strip()
        if not raw:
            return
        low = raw.lower()
        if "vocal transcrit" in low or low.startswith("[vocal"):
            self.record("transcript", raw)
        elif "résumé" in low or "widget résumé" in low:
            self.record("summary", raw)
        elif "onglet" in low:
            self.record("tab", raw)
        else:
            self.record("widget", raw)

    def hint_parts(self) -> list[str]:
        parts: list[str] = []
        if self.summary:
            parts.append(self.summary[:400])
        if self.widget:
            parts.append(self.widget[:400])
        if self.transcript:
            parts.append(self.transcript[:400])
        if self.tab:
            parts.append(self.tab[:200])
        return parts

    def clear(self) -> None:
        self.summary = self.widget = self.transcript = self.tab = ""


def _is_voice_message(message: discord.Message) -> bool:
    flags = getattr(message, "flags", None)
    if flags is None:
        return False
    return bool(getattr(flags, "value", 0) & _VOICE_FLAG)

_API_NAME_BAD_RE = re.compile(r"[\s<|\\/>]+")
# Aliases connus + le vrai nick Discord. Sert à retirer le ping en tête
# (« Maria passage… ») pour que luna ne prenne pas ça pour une étiquette de tour.
_BOT_NAME_ALIASES = ("Maria", "MARIA", "marie")
_IDENTITY_PUNCT_RE = re.compile(r"^[\s\"'«»*_`]+|[\s\"'«»*_`.,!?:;…\-—]+$")


def _api_message_name(message: discord.Message) -> str:
    """Identifiant stable et valide pour messages[].name (API OpenAI)."""
    raw = (message.author.name or "user").strip() or "user"
    safe = _API_NAME_BAD_RE.sub("_", raw).strip("_") or "user"
    return f"{safe}_{message.author.id}"


def _bot_identity(message: Optional[discord.Message]) -> tuple[Optional[int], list[str]]:
    """Id Discord du bot + noms à retirer en tête d'un message adressé."""
    names = list(_BOT_NAME_ALIASES)
    if message is None:
        return None, names
    me = getattr(getattr(message, "guild", None), "me", None)
    if me is None:
        return None, names
    for attr in ("name", "display_name", "global_name"):
        val = getattr(me, attr, None)
        if isinstance(val, str) and val.strip():
            names.append(val.strip())
    # Dédup en gardant les plus longs d'abord (évite de rater « MARIA »).
    uniq = sorted({n for n in names if n}, key=len, reverse=True)
    return me.id, uniq


def _strip_bot_address(text: str, *, bot_id: Optional[int], names: list[str]) -> str:
    """Retire un ping / nom de bot en tête (« @MARIA … » / « Maria, … »).

    Si le message n'est QUE le nom (« Maria ? »), on le garde : le vider ferait
    croire à un message vide.
    """
    out = (text or "").strip()
    if not out:
        return out
    original = out
    if bot_id is not None:
        out = re.sub(rf"^<@!?{bot_id}>\s*", "", out)
    for n in names:
        nxt = re.sub(
            rf"^{re.escape(n)}\b[\s,.:;!?\u2026\-—]*",
            "",
            out,
            count=1,
            flags=re.IGNORECASE,
        )
        if nxt != out:
            out = nxt
            break
    out = out.lstrip()
    return out if out else original


def _is_identity_stub(text: str, names: list[str]) -> bool:
    """True si la complétion n'est que le nom du bot (fuite d'étiquette de tour)."""
    t = _IDENTITY_PUNCT_RE.sub("", (text or "").strip())
    if not t or len(t) > 24:
        return False
    folded = t.casefold()
    return any(folded == n.casefold() for n in names)


def _payload_plain_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return ""


def _display_user_label(message: discord.Message) -> str:
    """Pseudo + id Discord pour le texte vu par le modèle (profils / attribution)."""
    return f"{message.author.name} ({message.author.id})"


async def resolve_message_reference(message: discord.Message) -> Optional[discord.Message]:
    """Résout le message cité (reply), avec fetch API en secours.

    `message.reference.resolved` n'est peuplé par le gateway que si Discord l'a inclus
    dans le payload ; sinon il vaut None ou DeletedReferencedMessage, et le contenu cité
    disparaît silencieusement du contexte sans ce fallback.
    """
    ref = message.reference
    if ref is None:
        return None
    resolved = ref.resolved
    if isinstance(resolved, discord.Message):
        return resolved
    if not ref.message_id:
        return None
    try:
        return await message.channel.fetch_message(ref.message_id)
    except (discord.NotFound, discord.HTTPException, discord.Forbidden):
        return None


def _components_v2_to_parts(
    components: list,
    *,
    _depth: int = 0,
) -> tuple[list[str], list[str]]:
    """Walk a components-v2 tree recursively.
    Returns (text_parts, image_urls).
    Stops at depth 6 to avoid runaway recursion.
    """
    if _depth > 6:
        return [], []

    texts: list[str] = []
    images: list[str] = []

    for comp in components:
        name = type(comp).__name__

        if name == "TextDisplay":
            content = getattr(comp, "content", None) or getattr(comp, "value", None)
            if content:
                texts.append(str(content))

        elif name in ("Container", "Section", "ActionRow"):
            children = (
                getattr(comp, "children", None)
                or getattr(comp, "components", None)
                or []
            )
            sub_texts, sub_imgs = _components_v2_to_parts(children, _depth=_depth + 1)
            texts.extend(sub_texts)
            images.extend(sub_imgs)
            accessory = getattr(comp, "accessory", None)
            if accessory:
                acc_texts, acc_imgs = _components_v2_to_parts([accessory], _depth=_depth + 1)
                texts.extend(acc_texts)
                images.extend(acc_imgs)

        elif name == "MediaGallery":
            for item in getattr(comp, "items", []):
                media = getattr(item, "media", None)
                url = getattr(media, "url", None) if media else None
                if url:
                    images.append(url)

        elif name in ("Thumbnail", "UnfurledMediaItem"):
            media = getattr(comp, "media", None)
            url = getattr(media, "url", None) if media else getattr(comp, "url", None)
            if url:
                images.append(url)

    return texts, images


def _reply_cite_line(label: str, preview: Optional[str] = None) -> str:
    """Barre Discord reply : citation d'un autre message, jamais le texte du membre."""
    if preview:
        return f'[Cité (reply, pas le texte de ce membre) — {label} : "{preview}"]'
    return f"[Cité (reply, pas le texte de ce membre) — {label}]"


def _cite_snippet(msg: discord.Message, limit: int = FOCUS_SNIPPET) -> str:
    """Aperçu du message cité, pour le [FOCUS] (texte, sinon titre/URL d'embed)."""
    text = (getattr(msg, "clean_content", None) or msg.content or "").strip()
    if text:
        return text[:limit]
    comps = getattr(msg, "components", None)
    if comps:
        comp_texts, _ = _components_v2_to_parts(list(comps))
        if comp_texts:
            return "\n".join(comp_texts)[:limit]
    for emb in getattr(msg, "embeds", None) or []:
        title = (emb.title or "").strip()
        url = (emb.url or "").strip()
        if title and url:
            return f"{title} ({url})"[:limit]
        if title or url:
            return (title or url)[:limit]
        if emb.video and emb.video.url:
            return str(emb.video.url)[:limit]
    return ""


def _embed_to_text(emb: discord.Embed) -> str:
    """Convertit un embed Discord en texte lisible pour le contexte."""
    lines: list[str] = []
    if emb.author and emb.author.name:
        lines.append(f"[{emb.author.name}]")
    if emb.title:
        title = emb.title
        if emb.url:
            title += f" ({emb.url})"
        lines.append(title)
    if emb.description:
        lines.append(emb.description[:500] + ("…" if len(emb.description) > 500 else ""))
    for field in emb.fields[:6]:
        if field.name and field.value:
            val = str(field.value)
            lines.append(f"{field.name}: {val[:200] + ('…' if len(val) > 200 else '')}")
    if emb.footer and emb.footer.text:
        lines.append(f"({emb.footer.text[:120]})")
    return "\n".join(lines)


class ChannelSession:
    """Session par salon — tous les messages vont dans le contexte GPT principal."""

    def __init__(
        self,
        channel_id: int,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        attachment_cache: AttachmentCache,
        developer_prompt_template: Callable[..., str],
        context_window: int = 12000,
        context_age_hours: float = 2,
        max_messages: int = 0,
    ):
        self.channel_id = channel_id
        self.client = client
        self.tool_registry = tool_registry
        self.attachment_cache = attachment_cache
        self.developer_prompt_template = developer_prompt_template
        self.context = ConversationContext(
            developer_prompt="",
            context_window=context_window,
            context_age=timedelta(hours=context_age_hours),
            max_messages=max_messages,
        )
        self._lock = asyncio.Lock()
        self.trigger_message: Optional[discord.Message] = None
        self._prompt_context: Optional[dict] = None
        # IDs Discord des messages déjà ingérés dans cette session (évite doublons de référence).
        # Borné : `_ingested_order` donne l'ordre d'éviction, `_ingested_ids` le test d'appartenance O(1).
        # `_ingested_records` garde une référence au MessageRecord produit, pour vérifier qu'il
        # est ENCORE dans le contexte courant (trim() peut l'avoir évincé entre-temps) avant de
        # se contenter d'un renvoi court type « [Suite de : X] » sans contenu.
        self._ingested_ids: set[int] = set()
        self._ingested_order: deque[int] = deque(maxlen=INGESTED_IDS_MAX)
        self._ingested_records: dict[int, MessageRecord] = {}
        # Notes système injectées récemment (résultats d'outils, widgets affichés…).
        # Surfacées dans le [FOCUS] pour que le LLM sache immédiatement le contexte actif.
        self._recent_system_notes: deque[tuple[datetime, str]] = deque(maxlen=6)
        self.artifacts = WorkingArtifacts()

    def _remember_ingested(self, message_id: int, record: MessageRecord) -> None:
        """Mémorise un ID ingéré (+ son record) en évinçant le plus ancien au-delà de la borne."""
        if message_id in self._ingested_ids:
            self._ingested_records[message_id] = record
            return
        if len(self._ingested_order) >= INGESTED_IDS_MAX:
            oldest = self._ingested_order.popleft()
            self._ingested_ids.discard(oldest)
            self._ingested_records.pop(oldest, None)
        self._ingested_order.append(message_id)
        self._ingested_ids.add(message_id)
        self._ingested_records[message_id] = record

    def prepare_edit_redo(self, user_message_id: int) -> bool:
        """Retire du contexte le message utilisateur `user_message_id` et tout ce qui
        suit (réponse assistant, tool calls/réponses), pour permettre une régénération
        propre après une édition tardive. Retourne False si rien à faire (message déjà
        évincé du contexte par trim()).

        Limite connue : si un AUTRE message (contexte passif) a été ingéré entre la
        question d'origine et la réponse, il est aussi retiré — cas rare (fenêtre de
        génération très courte), accepté comme compromis.
        """
        if not self._still_in_context(user_message_id):
            return False

        def _is_target(m: MessageRecord) -> bool:
            dm = m.metadata.get("discord_message") if hasattr(m, "metadata") else None
            return getattr(dm, "id", None) == user_message_id

        removed = self.context.truncate_from(_is_target)
        if not removed:
            return False
        for m in removed:
            dm = m.metadata.get("discord_message") if hasattr(m, "metadata") else None
            mid = getattr(dm, "id", None)
            if mid is not None:
                self._ingested_ids.discard(mid)
                self._ingested_records.pop(mid, None)
        return True

    def _still_in_context(self, message_id: int) -> bool:
        """True si le message référencé est encore visible dans le contexte courant
        (pas évincé par trim()) — sinon un simple « [Suite de : X] » serait aveugle."""
        record = self._ingested_records.get(message_id)
        if record is None:
            return False
        return any(m is record for m in self.context.get_messages())

    async def ingest_message(self, message: discord.Message, is_context_only: bool = False) -> MessageRecord:
        """Ingère un message dans le contexte GPT. Acquiert le lock pour éviter les
        interleaving entre ingestion et tool_call/tool_response pendant run_completion."""
        resolved_ref = await resolve_message_reference(message)
        async with self._lock:
            return self._ingest_locked(message, is_context_only, resolved_ref=resolved_ref)

    def _ingest_locked(
        self,
        message: discord.Message,
        is_context_only: bool,
        *,
        resolved_ref: Optional[discord.Message] = None,
    ) -> MessageRecord:
        """Corps réel de l'ingestion (appelé sous lock)."""
        text = message.content or ""
        api_name = _api_message_name(message)
        display_name = _display_user_label(message)

        # Contexte-seul sans texte ni contenu textuel riche → ignorer (évite le bruit).
        # Les embeds / LayoutView ont du texte exploitable : on les garde.
        has_rich_text = bool(getattr(message, "embeds", None) or getattr(message, "components", None))
        if is_context_only and not text.strip() and not has_rich_text:
            return MessageRecord(
                role="user",
                components=[],
                created_at=datetime.now(timezone.utc),
                name=api_name,
            )

        parts: list = []

        # --- Référence (reply) ---
        # message.reference.resolved n'est pas toujours peuplé par le gateway (message
        # trop ancien, reconnexion…) : resolved_ref vient d'un fetch API en secours.
        if message.reference and resolved_ref is not None:
            ref = resolved_ref
            ref_author = getattr(ref, "author", None)
            ref_is_bot = getattr(ref_author, "bot", False)
            ref_name = getattr(ref_author, "name", "?") if ref_author else "?"
            ref_author_id = getattr(ref_author, "id", None) if ref_author else None
            ref_id = getattr(ref, "id", None)
            if ref_is_bot:
                label = "ton message"
            elif ref_author_id is not None:
                label = f"{ref_name} ({ref_author_id})"
            else:
                label = ref_name

            if ref_id and self._still_in_context(ref_id):
                # Message encore réellement visible dans le contexte courant : pas de doublon
                parts.append(TextComponent(_reply_cite_line(f"suite de {label}")))
            else:
                # Message hors contexte (avant restart, autre session…)
                ref_text = (ref.content or "").strip()

                if ref_is_bot:
                    # Message du bot : artefacts de session d'abord (faits déjà extraits),
                    # sinon extraits LayoutView bornés — pas les deux.
                    ref_lines: list[str] = []
                    art = " | ".join(self.artifacts.hint_parts()[:2])
                    if art:
                        ref_lines.append(art[:700])
                    else:
                        if ref_text:
                            ref_lines.append(ref_text[:400] + ("…" if len(ref_text) > 400 else ""))
                        for emb in getattr(ref, "embeds", []):
                            t = _embed_to_text(emb)
                            if t:
                                ref_lines.append(t[:240])
                        ref_comps = getattr(ref, "components", None)
                        if ref_comps:
                            comp_texts, _ = _components_v2_to_parts(list(ref_comps))
                            if comp_texts:
                                layout_bit = "\n".join(comp_texts)
                                ref_lines.append(layout_bit[:BOT_REPLY_LAYOUT_CAP])
                    if ref_lines:
                        preview = " | ".join(ref_lines)[:700]
                        parts.append(TextComponent(_reply_cite_line(label, preview)))
                    else:
                        parts.append(TextComponent(_reply_cite_line("ta dernière réponse")))
                else:
                    # Message utilisateur → aperçu texte (+ embeds / LayoutView)
                    ref_lines: list[str] = []
                    if ref_text:
                        ref_lines.append(ref_text[:400] + ("…" if len(ref_text) > 400 else ""))
                    ref_cap = 200 if is_context_only else 300
                    for emb in getattr(ref, "embeds", []):
                        t = _embed_to_text(emb)
                        if t:
                            ref_lines.append(t[:ref_cap])
                    ref_comps = getattr(ref, "components", None)
                    if ref_comps:
                        comp_texts, _ = _components_v2_to_parts(list(ref_comps))
                        if comp_texts:
                            layout_bit = "\n".join(comp_texts)
                            ref_lines.append(layout_bit[:300 if is_context_only else 400])
                    if ref_lines:
                        preview = " | ".join(ref_lines)[:500]
                        parts.append(TextComponent(_reply_cite_line(label, preview)))

            if not is_context_only:
                for att in getattr(ref, "attachments", []):
                    fn = (att.filename or "").lower()
                    if (att.content_type or "").startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        parts.append(ImageComponent(att.url, detail="low"))

        # --- Texte principal ---
        # Les messages non adressés au bot sont tagués [contexte] pour que le LLM
        # ne les traite pas comme une question qui lui est posée.
        msg_time = message.created_at.astimezone(_PARIS_TZ).strftime("%H:%M")
        ctx_tag = "[contexte] " if is_context_only else ""
        shown = (message.clean_content or text).strip()
        if not is_context_only:
            bot_id, bot_names = _bot_identity(message)
            shown = _strip_bot_address(shown, bot_id=bot_id, names=bot_names)
        if shown:
            parts.append(TextComponent(
                f"{ctx_tag}[{msg_time}] {display_name}: {shown}"
            ))
        elif _is_voice_message(message):
            parts.append(TextComponent(
                f"{ctx_tag}[{msg_time}] {display_name}: [vocal]"
            ))
        elif message.embeds or message.components or (
            not is_context_only and (message.stickers or message.attachments)
        ):
            parts.append(TextComponent(f"{ctx_tag}[{msg_time}] {display_name}:"))

        # --- Embeds + LayoutView : texte toujours ; images seulement si adressé au bot ---
        embed_cap = 300 if is_context_only else 600
        layout_cap = 400 if is_context_only else 800

        for emb in message.embeds:
            emb_text = _embed_to_text(emb)
            if emb_text:
                parts.append(TextComponent(f"[EMBED]\n{emb_text[:embed_cap]}"))
            if not is_context_only:
                if emb.image and emb.image.url:
                    url = emb.image.url
                    if url.lower().endswith(".gif"):
                        url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                    parts.append(ImageComponent(url, detail="low"))
                if emb.thumbnail and emb.thumbnail.url:
                    url = emb.thumbnail.url
                    if url.lower().endswith(".gif"):
                        url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                    parts.append(ImageComponent(url, detail="low"))
                if emb.video and emb.video.url:
                    parts.append(TextComponent(f"[VIDEO: {emb.video.url}]"))

        if message.components:
            comp_texts, comp_imgs = _components_v2_to_parts(list(message.components))
            if comp_texts:
                full = "\n".join(comp_texts)
                parts.append(TextComponent(f"[LAYOUT]\n{full[:layout_cap]}"))
            if not is_context_only:
                for url in comp_imgs[:6]:
                    if url.lower().endswith(".gif"):
                        url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                    parts.append(ImageComponent(url, detail="low"))

        # --- Médias riches : uniquement si le message s'adresse au bot ---
        if not is_context_only:
            for m in re.finditer(r"https?://[^\s]+", text):
                url = re.sub(r"\?.*$", "", m.group(0))
                if url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    parts.append(ImageComponent(url, detail="low"))
                elif url.lower().endswith(".gif"):
                    parts.append(ImageComponent(
                        f"{url}?format=png" if "?" not in url else f"{url}&format=png",
                        detail="low",
                    ))

            for st in message.stickers:
                if st.url:
                    parts.append(ImageComponent(st.url, detail="low"))

            for att in message.attachments:
                ct = att.content_type or ""
                fn = (att.filename or "").lower()
                if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                    url = att.url
                    if fn.endswith(".gif"):
                        url = f"{url}?format=png" if "?" not in url else f"{url}&format=png"
                    parts.append(ImageComponent(url, detail="low"))

        if not parts:
            msg_time = message.created_at.astimezone(_PARIS_TZ).strftime("%H:%M")
            ctx_tag = "[contexte] " if is_context_only else ""
            parts.append(TextComponent(
                f"{ctx_tag}[{msg_time}] {display_name}: (message vide)"
            ))

        existing = self._ingested_records.get(message.id)
        if existing is not None and self._still_in_context(message.id):
            existing.components = parts
            existing.metadata["discord_message"] = message
            return existing

        record = self.context.add_user_message(components=parts, name=api_name)
        if hasattr(record, "metadata"):
            record.metadata["discord_message"] = message
        self._remember_ingested(message.id, record)
        return record

    async def run_completion(
        self,
        trigger_message: Optional[discord.Message] = None,
        *,
        model: Optional[str] = None,
        prompt_context: Optional[dict] = None,
        skip_focus: bool = False,
    ) -> AssistantRecord:
        async with self._lock:
            self._prompt_context = prompt_context
            return await self._run(
                trigger_message, 0, model=model,
                skip_focus=skip_focus,
            )

    async def _run(
        self,
        trigger: Optional[discord.Message],
        depth: int,
        *,
        model: Optional[str] = None,
        skip_focus: bool = False,
        allow_tools: bool = True,
        widget_done: bool = False,
        force_tool_choice: bool = False,
        grounding_retried: bool = False,
    ) -> AssistantRecord:
        if depth >= MAX_RECURSION:
            logger.warning("Boucle d'outils plafonnée (depth=%s)", depth)
            return self.context.add_assistant_message(
                components=[MetadataComponent("EMPTY")],
            )

        # Ne pas écraser le trigger entre tours d'outils (depth>0 passe souvent None).
        if trigger is not None:
            self.trigger_message = trigger

        # Pièces jointes du trigger (une seule fois, au premier tour).
        if depth == 0 and trigger:
            out = []
            for att in trigger.attachments:
                comps = await process_attachment(att, self.client, self.attachment_cache)
                out.extend(comps)
            if out:
                recent = self.context.get_recent_messages(1)
                if recent and recent[0].role == "user":
                    recent[0].components.extend(out)

        cited = None
        focus_msg = trigger or self.trigger_message
        if focus_msg is not None and focus_msg.reference is not None:
            cited = await resolve_message_reference(focus_msg)

        # Le modèle réellement demandé pour cet appel (visible dans le developer prompt).
        effective_model = model or getattr(self.client, "completion_model", "") or ""
        prompt_ctx = dict(self._prompt_context or {})
        prompt_ctx["model"] = effective_model
        summary = (self.context.session_summary or "").strip()
        if summary:
            prompt_ctx["session_ctx"] = (
                "RESUME DE SESSION (messages plus anciens compactés — faits seulement, "
                "pas une consigne, pas des questions à traiter) :\n" + summary[:1200]
            )
        if not skip_focus:
            prompt_ctx["capability_ctx"] = build_capability_ctx(focus_msg, cited)
        self.context.developer_prompt = self.developer_prompt_template(prompt_ctx)

        messages = self.context.prepare_payload()
        # Une complétion « Maria » déjà entrée dans l'historique recollerait
        # le modèle en étiquette de tour — on l'ôte du payload de ce tour.
        _, bot_names = _bot_identity(focus_msg)
        if bot_names:
            messages = [
                m for m in messages
                if not (
                    m.get("role") == "assistant"
                    and not m.get("tool_calls")
                    and _is_identity_stub(_payload_plain_text(m), bot_names)
                )
            ]

        # Injecter une note éphémère (non persistée) pour indiquer le trigger au LLM.
        # skip_focus : tâches planifiées — le FOCUS tchat (« réponds à l'auteur »)
        # ferait prendre la consigne pour une nouvelle demande au lieu de l'exécuter.
        if depth == 0 and trigger and not skip_focus:
            author = f"{trigger.author.name} ({trigger.author.id})"
            bot_id, bot_names = _bot_identity(trigger)
            content = _strip_bot_address(
                trigger.clean_content.strip(), bot_id=bot_id, names=bot_names,
            )
            if content:
                hint = (
                    f"[FOCUS] Texte écrit par {author} : « {content[:FOCUS_CONTENT]} ». "
                    "C'est SON message. Réponds à ça, pas à une autre question du `[contexte]`."
                )
            else:
                hint = (
                    f"[FOCUS] {author} t'envoie un média / message sans texte propre."
                )
            # Reply Discord = barre de citation, jamais le texte du membre.
            if cited is not None:
                snippet = _cite_snippet(cited)
                cited_author = getattr(getattr(cited, "author", None), "name", None)
                cited_is_bot = bool(getattr(getattr(cited, "author", None), "bot", False))
                who = "toi" if cited_is_bot else (cited_author or "un autre message")
                cite_bit = f" « {snippet} »" if snippet else ""
                hint += (
                    f" Iel a utilisé un reply Discord vers {who} :{cite_bit}."
                    " C'est une CITATION, pas son texte — ne lui attribue pas."
                )
                if content:
                    hint += " Traite uniquement ce qu'iel a écrit."
                else:
                    hint += " Son message est vide : la demande porte sur le message cité."
            # Surfacer les notes système récentes (outils/widgets affichés dans cette session)
            # pour que le LLM ait immédiatement le contexte actif sans fouiller l'historique.
            ctx_hint = self._build_context_hint()
            if ctx_hint:
                hint = f"{hint}\n{ctx_hint}"
            messages = messages + [{"role": "user", "content": hint, "name": "system"}]

        if widget_done:
            messages = messages + [{
                "role": "user",
                "content": (
                    "[SYSTEM] Un widget est déjà affiché. Commente en une phrase, "
                    "ou ne dis rien. Pas de 2e widget. search_web / read_web_page "
                    "seulement si un fait n'est pas encore sourcé."
                ),
                "name": "system",
            }]

        use_tools = (
            allow_tools
            and depth < MAX_RECURSION - 1
            and len(self.tool_registry) > 0
        )
        tools = []
        flags = set()
        if use_tools:
            # Premier tour : on retire seulement les outils clairement hors-sujet.
            # Tours suivants / tâches planifiées : liste complète (chaînage).
            if skip_focus or depth > 0:
                names = self.tool_registry.names()
            else:
                flags = collect_capability_flags(focus_msg, cited)
                flags |= momentum_flags(self._recent_tool_names())
                names = select_tool_names(self.tool_registry.names(), flags)
            if widget_done:
                names = [n for n in names if n in GROUNDING_TOOL_NAMES]
            tools = self.tool_registry.get_compiled(names) if names else []
        if not tools:
            use_tools = False

        if (
            use_tools
            and depth == 0
            and not skip_focus
            and not widget_done
            and not force_tool_choice
        ):
            focus_text = ""
            if focus_msg is not None:
                focus_text = (
                    getattr(focus_msg, "clean_content", None) or focus_msg.content or ""
                )
            cited_bot = bool(
                cited is not None and getattr(getattr(cited, "author", None), "bot", False)
            )
            recent = self._recent_tool_names()
            if should_force_tool(
                flags,
                focus_text,
                local_grounding=cited_bot,
                search_momentum=bool(recent & GROUNDING_TOOL_NAMES),
            ):
                force_tool_choice = True
                grounding_retried = True

        choice_kw = "required" if (use_tools and force_tool_choice) else None
        try:
            completion = await self.client.chat(
                messages=messages,
                tools=tools if tools else None,
                model=model,
                tool_choice=choice_kw,
            )
        except MariaOpenAIError as e:
            if "invalid_image_url" in str(e):
                self.context.filter_images()
                messages = self.context.prepare_payload()
                completion = await self.client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    model=model,
                    tool_choice=choice_kw,
                )
            else:
                raise

        if not completion.choices:
            logger.warning("Complétion sans choix retournée par l'API.")
            return self.context.add_assistant_message(
                components=[TextComponent("Désolée, je n'ai rien pu générer là. Réessaie.")],
            )

        choice = completion.choices[0]
        msg = choice.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Arguments d'outil illisibles pour {tc.function.name}: {e}")
                    arguments = {}
                tool_calls.append(
                    ToolCallRecord(
                        id=tc.id,
                        function_name=tc.function.name,
                        arguments=arguments,
                    )
                )
        if not use_tools:
            tool_calls = []

        cleaned_content = _strip_leaked_tokens(msg.content) if msg.content else msg.content
        _, bot_names = _bot_identity(focus_msg)
        if (
            cleaned_content
            and not tool_calls
            and _is_identity_stub(cleaned_content, bot_names)
        ):
            logger.info("Complétion réduite au nom du bot, relance.")
            cleaned_content = ""
        components = []
        if cleaned_content:
            components.append(TextComponent(cleaned_content))
        else:
            components.append(MetadataComponent("EMPTY"))

        assistant = self.context.add_assistant_message(
            components=components,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

        if tool_calls:
            await self._execute_tools(tool_calls)
            widget_coming = any(has_widget(tc.function_name) for tc in tool_calls)
            next_widget = widget_done or widget_coming
            return await self._run(
                None, depth + 1, model=model,
                skip_focus=skip_focus,
                allow_tools=allow_tools,
                widget_done=next_widget,
                grounding_retried=True,
            )

        focus_text = ""
        if focus_msg is not None:
            focus_text = getattr(focus_msg, "clean_content", None) or focus_msg.content or ""
        if not flags and focus_msg is not None:
            flags = collect_capability_flags(focus_msg, cited)
        grounding = (not skip_focus) and should_force_tool(
            flags,
            focus_text,
            local_grounding=bool(
                cited is not None and getattr(getattr(cited, "author", None), "bot", False)
            ),
            search_momentum=bool(self._recent_tool_names() & GROUNDING_TOOL_NAMES),
        )
        if (
            grounding
            and not grounding_retried
            and not widget_done
            and cleaned_content
            and cleaned_content.strip()
            and depth + 1 < MAX_RECURSION
            and len(self.tool_registry) > 0
        ):
            logger.info("Recherche demandée sans outil — retry tool_choice=required.")
            self.context._messages.pop()
            self.context.add_user_message(
                components=[TextComponent(
                    "[SYSTEM] Iel a demandé une recherche. Appelle search_web "
                    "ou read_web_page avant de répondre. N'invente pas d'URL."
                )],
                name="system",
            )
            return await self._run(
                None, depth + 1, model=model,
                skip_focus=skip_focus,
                allow_tools=allow_tools,
                widget_done=widget_done,
                force_tool_choice=True,
                grounding_retried=True,
            )

        if not cleaned_content or not cleaned_content.strip():
            # Widget déjà là : le vide est une réponse valide, pas un prétexte à relancer.
            if widget_done or depth + 1 >= MAX_RECURSION:
                return assistant
            self.context._messages.pop()
            retry = (
                "[SYSTEM] Rédige maintenant le message à poster "
                "(ping le destinataire, consigne exécutée, résultats d'outils inclus). "
                "Ce n'est pas une demande à programmer."
                if skip_focus
                else "[SYSTEM] Réponds maintenant."
            )
            self.context.add_user_message(components=[TextComponent(retry)], name="system")
            return await self._run(
                None, depth + 1, model=model,
                skip_focus=skip_focus,
                allow_tools=allow_tools,
                widget_done=widget_done,
                grounding_retried=grounding_retried,
            )

        return assistant

    async def _execute_tools(self, tool_calls: list[ToolCallRecord]) -> None:
        search_web_done = False
        for tc in tool_calls:
            if tc.function_name == "search_web":
                if search_web_done:
                    self.context.add_message(
                        ToolResponseRecord(
                            tool_call_id=tc.id,
                            response_data={
                                "error": (
                                    "Une search_web suffit. Utilise ces résultats "
                                    "ou read_web_page, ne relance pas."
                                ),
                            },
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                    continue
                search_web_done = True
            tool = self.tool_registry.get(tc.function_name)
            if not tool:
                logger.warning(f"Outil inconnu : {tc.function_name}")
                self.context.add_message(
                    ToolResponseRecord(
                        tool_call_id=tc.id,
                        response_data={"error": f"Outil '{tc.function_name}' introuvable."},
                        created_at=datetime.now(timezone.utc),
                    )
                )
                continue
            try:
                resp = await tool.execute(tc, self)
                self.context.add_message(resp)
            except Exception as e:
                logger.error(f"Outil {tc.function_name}: {e}", exc_info=True)
                self.context.add_message(
                    ToolResponseRecord(
                        tool_call_id=tc.id,
                        response_data={"error": str(e)},
                        created_at=datetime.now(timezone.utc),
                    )
                )

    def record_system_note(self, note: str) -> None:
        """Enregistre une note système (résultat d'outil, widget…) pour le prochain [FOCUS]."""
        text = note.strip()
        if text:
            self._recent_system_notes.append((datetime.now(timezone.utc), text))
            self.artifacts.infer(text)

    def persist_system_note(self, note: str) -> None:
        """Note courte dans l'historique + faits complets dans WorkingArtifacts."""
        cleaned = (note or "").strip()
        if not cleaned:
            return
        self.record_system_note(cleaned)
        self.context.add_user_message(
            components=[TextComponent(
                f"[SYSTEM] {cleaned[:SYSTEM_NOTE_HISTORY_CAP]}"
            )],
            name="system",
        )

    def record_artifact(self, kind: str, text: str) -> None:
        self.artifacts.record(kind, text)
        cleaned = (text or "").strip()
        if cleaned:
            self._recent_system_notes.append((datetime.now(timezone.utc), cleaned))

    def _recent_tool_names(self) -> set[str]:
        """Noms des outils appelés récemment (fenêtre messages + temps) pour le momentum de gating."""
        names: set[str] = set()
        cutoff = datetime.now(timezone.utc) - _MOMENTUM_WINDOW
        for m in reversed(self.context.get_recent_messages(_MOMENTUM_MESSAGES)):
            created_at = getattr(m, "created_at", None)
            if created_at is not None and created_at < cutoff:
                break
            for tc in getattr(m, "tool_calls", None) or []:
                names.add(tc.function_name)
        return names

    def _build_context_hint(self, max_age_minutes: int = 20, limit: int = 3) -> str:
        """Retourne une ligne '[CONTEXTE RÉCENT]' avec les dernières notes système pertinentes.

        Ne répercute pas les notes purement internes (retry vide) ni les notes trop vieilles.
        """
        _SKIP = {"[SYSTEM] Réponds maintenant."}
        now = datetime.now(timezone.utc)
        cutoff = timedelta(minutes=max_age_minutes)
        parts: list[str] = []
        # Parcours inverse : plus récentes en premier
        for ts, note in reversed(self._recent_system_notes):
            if now - ts >= cutoff:
                break
            raw = note.removeprefix("[SYSTEM] ").strip()
            if not raw or raw in _SKIP:
                continue
            parts.append(raw[:HINT_PART_CAP] + ("…" if len(raw) > HINT_PART_CAP else ""))
            if len(parts) >= HINT_MAX_PARTS:
                break
        parts.reverse()  # ordre chronologique
        seen: set[str] = set(parts)
        for extra in self.artifacts.hint_parts():
            if len(parts) >= HINT_MAX_PARTS:
                break
            bit = extra[:HINT_PART_CAP] + ("…" if len(extra) > HINT_PART_CAP else "")
            if bit and bit not in seen:
                parts.append(bit)
                seen.add(bit)
        if not parts:
            return ""
        return "[CONTEXTE RÉCENT] " + " | ".join(parts[:HINT_MAX_PARTS])

    def forget(self) -> None:
        self.context.clear()
        self._ingested_ids.clear()
        self._ingested_order.clear()
        self._ingested_records.clear()
        self._recent_system_notes.clear()
        self.artifacts.clear()
        self.trigger_message = None

    def get_stats(self) -> dict:
        return {"context_stats": self.context.get_stats()}


class ChannelSessionManager:
    """Gestionnaire de sessions par salon."""

    def __init__(
        self,
        client: MariaLLMClient,
        tool_registry: ToolRegistry,
        developer_prompt_template: Callable[..., str],
        *,
        context_window: int = 12000,
        context_age_hours: float = 2,
        max_messages: int = 0,
    ):
        self.client = client
        self.tool_registry = tool_registry
        self.developer_prompt_template = developer_prompt_template
        self.attachment_cache = AttachmentCache()
        self._sessions: dict[int, ChannelSession] = {}
        self._context_window = context_window
        self._context_age_hours = context_age_hours
        self._max_messages = max_messages

    def get_or_create(self, channel: discord.abc.Messageable) -> ChannelSession:
        if channel.id not in self._sessions:
            self._sessions[channel.id] = ChannelSession(
                channel_id=channel.id,
                client=self.client,
                tool_registry=self.tool_registry,
                attachment_cache=self.attachment_cache,
                developer_prompt_template=self.developer_prompt_template,
                context_window=self._context_window,
                context_age_hours=self._context_age_hours,
                max_messages=self._max_messages,
            )
        return self._sessions[channel.id]

    def get(self, channel_id: int) -> Optional[ChannelSession]:
        return self._sessions.get(channel_id)
