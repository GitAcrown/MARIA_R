"""Session par salon — contexte complet, lock, tools."""

import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

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
    build_capability_ctx,
    collect_capability_flags,
    select_tool_names,
)

logger = logging.getLogger("llm.session")

# Champ OpenAI `messages[].name` : pattern ^[^\s<|\\/>]+$ (pas d'espaces ni |<>\/).
# Le pseudo+id lisible va dans le *contenu* du message, pas dans ce champ.
USER_FORMAT = "{message.author.name}"
MAX_RECURSION = 8

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
# Borne le suivi des IDs déjà ingérés pour éviter une croissance mémoire illimitée par salon.
INGESTED_IDS_MAX = 500

_API_NAME_BAD_RE = re.compile(r"[\s<|\\/>]+")


def _api_message_name(message: discord.Message) -> str:
    """Identifiant stable et valide pour messages[].name (API OpenAI)."""
    raw = (message.author.name or "user").strip() or "user"
    safe = _API_NAME_BAD_RE.sub("_", raw).strip("_") or "user"
    return f"{safe}_{message.author.id}"


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


def _cite_snippet(msg: discord.Message, limit: int = 120) -> str:
    """Aperçu du message cité, pour le [FOCUS] (texte, sinon titre/URL d'embed)."""
    text = (getattr(msg, "clean_content", None) or msg.content or "").strip()
    if text:
        return text[:limit]
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
                parts.append(TextComponent(f"[Suite de : {label}]"))
            else:
                # Message hors contexte (avant restart, autre session…)
                ref_text = (ref.content or "").strip()

                if ref_is_bot:
                    # Message du bot : si texte présent → le citer, sinon note générique
                    # Ne jamais dumper les composants v2 (LayoutView) — c'est du markdown illisible
                    if ref_text:
                        parts.append(TextComponent(
                            f"[Répond à {label} : \"{ref_text[:300]}\"]"
                        ))
                    else:
                        parts.append(TextComponent(f"[Répond à la dernière réponse du bot]"))
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
                        parts.append(TextComponent(f"[Répond à {label} : \"{preview}\"]"))

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
        if text.strip():
            parts.append(TextComponent(
                f"{ctx_tag}[{msg_time}] {display_name}: {message.clean_content}"
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
        on_text_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        on_text_reset: Optional[Callable[[], Awaitable[None]]] = None,
        skip_focus: bool = False,
    ) -> AssistantRecord:
        async with self._lock:
            self._prompt_context = prompt_context
            return await self._run(
                trigger_message, 0, model=model,
                on_text_delta=on_text_delta, on_text_reset=on_text_reset,
                skip_focus=skip_focus,
            )

    async def _run(
        self,
        trigger: Optional[discord.Message],
        depth: int,
        *,
        model: Optional[str] = None,
        on_text_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        on_text_reset: Optional[Callable[[], Awaitable[None]]] = None,
        skip_focus: bool = False,
        allow_tools: bool = True,
        widget_done: bool = False,
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
        if not skip_focus:
            prompt_ctx["capability_ctx"] = build_capability_ctx(focus_msg, cited)
        self.context.developer_prompt = self.developer_prompt_template(prompt_ctx)

        messages = self.context.prepare_payload()

        # Injecter une note éphémère (non persistée) pour indiquer le trigger au LLM.
        # skip_focus : tâches planifiées — le FOCUS tchat (« réponds à l'auteur »)
        # ferait prendre la consigne pour une nouvelle demande au lieu de l'exécuter.
        if depth == 0 and trigger and not skip_focus:
            author = f"{trigger.author.name} ({trigger.author.id})"
            content = trigger.clean_content.strip()
            if content:
                hint = (
                    f"[FOCUS] Réponds UNIQUEMENT à {author} : « {content[:140]} ». "
                    f"Ignore les autres questions du fil ; le `[contexte]` n'est que du décor."
                )
            else:
                hint = (
                    f"[FOCUS] Réponds UNIQUEMENT à {author} "
                    f"(média / message sans texte). Pas aux autres messages du fil."
                )
            # Reply Discord : le message cité est l'objet de la demande, pas un concurrent.
            if cited is not None:
                snippet = _cite_snippet(cited)
                if snippet:
                    hint += (
                        f" Iel répond à : « {snippet} ». La demande porte sur ce contenu "
                        f"(lien, média, propos), pas sur tout le salon — sauf demande explicite."
                    )
                else:
                    hint += (
                        " C'est une réponse Discord : la demande porte sur le message cité "
                        "(lien, média, propos), pas sur tout le salon."
                    )
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
                    "[SYSTEM] Un widget est déjà affiché. "
                    "Commente en une phrase, ou ne dis rien. N'appelle plus d'outil."
                ),
                "name": "system",
            }]

        use_tools = (
            allow_tools
            and depth < MAX_RECURSION - 1
            and len(self.tool_registry) > 0
        )
        tools = []
        if use_tools:
            # Premier tour : on retire seulement les outils clairement hors-sujet.
            # Tours suivants / tâches planifiées : liste complète (chaînage).
            if skip_focus or depth > 0:
                tools = self.tool_registry.get_compiled()
            else:
                flags = collect_capability_flags(focus_msg, cited)
                names = select_tool_names(self.tool_registry.names(), flags)
                tools = self.tool_registry.get_compiled(names)
        stream = on_text_delta is not None

        async def _delta(raw: str) -> None:
            if on_text_delta is None:
                return
            cleaned = _strip_leaked_tokens(raw) if raw else raw
            if cleaned:
                await on_text_delta(cleaned)

        try:
            completion = await self.client.chat(
                messages=messages,
                tools=tools if tools else None,
                model=model,
                stream=stream,
                on_text_delta=_delta if stream else None,
                on_text_reset=on_text_reset if stream else None,
            )
        except MariaOpenAIError as e:
            if "invalid_image_url" in str(e):
                self.context.filter_images()
                messages = self.context.prepare_payload()
                completion = await self.client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    model=model,
                    stream=stream,
                    on_text_delta=_delta if stream else None,
                    on_text_reset=on_text_reset if stream else None,
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
            if on_text_reset is not None:
                try:
                    await on_text_reset()
                except Exception:
                    logger.exception("on_text_reset")
            await self._execute_tools(tool_calls)
            # Un widget va porter le commentaire : ne pas streamer le texte
            # (sinon le message s'affiche puis se fait remplacer).
            widget_coming = any(has_widget(tc.function_name) for tc in tool_calls)
            next_widget = widget_done or widget_coming
            return await self._run(
                None, depth + 1, model=model,
                on_text_delta=None if next_widget else on_text_delta,
                on_text_reset=None if next_widget else on_text_reset,
                skip_focus=skip_focus,
                allow_tools=allow_tools and not next_widget,
                widget_done=next_widget,
            )

        if not cleaned_content or not cleaned_content.strip():
            # Widget déjà là : le vide est une réponse valide, pas un prétexte à relancer.
            if widget_done or depth + 1 >= MAX_RECURSION:
                return assistant
            if on_text_reset is not None:
                try:
                    await on_text_reset()
                except Exception:
                    logger.exception("on_text_reset")
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
                on_text_delta=on_text_delta, on_text_reset=on_text_reset,
                skip_focus=skip_focus,
                allow_tools=allow_tools,
                widget_done=widget_done,
            )

        return assistant

    async def _execute_tools(self, tool_calls: list[ToolCallRecord]) -> None:
        for tc in tool_calls:
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
                logger.error(f"Outil {tc.function_name}: {e}")
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
            parts.append(raw[:280] + ("…" if len(raw) > 280 else ""))
            if len(parts) >= limit:
                break
        if not parts:
            return ""
        parts.reverse()  # ordre chronologique
        return "[CONTEXTE RÉCENT] " + " | ".join(parts)

    def forget(self) -> None:
        self.context.clear()
        self._ingested_ids.clear()
        self._ingested_order.clear()
        self._ingested_records.clear()
        self._recent_system_notes.clear()
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
