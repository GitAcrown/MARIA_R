"""Vues Discord (Components v2) du cog Chat — mémoire perso/collective, rappels.

Extrait de chat.py pour garder ce dernier centré sur l'orchestration LLM.
Contient aussi quelques helpers d'ingestion mémoire tightly coupled à ces vues
(_build_memory_ingest_text etc.), utilisés par chat.py au moment du on_message.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import discord

from common.memory.store import CATEGORY_USER, STATUS_ACTIVE, Memory, MemoryStore
from common.memory.summary import summarize_memories
from common.memory.vector import VectorStore
from common.rappels import RECURRENCE_NONE, REPEAT_EMOJI, Rappel, RappelStore
from common.timezones import PARIS_TZ

from cogs.chat.config import MODEL_MAIN

# ---------------------------------------------------------------------------
# UI — composants réutilisables
# ---------------------------------------------------------------------------

class InfoView(discord.ui.LayoutView):
    """Stats de la session en cours — lecture seule."""

    def __init__(
        self,
        stats: Optional[dict],
        channel,
        *,
        mode: str = "strict",
        stream: bool = False,
    ):
        super().__init__(timeout=60)
        ch_name = getattr(channel, "name", str(getattr(channel, "id", "?")))

        header = discord.ui.TextDisplay(f"## {ch_name}")
        sep = discord.ui.Separator()

        mode_labels = {"off": "Désactivé", "strict": "Mention uniquement", "greedy": "Mention + nom"}
        mode_str = mode_labels.get(mode, mode)
        stream_str = "oui" if stream else "non"
        config = discord.ui.TextDisplay(f"**Mode** · {mode_str}\n**Streaming** · {stream_str}")

        if stats:
            ctx = stats["context_stats"]
            pct = ctx["window_usage_pct"]
            filled = int(20 * pct / 100)
            bar = "█" * filled + "░" * (20 - filled)
            session = discord.ui.TextDisplay(
                f"**Messages** · {ctx['total_messages']}\n"
                f"**Tokens** · {ctx['total_tokens']:,} / {ctx['context_window']:,}\n"
                f"`{bar}` {pct:.0f}%"
            )
        else:
            session = discord.ui.TextDisplay("-# Aucune session active.")

        self.add_item(discord.ui.Container(header, sep, config, discord.ui.Separator(), session))


class AddPersonalMemoryModal(discord.ui.Modal, title="À retenir sur moi"):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__()
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name
        self.fact = discord.ui.TextInput(
            label="Info à retenir",
            placeholder="Ex: J'habite à Lyon · Anniversaire le 12 mars · Je déteste le café",
            style=discord.TextStyle.paragraph,
            max_length=280,
            required=True,
        )
        self.add_item(self.fact)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        content = self.fact.value.strip()
        if not content:
            return await interaction.response.send_message("Info vide.", ephemeral=True)
        await interaction.response.defer(ephemeral=True)
        mem = await asyncio.to_thread(
            self.store.create,
            category=CATEGORY_USER,
            guild_id=self.guild_id,
            content=content,
            user_id=self.user_id,
            confidence=1.0,
            status=STATUS_ACTIVE,
        )
        self.vectors.upsert(
            mem.id, mem.content,
            category=mem.category, guild_id=mem.guild_id,
            user_id=mem.user_id, confidence=mem.confidence,
        )
        memories = await asyncio.to_thread(
            self.store.list_for_user, self.guild_id, self.user_id, limit=40, include_server=False,
        )
        summary = f"› {mem.content}"
        chat_cog = interaction.client.get_cog("Chat")
        if chat_cog is not None and hasattr(chat_cog, "gpt_api"):
            summary = await summarize_memories(
                chat_cog.gpt_api.client,
                model=MODEL_MAIN,
                memories=memories,
                scope="user",
                display_name=self.display_name,
            )
        view = MeMemoryView(
            self.display_name, summary, memories,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            note="Souvenir retenu.",
        )
        await interaction.edit_original_response(view=view)


class _AddPersonalMemoryButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.primary, label="Retenir…")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.send_modal(
            AddPersonalMemoryModal(
                self.store, self.vectors, self.guild_id, self.user_id, self.display_name,
            )
        )


class _ResetPersonalButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout oublier")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.edit_message(
            view=ConfirmResetMeView(
                self.store, self.vectors, self.guild_id, self.user_id, self.display_name,
            ),
        )


class _ConfirmResetMeButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.defer()
        chroma_ids = await asyncio.to_thread(self.store.clear_user, self.user_id)
        for mid in chroma_ids:
            self.vectors.delete(mid)
        view = MeMemoryView(
            self.display_name, "Rien de notable pour l'instant.", [],
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            note="Mémoire perso vidée.",
        )
        await interaction.edit_original_response(view=view)


class _CancelResetMeButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.defer()
        view = await _rebuild_me_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name,
        )
        await interaction.edit_original_response(view=view)


class ConfirmResetMeView(discord.ui.LayoutView):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
    ):
        super().__init__(timeout=60)
        self.add_item(discord.ui.Container(
            discord.ui.TextDisplay(f"## Mémoire · {display_name}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay("Effacer **toute** ta mémoire perso ? Irréversible."),
            discord.ui.Separator(),
            discord.ui.ActionRow(
                _ConfirmResetMeButton(store, vectors, guild_id, user_id, display_name),
                _CancelResetMeButton(store, vectors, guild_id, user_id, display_name),
            ),
        ))


class MeMemoryView(discord.ui.LayoutView):
    """Mémoire personnelle — /moi."""

    def __init__(
        self,
        display_name: str,
        summary: str,
        memories: list[Memory],
        *,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        note: str = "",
    ):
        super().__init__(timeout=180)
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## Mémoire · {display_name}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(summary),
        ]
        personal = [m for m in memories if m.category == "user" or m.user_id == user_id]
        if personal:
            shown = personal[:_MEMORY_LIST_LIMIT]
            lines = [_memory_line(m) for m in shown]
            if len(personal) > _MEMORY_LIST_LIMIT:
                lines.append(f"-# … +{len(personal) - _MEMORY_LIST_LIMIT}")
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("\n".join(lines)),
            ]
        else:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("-# Aucun souvenir perso pour l'instant."),
            ]
        _append_controls(
            children,
            note=note,
            button_row=discord.ui.ActionRow(
                _AddPersonalMemoryButton(store, vectors, guild_id, user_id, display_name),
                _ResetPersonalButton(store, vectors, guild_id, user_id, display_name),
            ),
            select_row=(
                discord.ui.ActionRow(
                    _ForgetPersonalSelect(store, vectors, guild_id, user_id, display_name, personal),
                )
                if personal
                else None
            ),
        )
        self.add_item(discord.ui.Container(*children))


def _ui_note_text(note: str) -> str:
    """Normalise une notification UI en style -#."""
    text = (note or "").strip()
    if not text:
        return ""
    if text.startswith("-#"):
        return text
    return f"-# {text}"


def _append_controls(
    children: list[discord.ui.Item],
    *,
    note: str = "",
    button_row: discord.ui.ActionRow | None = None,
    select_row: discord.ui.ActionRow | None = None,
) -> None:
    """Pied de vue dynamique : notif optionnelle, puis boutons, puis select (rows séparées)."""
    notif = _ui_note_text(note)
    if notif:
        children += [
            discord.ui.Separator(),
            discord.ui.TextDisplay(notif),
        ]
    if button_row is not None or select_row is not None:
        children.append(discord.ui.Separator())
        if button_row is not None:
            children.append(button_row)
        if select_row is not None:
            if button_row is not None:
                children.append(discord.ui.Separator())
            children.append(select_row)


_MEMORY_LIST_LIMIT = 5
_MEMORY_LINE_MAX = 72


def _memory_line(m: Memory) -> str:
    content = m.content.strip()
    if len(content) > _MEMORY_LINE_MAX:
        content = content[: _MEMORY_LINE_MAX - 1] + "…"
    return f"-# › {content} · {m.confidence:.0%}"


def _is_memory_mod(member: discord.Member | discord.User) -> bool:
    """Admins / manage server / manage messages (modos)."""
    if not isinstance(member, discord.Member):
        return False
    perms = member.guild_permissions
    return bool(perms.administrator or perms.manage_guild or perms.manage_messages)


def _memory_resolve_mentions(
    text: str,
    mentions: list,
    *,
    bot_user: Optional[discord.ClientUser],
) -> str:
    """Remplace <@id> par @pseudo(id) lisible pour l'agent mémoire."""
    out = text or ""
    for member in mentions:
        for token in (f"<@{member.id}>", f"<@!{member.id}>"):
            is_bot = bool(
                member.bot or (bot_user is not None and member.id == bot_user.id)
            )
            label = (
                f"@{member.name} (le bot)"
                if is_bot
                else f"@{member.name}({member.id})"
            )
            out = out.replace(token, label)
    return out


def _memory_plain_from_components(components: list, *, depth: int = 0) -> str:
    """Extrait le texte des composants v2 (réponses widget du bot, etc.)."""
    if depth > 5 or not components:
        return ""
    parts: list[str] = []
    for comp in components:
        name = type(comp).__name__
        if name == "TextDisplay":
            content = getattr(comp, "content", None) or getattr(comp, "value", None)
            if content:
                parts.append(str(content).strip())
        elif name in ("Container", "Section", "ActionRow"):
            children = (
                getattr(comp, "children", None)
                or getattr(comp, "components", None)
                or []
            )
            sub = _memory_plain_from_components(list(children), depth=depth + 1)
            if sub:
                parts.append(sub)
    return " ".join(p for p in parts if p)


def _memory_source_text(message: discord.Message) -> str:
    """Texte utile d'un message Discord (content, sinon composants v2)."""
    text = (message.content or "").strip()
    if text:
        return text
    if message.components:
        return _memory_plain_from_components(list(message.components)).strip()
    return ""


def _memory_media_tags(message: discord.Message) -> list[str]:
    """Tags médias / transferts pour que l'agent ne confonde pas avec un fait affirmé."""
    tags: list[str] = []
    for att in message.attachments[:4]:
        fn = (att.filename or "fichier").replace("\n", " ")[:80]
        ct = (att.content_type or "").lower()
        kind = "image" if ct.startswith("image/") or fn.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif")
        ) else "fichier"
        tags.append(f"[{kind}: {fn}]")
    for sticker in message.stickers[:3]:
        name = getattr(sticker, "name", None) or "sticker"
        tags.append(f"[sticker: {name}]")
    for emb in message.embeds[:2]:
        bit = (emb.title or emb.description or "").replace("\n", " ").strip()
        if bit:
            tags.append(f"[embed: {bit[:100]}]")
    for snap in getattr(message, "message_snapshots", None) or []:
        snap_text = (getattr(snap, "content", None) or "").replace("\n", " ").strip()
        if snap_text:
            tags.append(f'[transfère: "{snap_text[:200]}"]')
            continue
        # Forward média sans texte
        for att in getattr(snap, "attachments", None) or []:
            fn = (getattr(att, "filename", None) or "fichier")[:60]
            tags.append(f"[transfère: fichier {fn}]")
            break
    return tags


def _build_memory_ingest_text(
    message: discord.Message,
    *,
    bot_user: Optional[discord.ClientUser],
) -> str:
    """Contenu ingéré par la mémoire : texte + tags médias/transferts."""
    text = _memory_resolve_mentions(
        _memory_source_text(message), message.mentions, bot_user=bot_user,
    )
    tags = _memory_media_tags(message)
    if text and tags:
        return f"{text} {' '.join(tags)}"
    if text:
        return text
    return " ".join(tags)


async def _rebuild_me_view(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    user_id: int,
    display_name: str,
    note: str = "",
) -> MeMemoryView:
    memories = await asyncio.to_thread(
        store.list_for_user, guild_id, user_id, limit=40, include_server=False,
    )
    if memories:
        chat_cog = interaction.client.get_cog("Chat")
        if chat_cog is not None and hasattr(chat_cog, "gpt_api"):
            summary = await summarize_memories(
                chat_cog.gpt_api.client,
                model=MODEL_MAIN,
                memories=memories,
                scope="user",
                display_name=display_name,
            )
        else:
            summary = "\n".join(f"› {m.content}" for m in memories[:8])
    else:
        summary = "Rien de notable pour l'instant."
    return MeMemoryView(
        display_name, summary, memories,
        store=store, vectors=vectors,
        guild_id=guild_id, user_id=user_id, note=note,
    )


async def _rebuild_global_view(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    guild_name: str,
    note: str = "",
) -> AllMemoryView:
    memories = await asyncio.to_thread(store.list_server, guild_id, limit=40)
    if memories:
        chat_cog = interaction.client.get_cog("Chat")
        if chat_cog is not None and hasattr(chat_cog, "gpt_api"):
            summary = await summarize_memories(
                chat_cog.gpt_api.client,
                model=MODEL_MAIN,
                memories=memories,
                scope="server",
                display_name=guild_name,
            )
        else:
            summary = "\n".join(f"› {m.content}" for m in memories[:8])
    else:
        summary = "Rien de notable pour l'instant."
    return AllMemoryView(
        guild_name, summary, memories,
        store=store, vectors=vectors,
        guild_id=guild_id, can_manage=_is_memory_mod(interaction.user), note=note,
    )


class _ForgetPersonalSelect(discord.ui.Select):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
        memories: list[Memory],
    ):
        options = []
        for m in memories[:25]:
            options.append(discord.SelectOption(
                label=m.content[:100],
                value=m.id,
                description=f"conf. {m.confidence:.0%}"[:100],
            ))
        super().__init__(
            placeholder="Oublier un souvenir…",
            options=options,
            min_values=1,
            max_values=1,
        )
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.defer()
        mid = self.values[0]
        ok, chroma = await asyncio.to_thread(self.store.forget_user_memory, mid, self.user_id)
        if chroma:
            self.vectors.delete(chroma)
        note = "Souvenir oublié." if ok else "Souvenir introuvable."
        view = await _rebuild_me_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name, note=note,
        )
        await interaction.edit_original_response(view=view)


class _ForgetServerSelect(discord.ui.Select):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        guild_name: str,
        memories: list[Memory],
    ):
        options = []
        for m in memories[:25]:
            cat = {"server": "collectif", "event": "événement"}.get(m.category, m.category)
            options.append(discord.SelectOption(
                label=m.content[:100],
                value=m.id,
                description=f"{cat} · conf. {m.confidence:.0%}"[:100],
            ))
        super().__init__(
            placeholder="Oublier un souvenir…",
            options=options,
            min_values=1,
            max_values=1,
        )
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        await interaction.response.defer()
        mid = self.values[0]
        ok, chroma = await asyncio.to_thread(self.store.forget_server_memory, mid, self.guild_id)
        if chroma:
            self.vectors.delete(chroma)
        note = "Souvenir oublié." if ok else "Souvenir introuvable."
        view = await _rebuild_global_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, guild_name=self.guild_name, note=note,
        )
        await interaction.edit_original_response(view=view)


class _ResetServerButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        guild_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout oublier")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        await interaction.response.edit_message(
            view=ConfirmResetAllView(
                self.store, self.vectors, self.guild_id, self.guild_name,
            ),
        )


class _ConfirmResetAllButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        guild_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        await interaction.response.defer()
        chroma_ids = await asyncio.to_thread(self.store.clear_server, self.guild_id)
        for mid in chroma_ids:
            self.vectors.delete(mid)
        view = AllMemoryView(
            self.guild_name, "Rien de notable pour l'instant.", [],
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, can_manage=True,
            note="Mémoire collective vidée.",
        )
        await interaction.edit_original_response(view=view)


class _CancelResetAllButton(discord.ui.Button):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        guild_name: str,
    ):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        await interaction.response.defer()
        view = await _rebuild_global_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, guild_name=self.guild_name,
        )
        await interaction.edit_original_response(view=view)


class ConfirmResetAllView(discord.ui.LayoutView):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        guild_name: str,
    ):
        super().__init__(timeout=60)
        self.add_item(discord.ui.Container(
            discord.ui.TextDisplay(f"## Mémoire · {guild_name}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(
                "Effacer **toute** la mémoire collective de ce serveur ? Irréversible."
            ),
            discord.ui.Separator(),
            discord.ui.ActionRow(
                _ConfirmResetAllButton(store, vectors, guild_id, guild_name),
                _CancelResetAllButton(store, vectors, guild_id, guild_name),
            ),
        ))


class AllMemoryView(discord.ui.LayoutView):
    """Mémoire collective — /global."""

    def __init__(
        self,
        guild_name: str,
        summary: str,
        memories: list[Memory],
        *,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        can_manage: bool = False,
        note: str = "",
    ):
        super().__init__(timeout=180)
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## Mémoire · {guild_name}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(summary),
        ]
        if memories:
            by_cat: dict[str, list[Memory]] = {}
            for m in memories:
                by_cat.setdefault(m.category, []).append(m)
            labels = {"server": "Collectif", "event": "Événements"}
            for cat in ("server", "event"):
                items = by_cat.get(cat) or []
                if not items:
                    continue
                shown = items[:_MEMORY_LIST_LIMIT]
                lines = [_memory_line(m) for m in shown]
                if len(items) > _MEMORY_LIST_LIMIT:
                    lines.append(f"-# … +{len(items) - _MEMORY_LIST_LIMIT}")
                children += [
                    discord.ui.Separator(),
                    discord.ui.TextDisplay(f"-# **{labels.get(cat, cat)}**"),
                    discord.ui.TextDisplay("\n".join(lines)),
                ]
        else:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("-# Aucun souvenir serveur pour l'instant."),
            ]
        button_row = (
            discord.ui.ActionRow(_ResetServerButton(store, vectors, guild_id, guild_name))
            if can_manage
            else None
        )
        select_row = (
            discord.ui.ActionRow(
                _ForgetServerSelect(store, vectors, guild_id, guild_name, memories),
            )
            if can_manage and memories
            else None
        )
        _append_controls(children, note=note, button_row=button_row, select_row=select_row)
        self.add_item(discord.ui.Container(*children))


class RecentMemoriesView(discord.ui.LayoutView):
    """Derniers souvenirs créés — /souvenirs (modos)."""

    def __init__(
        self,
        guild: discord.Guild,
        memories: list[Memory],
        *,
        category_label: str = "toutes",
    ):
        super().__init__(timeout=180)
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## Souvenirs récents · {guild.name}"),
            discord.ui.TextDisplay(f"-# Filtre : {category_label} · {len(memories)} ligne(s)"),
        ]
        if not memories:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("-# Aucun souvenir pour ce filtre."),
            ]
        else:
            lines: list[str] = []
            for m in memories:
                ts = int(m.created_at.timestamp())
                who = ""
                if m.user_id:
                    member = guild.get_member(m.user_id)
                    label = member.display_name if member else "?"
                    who = f" · {label} (`{m.user_id}`)"
                content = m.content.strip()
                # Affichage : ids seulement utiles pour les liens ↔ ; sinon bruit.
                if "↔" not in content:
                    content = re.sub(r"\s*\((\d{17,20})\)", "", content).strip()
                    content = re.sub(r"\s{2,}", " ", content)
                if len(content) > _MEMORY_LINE_MAX:
                    content = content[: _MEMORY_LINE_MAX - 1] + "…"
                lines.append(
                    f"-# [`{m.status}`/{m.category}]{who}\n"
                    f"› {content} · <t:{ts}:R> · {m.confidence:.0%}"
                )
            # Discord TextDisplay ~4000 ; on coupe proprement.
            chunk: list[str] = []
            size = 0
            for line in lines:
                add = len(line) + 1
                if chunk and size + add > 3500:
                    children += [
                        discord.ui.Separator(),
                        discord.ui.TextDisplay("\n".join(chunk)),
                    ]
                    chunk = []
                    size = 0
                chunk.append(line)
                size += add
            if chunk:
                children += [
                    discord.ui.Separator(),
                    discord.ui.TextDisplay("\n".join(chunk)),
                ]
        self.add_item(discord.ui.Container(*children))


# ---------------------------------------------------------------------------
# Rappels — /rappels
# ---------------------------------------------------------------------------

_RECURRENCE_LABEL = {
    "daily": "quotidien",
    "weekly": "hebdo",
}


def _format_rappel_line(r: Rappel) -> str:
    ts = int(r.execute_at.timestamp())
    if r.recurrence != RECURRENCE_NONE:
        label = _RECURRENCE_LABEL.get(r.recurrence, r.recurrence)
        until = ""
        if r.recurrence_until:
            until = f" · jusqu'au <t:{int(r.recurrence_until.timestamp())}:d>"
        return f"**#{r.id}** {REPEAT_EMOJI} · <t:{ts}:f> · {label}{until}\n› {r.description}"
    return f"**#{r.id}** · <t:{ts}:f> (<t:{ts}:R>)\n› {r.description}"


class _CancelRappelSelect(discord.ui.Select):
    def __init__(self, store: RappelStore, user_id: int, rappels: list[Rappel]):
        options = []
        for r in rappels[:25]:
            label = f"#{r.id} · {r.description}"[:100]
            desc = "récurrent" if r.recurrence != RECURRENCE_NONE else r.execute_at.astimezone(PARIS_TZ).strftime("%d/%m %H:%M")
            options.append(discord.SelectOption(label=label, value=str(r.id), description=desc[:100]))
        super().__init__(placeholder="Annuler un rappel…", options=options, min_values=1, max_values=1)
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes rappels.", ephemeral=True)
        rid = int(self.values[0])
        ok = self.store.cancel(rid, self.user_id)
        remaining = self.store.get_user_rappels(self.user_id)
        note = f"Rappel #{rid} annulé." if ok else f"Rappel #{rid} introuvable."
        await interaction.response.edit_message(view=RappelsView(self.store, self.user_id, remaining, note=note))


class _CancelAllRappelsButton(discord.ui.Button):
    def __init__(self, store: RappelStore, user_id: int):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout annuler")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes rappels.", ephemeral=True)
        await interaction.response.edit_message(
            view=ConfirmCancelAllRappelsView(self.store, self.user_id),
        )


class _ConfirmCancelAllRappelsButton(discord.ui.Button):
    def __init__(self, store: RappelStore, user_id: int):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes rappels.", ephemeral=True)
        n = self.store.cancel_all(self.user_id)
        await interaction.response.edit_message(
            view=RappelsView(self.store, self.user_id, [], note=f"{n} rappel(s) annulé(s)."),
        )


class _CancelCancelAllRappelsButton(discord.ui.Button):
    def __init__(self, store: RappelStore, user_id: int):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes rappels.", ephemeral=True)
        remaining = self.store.get_user_rappels(self.user_id)
        await interaction.response.edit_message(view=RappelsView(self.store, self.user_id, remaining))


class ConfirmCancelAllRappelsView(discord.ui.LayoutView):
    def __init__(self, store: RappelStore, user_id: int):
        super().__init__(timeout=60)
        self.add_item(discord.ui.Container(
            discord.ui.TextDisplay("## Rappels"),
            discord.ui.Separator(),
            discord.ui.TextDisplay("Annuler **tous** tes rappels en attente (séries incluses) ?"),
            discord.ui.Separator(),
            discord.ui.ActionRow(
                _ConfirmCancelAllRappelsButton(store, user_id),
                _CancelCancelAllRappelsButton(store, user_id),
            ),
        ))


class RappelsView(discord.ui.LayoutView):
    """Gestion des rappels — /rappels."""

    def __init__(
        self,
        store: RappelStore,
        user_id: int,
        rappels: list[Rappel],
        *,
        note: str = "",
    ):
        super().__init__(timeout=180)
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay("## Rappels"),
            discord.ui.Separator(),
        ]
        if not rappels:
            children.append(discord.ui.TextDisplay("-# Aucun rappel en attente."))
            _append_controls(children, note=note)
        else:
            body = "\n\n".join(_format_rappel_line(r) for r in rappels[:8])
            if len(rappels) > 8:
                body += f"\n\n-# … +{len(rappels) - 8}"
            children.append(discord.ui.TextDisplay(body))
            _append_controls(
                children,
                note=note,
                button_row=discord.ui.ActionRow(_CancelAllRappelsButton(store, user_id)),
                select_row=discord.ui.ActionRow(_CancelRappelSelect(store, user_id, rappels)),
            )
        self.add_item(discord.ui.Container(*children))
