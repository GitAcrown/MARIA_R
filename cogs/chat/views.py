"""Vues Discord (Components v2) du cog Chat — mémoire perso/collective, tâches.

Extrait de chat.py pour garder ce dernier centré sur l'orchestration LLM.
Contient aussi quelques helpers d'ingestion mémoire tightly coupled à ces vues
(_build_memory_ingest_text etc.), utilisés par chat.py au moment du on_message.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import discord

from common.emojis import REPEAT_REMINDER
from common.memory.store import (
    CATEGORY_USER,
    MEMORY_CONTENT_MAX,
    STATUS_ACTIVE,
    STATUS_PENDING,
    Memory,
    MemoryStore,
)
from common.memory.summary import summarize_memories
from common.memory.vector import VectorStore
from common.tasks import (
    SCHEDULE_ONCE,
    STATUS_PAUSED,
    STATUS_PENDING as TASK_PENDING,
    TASK_INSTRUCTION_MAX,
    ScheduledTask,
    TaskStore,
    format_schedule,
    normalize_time_of_day,
)
from common.timezones import PARIS_TZ

from cogs.chat.config import MODEL_MAIN

_VIEW_TIMEOUT = 300
_PAGE_BUDGET = 3500


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

        mode_labels = {
            "off": "Désactivé",
            "strict": "Mention ou réponse à MARIA",
            "greedy": "Mention + nom",
        }
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


def _ui_note_text(note: str) -> str:
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
    rows: list[discord.ui.ActionRow] | None = None,
) -> None:
    notif = _ui_note_text(note)
    if notif:
        children += [discord.ui.Separator(), discord.ui.TextDisplay(notif)]
    if rows:
        children.append(discord.ui.Separator())
        for i, row in enumerate(rows):
            if i:
                children.append(discord.ui.Separator())
            children.append(row)


def _paginate(items: list, format_fn: Callable, budget: int = _PAGE_BUDGET) -> list[list]:
    pages: list[list] = []
    current: list = []
    size = 0
    for item in items:
        line = format_fn(item)
        add = len(line) + 2
        if current and size + add > budget:
            pages.append(current)
            current = []
            size = 0
        current.append(item)
        size += add
    if current:
        pages.append(current)
    return pages or [[]]


def _memory_line(m: Memory) -> str:
    content = (m.content or "").strip()
    status = "en attente · " if m.status == STATUS_PENDING else ""
    return f"-# {status}› {content} · {m.confidence:.0%}"


def _is_memory_mod(member: discord.Member | discord.User) -> bool:
    if not isinstance(member, discord.Member):
        return False
    perms = member.guild_permissions
    return bool(perms.administrator or perms.manage_guild or perms.manage_messages)


def _upsert_vector(vectors: VectorStore, mem: Memory) -> None:
    if mem.status != STATUS_ACTIVE:
        return
    vectors.upsert(
        mem.id, mem.content,
        category=mem.category, guild_id=mem.guild_id,
        user_id=mem.user_id, confidence=mem.confidence,
    )


# ---------------------------------------------------------------------------
# Ingestion mémoire (utilisée par chat.py)
# ---------------------------------------------------------------------------

def _memory_resolve_mentions(
    text: str,
    mentions: list,
    *,
    bot_user: Optional[discord.ClientUser],
) -> str:
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
    text = (message.content or "").strip()
    if text:
        return text
    if message.components:
        return _memory_plain_from_components(list(message.components)).strip()
    return ""


def _memory_media_tags(message: discord.Message) -> list[str]:
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
    text = _memory_resolve_mentions(
        _memory_source_text(message), message.mentions, bot_user=bot_user,
    )
    tags = _memory_media_tags(message)
    if text and tags:
        return f"{text} {' '.join(tags)}"
    if text:
        return text
    return " ".join(tags)


# ---------------------------------------------------------------------------
# Mémoire perso — /moi
# ---------------------------------------------------------------------------

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
            placeholder="Ex: J'habite à Lyon · Anniversaire le 12 mars",
            style=discord.TextStyle.paragraph,
            max_length=MEMORY_CONTENT_MAX,
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
        _upsert_vector(self.vectors, mem)
        view = await _rebuild_me_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name, note="Souvenir retenu.",
        )
        await interaction.edit_original_response(view=view)


class EditMemoryModal(discord.ui.Modal, title="Modifier le souvenir"):
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        memory: Memory,
        *,
        guild_id: int,
        user_id: int,
        display_name: str,
        scope: str,
        guild_name: str = "",
        can_manage: bool = False,
        filter_key: str = "actifs",
    ):
        super().__init__()
        self.store = store
        self.vectors = vectors
        self.memory = memory
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name
        self.scope = scope
        self.guild_name = guild_name
        self.can_manage = can_manage
        self.filter_key = filter_key
        self.fact = discord.ui.TextInput(
            label="Souvenir",
            style=discord.TextStyle.paragraph,
            max_length=MEMORY_CONTENT_MAX,
            required=True,
            default=(memory.content or "")[:MEMORY_CONTENT_MAX],
        )
        self.add_item(self.fact)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        content = self.fact.value.strip()
        if not content:
            return await interaction.response.send_message("Info vide.", ephemeral=True)
        await interaction.response.defer()
        mem = await asyncio.to_thread(self.store.replace_content, self.memory.id, content)
        if mem:
            _upsert_vector(self.vectors, mem)
        if self.scope == "me":
            view = await _rebuild_me_view(
                interaction,
                store=self.store, vectors=self.vectors,
                guild_id=self.guild_id, user_id=self.user_id,
                display_name=self.display_name, note="Souvenir modifié.",
            )
        else:
            view = await _rebuild_global_view(
                interaction,
                store=self.store, vectors=self.vectors,
                guild_id=self.guild_id, guild_name=self.guild_name,
                note="Souvenir modifié.", filter_key=self.filter_key,
            )
        await interaction.edit_original_response(view=view)


class _AddPersonalMemoryButton(discord.ui.Button):
    def __init__(self, store, vectors, guild_id, user_id, display_name):
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
    def __init__(self, store, vectors, guild_id, user_id, display_name):
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
    def __init__(self, store, vectors, guild_id, user_id, display_name):
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
    def __init__(self, store, vectors, guild_id, user_id, display_name):
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
    def __init__(self, store, vectors, guild_id, user_id, display_name):
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


class _MemoryPageButton(discord.ui.Button):
    def __init__(self, label: str, delta: int, rebuild):
        super().__init__(style=discord.ButtonStyle.secondary, label=label)
        self.delta = delta
        self.rebuild = rebuild

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(view=self.rebuild(self.delta))


class _ForgetPersonalSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, user_id, display_name, memories: list[Memory]):
        options = []
        for m in memories[:25]:
            prefix = "Attente · " if m.status == STATUS_PENDING else ""
            options.append(discord.SelectOption(
                label=(prefix + m.content)[:100],
                value=m.id,
                description=f"conf. {m.confidence:.0%}"[:100],
            ))
        super().__init__(placeholder="Oublier un souvenir de cette page…", options=options)
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
        view = await _rebuild_me_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name,
            note="Souvenir oublié." if ok else "Souvenir introuvable.",
        )
        await interaction.edit_original_response(view=view)


class _EditPersonalSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, user_id, display_name, memories: list[Memory]):
        options = [
            discord.SelectOption(label=m.content[:100], value=m.id)
            for m in memories[:25]
        ]
        super().__init__(placeholder="Modifier un souvenir de cette page…", options=options)
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name
        self._by_id = {m.id: m for m in memories}

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        mem = self._by_id.get(self.values[0])
        if mem is None:
            return await interaction.response.send_message("Souvenir introuvable.", ephemeral=True)
        await interaction.response.send_modal(EditMemoryModal(
            self.store, self.vectors, mem,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name, scope="me",
        ))


class _PendingPersonalSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, user_id, display_name, pending: list[Memory]):
        options = []
        for m in pending[:12]:
            options.append(discord.SelectOption(
                label=("OK · " + m.content)[:100],
                value=f"c:{m.id}",
                description="Confirmer",
            ))
            options.append(discord.SelectOption(
                label=("Non · " + m.content)[:100],
                value=f"r:{m.id}",
                description="Rejeter",
            ))
        super().__init__(placeholder="Confirmer ou rejeter un souvenir en attente…", options=options[:25])
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.defer()
        raw = self.values[0]
        action, mid = raw.split(":", 1)
        if action == "c":
            mem = await asyncio.to_thread(self.store.promote_direct, mid)
            if mem:
                _upsert_vector(self.vectors, mem)
            note = "Souvenir confirmé."
        else:
            ok, chroma = await asyncio.to_thread(self.store.forget_user_memory, mid, self.user_id)
            if chroma:
                self.vectors.delete(chroma)
            note = "Souvenir rejeté." if ok else "Souvenir introuvable."
        view = await _rebuild_me_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name, note=note,
        )
        await interaction.edit_original_response(view=view)


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
        page: int = 0,
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        personal = [m for m in memories if m.category == "user" or m.user_id == user_id]
        pages = _paginate(personal, _memory_line)
        page = max(0, min(page, len(pages) - 1))
        shown = pages[page]
        pending = [m for m in personal if m.status == STATUS_PENDING]

        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## Mémoire · {display_name}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(summary),
        ]
        if personal:
            footer = f"-# Page {page + 1}/{len(pages)} · {len(personal)} souvenir(s)"
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("\n".join(_memory_line(m) for m in shown)),
                discord.ui.TextDisplay(footer),
            ]
        else:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("-# Aucun souvenir perso pour l'instant."),
            ]

        rows: list[discord.ui.ActionRow] = [
            discord.ui.ActionRow(
                _AddPersonalMemoryButton(store, vectors, guild_id, user_id, display_name),
                _ResetPersonalButton(store, vectors, guild_id, user_id, display_name),
            ),
        ]
        if len(pages) > 1:
            def rebuild(delta: int) -> MeMemoryView:
                return MeMemoryView(
                    display_name, summary, memories,
                    store=store, vectors=vectors,
                    guild_id=guild_id, user_id=user_id,
                    note=note, page=page + delta,
                )
            nav = []
            if page > 0:
                nav.append(_MemoryPageButton("Precedent", -1, rebuild))
            if page < len(pages) - 1:
                nav.append(_MemoryPageButton("Suivant", 1, rebuild))
            if nav:
                rows.append(discord.ui.ActionRow(*nav))
        if shown:
            rows.append(discord.ui.ActionRow(
                _ForgetPersonalSelect(store, vectors, guild_id, user_id, display_name, shown),
            ))
            rows.append(discord.ui.ActionRow(
                _EditPersonalSelect(store, vectors, guild_id, user_id, display_name, shown),
            ))
        if pending:
            rows.append(discord.ui.ActionRow(
                _PendingPersonalSelect(store, vectors, guild_id, user_id, display_name, pending),
            ))
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))


async def _rebuild_me_view(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    user_id: int,
    display_name: str,
    note: str = "",
    page: int = 0,
) -> MeMemoryView:
    memories = await asyncio.to_thread(
        lambda: store.list_for_user(
            guild_id, user_id, limit=80, include_server=False, include_pending=True,
        ),
    )
    if memories:
        chat_cog = interaction.client.get_cog("Chat")
        if chat_cog is not None and hasattr(chat_cog, "gpt_api"):
            summary = await summarize_memories(
                chat_cog.gpt_api.client,
                model=MODEL_MAIN,
                memories=[m for m in memories if m.status == STATUS_ACTIVE],
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
        guild_id=guild_id, user_id=user_id, note=note, page=page,
    )


# ---------------------------------------------------------------------------
# Mémoire collective — /global
# ---------------------------------------------------------------------------

class _ResetServerButton(discord.ui.Button):
    def __init__(self, store, vectors, guild_id, guild_name):
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
            view=ConfirmResetAllView(self.store, self.vectors, self.guild_id, self.guild_name),
        )


class _ConfirmResetAllButton(discord.ui.Button):
    def __init__(self, store, vectors, guild_id, guild_name):
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
    def __init__(self, store, vectors, guild_id, guild_name):
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
    def __init__(self, store, vectors, guild_id, guild_name):
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


class _ForgetServerSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, guild_name, memories, filter_key):
        options = [
            discord.SelectOption(label=m.content[:100], value=m.id)
            for m in memories[:25]
        ]
        super().__init__(placeholder="Oublier un souvenir de cette page…", options=options)
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name
        self.filter_key = filter_key

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
        view = await _rebuild_global_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, guild_name=self.guild_name,
            note="Souvenir oublié." if ok else "Souvenir introuvable.",
            filter_key=self.filter_key,
        )
        await interaction.edit_original_response(view=view)


class _EditServerSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, guild_name, memories, filter_key):
        options = [
            discord.SelectOption(label=m.content[:100], value=m.id)
            for m in memories[:25]
        ]
        super().__init__(placeholder="Modifier un souvenir de cette page…", options=options)
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name
        self.filter_key = filter_key
        self._by_id = {m.id: m for m in memories}

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        mem = self._by_id.get(self.values[0])
        if mem is None:
            return await interaction.response.send_message("Souvenir introuvable.", ephemeral=True)
        await interaction.response.send_modal(EditMemoryModal(
            self.store, self.vectors, mem,
            guild_id=self.guild_id, user_id=interaction.user.id,
            display_name=self.guild_name, scope="global",
            guild_name=self.guild_name, can_manage=True,
            filter_key=self.filter_key,
        ))


class _GlobalFilterSelect(discord.ui.Select):
    def __init__(self, store, vectors, guild_id, guild_name, can_manage, current: str):
        options = [
            discord.SelectOption(label="Actifs", value="actifs", default=current == "actifs"),
            discord.SelectOption(label="Recents", value="recents", default=current == "recents"),
            discord.SelectOption(label="En attente", value="pending", default=current == "pending"),
        ]
        super().__init__(placeholder="Filtre…", options=options)
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.guild_name = guild_name
        self.can_manage = can_manage

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        view = await _rebuild_global_view(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, guild_name=self.guild_name,
            filter_key=self.values[0],
        )
        await interaction.edit_original_response(view=view)


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
        page: int = 0,
        filter_key: str = "actifs",
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        labels = {"actifs": "Actifs", "recents": "Recents", "pending": "En attente"}
        pages = _paginate(memories, _memory_line)
        page = max(0, min(page, len(pages) - 1))
        shown = pages[page]

        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## Mémoire · {guild_name}"),
            discord.ui.TextDisplay(f"-# Filtre : {labels.get(filter_key, filter_key)}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay(summary),
        ]
        if memories:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("\n".join(_memory_line(m) for m in shown)),
                discord.ui.TextDisplay(f"-# Page {page + 1}/{len(pages)} · {len(memories)} souvenir(s)"),
            ]
        else:
            children += [
                discord.ui.Separator(),
                discord.ui.TextDisplay("-# Aucun souvenir pour ce filtre."),
            ]

        rows: list[discord.ui.ActionRow] = []
        if can_manage:
            rows.append(discord.ui.ActionRow(
                _GlobalFilterSelect(store, vectors, guild_id, guild_name, can_manage, filter_key),
            ))
            rows.append(discord.ui.ActionRow(
                _ResetServerButton(store, vectors, guild_id, guild_name),
            ))
        if len(pages) > 1:
            def rebuild(delta: int) -> AllMemoryView:
                return AllMemoryView(
                    guild_name, summary, memories,
                    store=store, vectors=vectors, guild_id=guild_id,
                    can_manage=can_manage, note=note,
                    page=page + delta, filter_key=filter_key,
                )
            nav = []
            if page > 0:
                nav.append(_MemoryPageButton("Precedent", -1, rebuild))
            if page < len(pages) - 1:
                nav.append(_MemoryPageButton("Suivant", 1, rebuild))
            if nav:
                rows.append(discord.ui.ActionRow(*nav))
        if can_manage and shown:
            rows.append(discord.ui.ActionRow(
                _ForgetServerSelect(store, vectors, guild_id, guild_name, shown, filter_key),
            ))
            rows.append(discord.ui.ActionRow(
                _EditServerSelect(store, vectors, guild_id, guild_name, shown, filter_key),
            ))
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))


async def _rebuild_global_view(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    guild_name: str,
    note: str = "",
    filter_key: str = "actifs",
    page: int = 0,
) -> AllMemoryView:
    if filter_key == "recents":
        memories = await asyncio.to_thread(
            lambda: store.list_recent(guild_id, category=None, limit=40, include_pending=True),
        )
    elif filter_key == "pending":
        memories = await asyncio.to_thread(
            lambda: store.list_server(guild_id, limit=40, pending_only=True),
        )
    else:
        memories = await asyncio.to_thread(store.list_server, guild_id, limit=40)
    active = [m for m in memories if m.status == STATUS_ACTIVE]
    if active:
        chat_cog = interaction.client.get_cog("Chat")
        if chat_cog is not None and hasattr(chat_cog, "gpt_api"):
            summary = await summarize_memories(
                chat_cog.gpt_api.client,
                model=MODEL_MAIN,
                memories=active,
                scope="server",
                display_name=guild_name,
            )
        else:
            summary = "\n".join(f"› {m.content}" for m in active[:8])
    else:
        summary = "Rien de notable pour l'instant."
    return AllMemoryView(
        guild_name, summary, memories,
        store=store, vectors=vectors,
        guild_id=guild_id, can_manage=_is_memory_mod(interaction.user),
        note=note, page=page, filter_key=filter_key,
    )


# ---------------------------------------------------------------------------
# Tâches — /taches
# ---------------------------------------------------------------------------

def _format_task_body(t: ScheduledTask) -> str:
    ts = int(t.execute_at.timestamp())
    status = {
        TASK_PENDING: "active",
        STATUS_PAUSED: "en pause",
        "failed": "échec",
    }.get(t.status, t.status)
    rec = ""
    if t.schedule_kind != SCHEDULE_ONCE:
        rec = f" {REPEAT_REMINDER} · {format_schedule(t)}"
        if t.until_at:
            rec += f" · jusqu'au <t:{int(t.until_at.timestamp())}:d>"
    err = f"\n-# Dernière erreur : {t.last_error}" if t.last_error else ""
    return (
        f"**#{t.id}** · {status}{rec}\n"
        f"{t.instruction}\n"
        f"-# Prochaine : <t:{ts}:f> (<t:{ts}:R>){err}"
    )


class EditTaskModal(discord.ui.Modal, title="Modifier la tâche"):
    def __init__(self, store: TaskStore, user_id: int, task: ScheduledTask):
        super().__init__()
        self.store = store
        self.user_id = user_id
        self.task = task
        self.instruction = discord.ui.TextInput(
            label="Consigne",
            style=discord.TextStyle.paragraph,
            max_length=min(TASK_INSTRUCTION_MAX, 1024),
            required=True,
            default=(task.instruction or "")[:1024],
        )
        self.when = discord.ui.TextInput(
            label="Prochaine date (ISO) ou HH:MM",
            style=discord.TextStyle.short,
            required=False,
            max_length=32,
            placeholder="2026-08-20T18:00 ou 18:00",
            default=task.time_of_day or "",
        )
        self.add_item(self.instruction)
        self.add_item(self.when)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        await interaction.response.defer()
        instr = self.instruction.value.strip()
        when = (self.when.value or "").strip()
        execute_at = None
        time_of_day = None
        if when:
            if "T" in when or "-" in when:
                try:
                    dt = datetime.fromisoformat(when)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=PARIS_TZ)
                    execute_at = dt.astimezone(timezone.utc)
                except ValueError:
                    time_of_day = normalize_time_of_day(when, self.task.execute_at)
            else:
                time_of_day = normalize_time_of_day(when, self.task.execute_at)
        self.store.edit(
            self.task.id, self.user_id,
            instruction=instr or None,
            execute_at=execute_at,
            time_of_day=time_of_day,
        )
        remaining = self.store.get_user_tasks(self.user_id)
        await interaction.edit_original_response(
            view=TasksView(self.store, self.user_id, remaining, note="Tâche modifiée."),
        )


class _TaskNavButton(discord.ui.Button):
    def __init__(self, label: str, store, user_id, tasks, page, delta, note=""):
        super().__init__(style=discord.ButtonStyle.secondary, label=label)
        self.store = store
        self.user_id = user_id
        self.tasks = tasks
        self.page = page
        self.delta = delta
        self.note = note

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        await interaction.response.edit_message(
            view=TasksView(
                self.store, self.user_id, self.tasks,
                note=self.note, page=self.page + self.delta,
            ),
        )


class _PauseTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask):
        paused = task.status == STATUS_PAUSED
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Reprendre" if paused else "Pause",
        )
        self.store = store
        self.user_id = user_id
        self.task = task
        self.paused = paused

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        if self.paused:
            ok = self.store.resume(self.task.id, self.user_id)
            note = "Tâche reprise." if ok else "Impossible de reprendre."
        else:
            ok = self.store.pause(self.task.id, self.user_id)
            note = "Tâche en pause." if ok else "Impossible de mettre en pause."
        remaining = self.store.get_user_tasks(self.user_id)
        await interaction.response.edit_message(
            view=TasksView(self.store, self.user_id, remaining, note=note),
        )


class _SkipTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.secondary, label="Sauter")
        self.store = store
        self.user_id = user_id
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        nxt = self.store.skip_next(self.task.id, self.user_id)
        note = "Prochaine occurrence sautée." if nxt else "Pas de prochaine occurrence."
        remaining = self.store.get_user_tasks(self.user_id)
        await interaction.response.edit_message(
            view=TasksView(self.store, self.user_id, remaining, note=note),
        )


class _EditTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.primary, label="Modifier")
        self.store = store
        self.user_id = user_id
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        await interaction.response.send_modal(EditTaskModal(self.store, self.user_id, self.task))


class _CancelTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.danger, label="Annuler")
        self.store = store
        self.user_id = user_id
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        ok = self.store.cancel(self.task.id, self.user_id)
        remaining = self.store.get_user_tasks(self.user_id)
        await interaction.response.edit_message(
            view=TasksView(
                self.store, self.user_id, remaining,
                note="Tâche annulée." if ok else "Tâche introuvable.",
            ),
        )


class _CancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout annuler")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        await interaction.response.edit_message(
            view=ConfirmCancelAllTasksView(self.store, self.user_id),
        )


class _ConfirmCancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        n = self.store.cancel_all(self.user_id)
        await interaction.response.edit_message(
            view=TasksView(self.store, self.user_id, [], note=f"{n} tâche(s) annulée(s)."),
        )


class _CancelCancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        remaining = self.store.get_user_tasks(self.user_id)
        await interaction.response.edit_message(view=TasksView(self.store, self.user_id, remaining))


class ConfirmCancelAllTasksView(discord.ui.LayoutView):
    def __init__(self, store: TaskStore, user_id: int):
        super().__init__(timeout=60)
        self.add_item(discord.ui.Container(
            discord.ui.TextDisplay("## Tâches"),
            discord.ui.Separator(),
            discord.ui.TextDisplay("Annuler **toutes** tes tâches (séries incluses) ?"),
            discord.ui.Separator(),
            discord.ui.ActionRow(
                _ConfirmCancelAllTasksButton(store, user_id),
                _CancelCancelAllTasksButton(store, user_id),
            ),
        ))


class _JumpTaskSelect(discord.ui.Select):
    def __init__(self, store, user_id, tasks: list[ScheduledTask], page: int):
        options = []
        for t in tasks[:25]:
            label = f"#{t.id} · {(t.title or t.instruction)}"[:100]
            options.append(discord.SelectOption(label=label, value=str(t.id)))
        super().__init__(placeholder="Aller à une tâche…", options=options)
        self.store = store
        self.user_id = user_id
        self.tasks = tasks

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
        tid = int(self.values[0])
        idx = next((i for i, t in enumerate(self.tasks) if t.id == tid), 0)
        await interaction.response.edit_message(
            view=TasksView(self.store, self.user_id, self.tasks, page=idx),
        )


class TasksView(discord.ui.LayoutView):
    """Gestion des tâches — /taches. Une tâche par page, consigne complète."""

    def __init__(
        self,
        store: TaskStore,
        user_id: int,
        tasks: list[ScheduledTask],
        *,
        note: str = "",
        page: int = 0,
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        page = max(0, min(page, max(len(tasks) - 1, 0)))
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay("## Tâches"),
            discord.ui.Separator(),
        ]
        rows: list[discord.ui.ActionRow] = []
        if not tasks:
            children.append(discord.ui.TextDisplay("-# Aucune tâche en attente."))
        else:
            current = tasks[page]
            children.append(discord.ui.TextDisplay(_format_task_body(current)))
            children.append(discord.ui.TextDisplay(f"-# {page + 1}/{len(tasks)}"))
            action_btns = [
                _PauseTaskButton(store, user_id, current),
                _EditTaskButton(store, user_id, current),
            ]
            if current.schedule_kind != SCHEDULE_ONCE:
                action_btns.append(_SkipTaskButton(store, user_id, current))
            rows.append(discord.ui.ActionRow(*action_btns))
            rows.append(discord.ui.ActionRow(
                _CancelTaskButton(store, user_id, current),
                _CancelAllTasksButton(store, user_id),
            ))
            if len(tasks) > 1:
                nav = []
                if page > 0:
                    nav.append(_TaskNavButton("Precedent", store, user_id, tasks, page, -1, note))
                if page < len(tasks) - 1:
                    nav.append(_TaskNavButton("Suivant", store, user_id, tasks, page, 1, note))
                if nav:
                    rows.append(discord.ui.ActionRow(*nav))
                rows.append(discord.ui.ActionRow(_JumpTaskSelect(store, user_id, tasks, page)))
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))
