"""Vues Discord (Components v2) du cog Chat — mémoire perso/collective, tâches.

Extrait de chat.py pour garder ce dernier centré sur l'orchestration LLM.
Contient aussi quelques helpers d'ingestion mémoire tightly coupled à ces vues
(_build_memory_ingest_text etc.), utilisés par chat.py au moment du on_message.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

import discord

from common.emojis import PENCIL, REPEAT_REMINDER, SMALL_TASK
from common.memory.store import (
    CATEGORY_EVENT,
    CATEGORY_SERVER,
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
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PENDING as TASK_PENDING,
    TASK_INSTRUCTION_MAX,
    TASK_MAX_PENDING,
    ScheduledTask,
    TaskStore,
    format_schedule,
    normalize_time_of_day,
)
from common.timezones import PARIS_TZ

from cogs.chat.config import MODEL_MAIN

_VIEW_TIMEOUT = 300
_MEM_PAGE = 25
_TASK_PAGE = 5


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


def _is_memory_mod(member: discord.Member | discord.User) -> bool:
    if not isinstance(member, discord.Member):
        return False
    perms = member.guild_permissions
    return bool(perms.administrator or perms.manage_guild or perms.manage_messages)


def _upsert_vector(vectors: VectorStore, mem: Memory) -> None:
    if mem.status != STATUS_ACTIVE:
        return
    vectors.upsert(
        mem.id,
        mem.content,
        category=mem.category,
        guild_id=mem.guild_id,
        user_id=mem.user_id,
        confidence=mem.confidence,
    )


def _sorted_memories(memories: list[Memory]) -> list[Memory]:
    return sorted(
        memories,
        key=lambda m: (0 if m.status != STATUS_PENDING else 1, -float(m.confidence or 0)),
    )


def _mem_status_label(m: Memory) -> str:
    return "en attente" if m.status == STATUS_PENDING else "confirmé"


def _mem_conf(m: Memory) -> str:
    return f"{m.confidence:.0%}"


def _clip(text: str, n: int) -> str:
    raw = (text or "").strip().replace("\n", " ")
    if len(raw) <= n:
        return raw
    return raw[: n - 1] + "…"


def _format_memory_catalog(items: list[Memory]) -> str:
    pending = [m for m in items if m.status == STATUS_PENDING]
    active = [m for m in items if m.status != STATUS_PENDING]
    blocks: list[str] = []

    def _block(title: str, group: list[Memory]) -> None:
        if not group:
            return
        lines = [f"### {title} · {len(group)}"]
        for m in group:
            lines.append(f"**{_mem_conf(m)}** · {_clip(m.content, 160)}")
        blocks.append("\n".join(lines))

    _block("Confirmés", active)
    _block("En attente", pending)
    return "\n\n".join(blocks) if blocks else "-# Aucun souvenir."


def _memory_option(m: Memory) -> discord.SelectOption:
    return discord.SelectOption(
        label=_clip(m.content, 100) or "(vide)",
        value=m.id,
        description=f"{_mem_status_label(m)} · {_mem_conf(m)}"[:100],
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
        page: int = 0,
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
        self.page = page
        self.fact = discord.ui.TextInput(
            label="Souvenir",
            style=discord.TextStyle.paragraph,
            max_length=MEMORY_CONTENT_MAX,
            required=True,
            default=(memory.content or "")[:MEMORY_CONTENT_MAX],
        )
        self.add_item(self.fact)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.scope, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        content = self.fact.value.strip()
        if not content:
            return await interaction.response.send_message("Info vide.", ephemeral=True)
        await interaction.response.defer()
        mem = await asyncio.to_thread(self.store.replace_content, self.memory.id, content)
        if mem:
            _upsert_vector(self.vectors, mem)
        view = await _rebuild_memory_list(
            interaction,
            store=self.store, vectors=self.vectors,
            guild_id=self.guild_id, user_id=self.user_id,
            display_name=self.display_name, guild_name=self.guild_name,
            scope=self.scope, note="Souvenir modifié.", page=self.page,
        )
        await interaction.edit_original_response(view=view)


def _memory_deny(interaction: discord.Interaction, scope: str, user_id: int) -> Optional[str]:
    if scope == "me" and interaction.user.id != user_id:
        return "C'est pas ta mémoire."
    if scope == "global" and not _is_memory_mod(interaction.user):
        return "Réservé aux modos du serveur."
    return None


def _memory_pages(memories: list[Memory]) -> list[list[Memory]]:
    ordered = _sorted_memories(memories)
    if not ordered:
        return [[]]
    return [ordered[i:i + _MEM_PAGE] for i in range(0, len(ordered), _MEM_PAGE)]


def _memory_cat_label(m: Memory) -> str:
    return {
        CATEGORY_USER: "perso",
        CATEGORY_SERVER: "serveur",
        CATEGORY_EVENT: "event",
    }.get(m.category, m.category)


async def _forget_one(
    store: MemoryStore,
    vectors: VectorStore,
    mem: Memory,
    *,
    scope: str,
    user_id: int,
    guild_id: int,
) -> bool:
    if scope == "me":
        ok, chroma = await asyncio.to_thread(store.forget_user_memory, mem.id, user_id)
    else:
        ok, chroma = await asyncio.to_thread(store.forget_server_memory, mem.id, guild_id)
    if chroma:
        vectors.delete(chroma)
    return ok


async def _rebuild_memory_list(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    user_id: int,
    display_name: str,
    scope: str,
    guild_name: str = "",
    note: str = "",
    page: int = 0,
) -> discord.ui.LayoutView:
    if scope == "me":
        return await _rebuild_me_view(
            interaction,
            store=store, vectors=vectors,
            guild_id=guild_id, user_id=user_id,
            display_name=display_name, note=note, page=page,
        )
    return await _rebuild_global_view(
        interaction,
        store=store, vectors=vectors,
        guild_id=guild_id, guild_name=guild_name or display_name,
        note=note, page=page,
    )


# ---------------------------------------------------------------------------
# Sous-menu souvenir (un select → actions contextuelles)
# ---------------------------------------------------------------------------

class _MemBackButton(discord.ui.Button):
    def __init__(self, **kw):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.kw["scope"], self.kw["user_id"])
        if err and self.kw["scope"] == "me":
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        view = await _rebuild_memory_list(interaction, **self.kw)
        await interaction.edit_original_response(view=view)


class _MemConfirmButton(discord.ui.Button):
    def __init__(self, mem: Memory, **kw):
        super().__init__(style=discord.ButtonStyle.success, label="Confirmer")
        self.mem = mem
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.kw["scope"], self.kw["user_id"])
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        mem = await asyncio.to_thread(self.kw["store"].promote_direct, self.mem.id)
        if mem:
            _upsert_vector(self.kw["vectors"], mem)
        view = await _rebuild_memory_list(
            interaction, **self.kw, note="Souvenir confirmé.",
        )
        await interaction.edit_original_response(view=view)


class _MemRejectButton(discord.ui.Button):
    def __init__(self, mem: Memory, **kw):
        super().__init__(style=discord.ButtonStyle.danger, label="Rejeter")
        self.mem = mem
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.kw["scope"], self.kw["user_id"])
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        ok = await _forget_one(
            self.kw["store"], self.kw["vectors"], self.mem,
            scope=self.kw["scope"], user_id=self.kw["user_id"],
            guild_id=self.kw["guild_id"],
        )
        view = await _rebuild_memory_list(
            interaction, **self.kw,
            note="Souvenir rejeté." if ok else "Souvenir introuvable.",
        )
        await interaction.edit_original_response(view=view)


class _MemForgetButton(discord.ui.Button):
    def __init__(self, mem: Memory, **kw):
        super().__init__(style=discord.ButtonStyle.danger, label="Oublier")
        self.mem = mem
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.kw["scope"], self.kw["user_id"])
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        ok = await _forget_one(
            self.kw["store"], self.kw["vectors"], self.mem,
            scope=self.kw["scope"], user_id=self.kw["user_id"],
            guild_id=self.kw["guild_id"],
        )
        view = await _rebuild_memory_list(
            interaction, **self.kw,
            note="Souvenir oublié." if ok else "Souvenir introuvable.",
        )
        await interaction.edit_original_response(view=view)


class _MemEditButton(discord.ui.Button):
    def __init__(self, mem: Memory, **kw):
        super().__init__(style=discord.ButtonStyle.primary, label="Modifier")
        self.mem = mem
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self.kw["scope"], self.kw["user_id"])
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(EditMemoryModal(
            self.kw["store"], self.kw["vectors"], self.mem,
            guild_id=self.kw["guild_id"], user_id=self.kw["user_id"],
            display_name=self.kw["display_name"], scope=self.kw["scope"],
            guild_name=self.kw.get("guild_name") or "",
            can_manage=True, page=self.kw.get("page") or 0,
        ))


class MemoryDetailView(discord.ui.LayoutView):
    """Fiche d'un souvenir — actions selon statut (attente / confirmé)."""

    def __init__(
        self,
        mem: Memory,
        *,
        store: MemoryStore,
        vectors: VectorStore,
        guild_id: int,
        user_id: int,
        display_name: str,
        scope: str,
        guild_name: str = "",
        can_manage: bool = False,
        page: int = 0,
        note: str = "",
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        kw = dict(
            store=store, vectors=vectors, guild_id=guild_id, user_id=user_id,
            display_name=display_name, scope=scope, guild_name=guild_name,
            page=page,
        )
        pending = mem.status == STATUS_PENDING
        meta = f"{_mem_status_label(mem)} · {_mem_conf(mem)}"
        if scope == "global":
            meta += f" · {_memory_cat_label(mem)}"
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay("## Souvenir"),
            discord.ui.TextDisplay(f"-# {meta}"),
            discord.ui.Separator(),
            discord.ui.TextDisplay((mem.content or "").strip() or "-# (vide)"),
        ]
        rows: list[discord.ui.ActionRow] = []
        actions: list[discord.ui.Button] = []
        if can_manage:
            if pending:
                actions += [
                    _MemConfirmButton(mem, **kw),
                    _MemRejectButton(mem, **kw),
                    _MemEditButton(mem, **kw),
                ]
            else:
                actions += [
                    _MemEditButton(mem, **kw),
                    _MemForgetButton(mem, **kw),
                ]
        actions.append(_MemBackButton(**kw))
        rows.append(discord.ui.ActionRow(*actions))
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))


class _PickMemorySelect(discord.ui.Select):
    def __init__(self, items: list[Memory], **kw):
        super().__init__(
            placeholder="Ouvrir un souvenir…",
            options=[_memory_option(m) for m in items[:_MEM_PAGE]],
        )
        self.items = {m.id: m for m in items}
        self.kw = kw

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.kw["scope"] == "me" and interaction.user.id != self.kw["user_id"]:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        mem = self.items.get(self.values[0])
        if mem is None:
            return await interaction.response.send_message("Souvenir introuvable.", ephemeral=True)
        can_manage = (
            self.kw["scope"] == "me"
            or _is_memory_mod(interaction.user)
        )
        await interaction.response.edit_message(
            view=MemoryDetailView(mem, can_manage=can_manage, **self.kw),
        )


class _MemPageButton(discord.ui.Button):
    def __init__(self, label: str, delta: int, rebuild):
        super().__init__(style=discord.ButtonStyle.secondary, label=label)
        self.delta = delta
        self.rebuild = rebuild

    async def callback(self, interaction: discord.Interaction) -> None:
        await interaction.response.edit_message(view=self.rebuild(self.delta))


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


def _build_memory_catalog_view(
    *,
    title: str,
    subtitle: str,
    summary: str,
    memories: list[Memory],
    page: int,
    note: str,
    select_kw: dict,
    extra_buttons: list[discord.ui.Button],
    rebuild,
) -> list[discord.ui.Item]:
    pages = _memory_pages(memories)
    page = max(0, min(page, len(pages) - 1))
    shown = pages[page]
    pending_n = sum(1 for m in memories if m.status == STATUS_PENDING)
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## {title}"),
        discord.ui.TextDisplay(subtitle),
        discord.ui.Separator(),
        discord.ui.TextDisplay(summary),
    ]
    if memories:
        children += [
            discord.ui.Separator(),
            discord.ui.TextDisplay(_format_memory_catalog(shown)),
            discord.ui.TextDisplay(
                f"-# Page {page + 1}/{len(pages)} · {len(memories)} souvenir(s)"
                + (f" · {pending_n} en attente" if pending_n else "")
            ),
        ]
    else:
        children += [
            discord.ui.Separator(),
            discord.ui.TextDisplay("-# Aucun souvenir pour l'instant."),
        ]

    rows: list[discord.ui.ActionRow] = []
    if shown and shown[0].id:
        rows.append(discord.ui.ActionRow(
            _PickMemorySelect(shown, **select_kw, page=page),
        ))
    top: list[discord.ui.Button] = list(extra_buttons)
    if len(pages) > 1:
        if page > 0:
            top.append(_MemPageButton("Precedent", -1, rebuild))
        if page < len(pages) - 1:
            top.append(_MemPageButton("Suivant", 1, rebuild))
    if top:
        rows.append(discord.ui.ActionRow(*top[:5]))
    _append_controls(children, note=note, rows=rows)
    return children


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
        pending_n = sum(1 for m in personal if m.status == STATUS_PENDING)
        subtitle = (
            f"-# Classé par statut, puis confiance"
            + (f" · {pending_n} à confirmer" if pending_n else "")
        )

        def rebuild(delta: int) -> MeMemoryView:
            return MeMemoryView(
                display_name, summary, memories,
                store=store, vectors=vectors,
                guild_id=guild_id, user_id=user_id,
                note=note, page=page + delta,
            )

        children = _build_memory_catalog_view(
            title=f"Mémoire · {display_name}",
            subtitle=subtitle,
            summary=summary,
            memories=personal,
            page=page,
            note=note,
            select_kw=dict(
                store=store, vectors=vectors, guild_id=guild_id,
                user_id=user_id, display_name=display_name, scope="me",
            ),
            extra_buttons=[
                _AddPersonalMemoryButton(store, vectors, guild_id, user_id, display_name),
                _ResetPersonalButton(store, vectors, guild_id, user_id, display_name),
            ],
            rebuild=rebuild,
        )
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
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        pending_n = sum(1 for m in memories if m.status == STATUS_PENDING)
        subtitle = (
            "-# Classé par statut, puis confiance"
            + (f" · {pending_n} à confirmer" if pending_n else "")
        )
        extra: list[discord.ui.Button] = []
        if can_manage:
            extra.append(_ResetServerButton(store, vectors, guild_id, guild_name))

        def rebuild(delta: int) -> AllMemoryView:
            return AllMemoryView(
                guild_name, summary, memories,
                store=store, vectors=vectors, guild_id=guild_id,
                can_manage=can_manage, note=note, page=page + delta,
            )

        children = _build_memory_catalog_view(
            title=f"Mémoire · {guild_name}",
            subtitle=subtitle,
            summary=summary,
            memories=memories,
            page=page,
            note=note,
            select_kw=dict(
                store=store, vectors=vectors, guild_id=guild_id,
                user_id=0, display_name=guild_name, scope="global",
                guild_name=guild_name,
            ),
            extra_buttons=extra,
            rebuild=rebuild,
        )
        self.add_item(discord.ui.Container(*children))


async def _rebuild_global_view(
    interaction: discord.Interaction,
    *,
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    guild_name: str,
    note: str = "",
    page: int = 0,
) -> AllMemoryView:
    memories = await asyncio.to_thread(
        lambda: store.list_server(guild_id, limit=80, include_pending=True),
    )
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
        note=note, page=page,
    )

# ---------------------------------------------------------------------------
# Tâches — /taches
# ---------------------------------------------------------------------------

def _task_deny(interaction: discord.Interaction, user_id: int) -> Optional[str]:
    if interaction.user.id != user_id:
        return "C'est pas tes tâches."
    return None


def _task_status_label(t: ScheduledTask) -> str:
    if t.status == STATUS_PAUSED:
        return "en pause"
    if t.status == STATUS_FAILED:
        return "échec"
    return "active"


def _task_rank(t: ScheduledTask) -> int:
    if t.status == STATUS_PAUSED:
        return 1
    if t.status == STATUS_FAILED:
        return 2
    return 0


def _sorted_tasks(tasks: list[ScheduledTask]) -> list[ScheduledTask]:
    return sorted(
        tasks,
        key=lambda t: (_task_rank(t), t.execute_at.timestamp() if t.execute_at else 0),
    )


def _task_pages(tasks: list[ScheduledTask]) -> list[list[ScheduledTask]]:
    ordered = _sorted_tasks(tasks)
    if not ordered:
        return [[]]
    return [ordered[i:i + _TASK_PAGE] for i in range(0, len(ordered), _TASK_PAGE)]


def _task_heading(t: ScheduledTask) -> str:
    raw = (t.title or "").strip() or (t.instruction or "").strip() or "Sans consigne"
    return f"**{_clip(raw, 120)}**"


def _task_meta(t: ScheduledTask) -> str:
    ts = int(t.execute_at.timestamp())
    bits: list[str] = [_task_status_label(t)]
    if t.schedule_kind != SCHEDULE_ONCE:
        bits.append(f"{REPEAT_REMINDER} {format_schedule(t)}")
    bits.append(f"<t:{ts}:R>")
    if t.deliver_dm:
        bits.append("MP")
    return " · ".join(bits)


def _task_section(
    store: TaskStore,
    user_id: int,
    t: ScheduledTask,
    *,
    page: int,
) -> discord.ui.Section:
    lines: list[discord.ui.TextDisplay] = [
        discord.ui.TextDisplay(_task_heading(t)),
        discord.ui.TextDisplay(f"-# {_task_meta(t)}"),
    ]
    if t.last_error:
        lines.append(discord.ui.TextDisplay(f"-# {_clip(t.last_error, 80)}"))
    return discord.ui.Section(
        *lines,
        accessory=_OpenTaskButton(store, user_id, t, page=page),
    )


def _format_task_body(t: ScheduledTask) -> str:
    ts = int(t.execute_at.timestamp())
    rec = format_schedule(t)
    if t.schedule_kind != SCHEDULE_ONCE:
        rec = f"{REPEAT_REMINDER} {rec}"
        if t.until_at:
            rec += f" · jusqu'au <t:{int(t.until_at.timestamp())}:d>"
    if t.deliver_dm:
        rec += " · MP"
    err = f"\n-# Dernière erreur : {t.last_error}" if t.last_error else ""
    return (
        f"-# {_task_status_label(t)} · {rec}\n"
        f"-# Prochaine : <t:{ts}:f> (<t:{ts}:R>){err}"
    )


def _reload_tasks(store: TaskStore, user_id: int, *, note: str = "", page: int = 0) -> "TasksView":
    return TasksView(store, user_id, store.get_user_tasks(user_id), note=note, page=page)


class EditTaskModal(discord.ui.Modal, title="Modifier la tâche"):
    def __init__(self, store: TaskStore, user_id: int, task: ScheduledTask, *, page: int = 0):
        super().__init__()
        self.store = store
        self.user_id = user_id
        self.task = task
        self.page = page
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
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
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
        await interaction.edit_original_response(
            view=_reload_tasks(self.store, self.user_id, note="Tâche modifiée.", page=self.page),
        )


class _TaskBackButton(discord.ui.Button):
    def __init__(self, store, user_id, page: int = 0):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self.store = store
        self.user_id = user_id
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=_reload_tasks(self.store, self.user_id, page=self.page),
        )


class _PauseTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask, *, page: int = 0):
        paused = task.status == STATUS_PAUSED
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Reprendre" if paused else "Pause",
        )
        self.store = store
        self.user_id = user_id
        self.task = task
        self.paused = paused
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        if self.paused:
            ok = self.store.resume(self.task.id, self.user_id)
            note = "Tâche reprise." if ok else "Impossible de reprendre."
        else:
            ok = self.store.pause(self.task.id, self.user_id)
            note = "Tâche en pause." if ok else "Impossible de mettre en pause."
        await interaction.response.edit_message(
            view=_reload_tasks(self.store, self.user_id, note=note, page=self.page),
        )


class _SkipTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask, *, page: int = 0):
        super().__init__(style=discord.ButtonStyle.secondary, label="Sauter")
        self.store = store
        self.user_id = user_id
        self.task = task
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        nxt = self.store.skip_next(self.task.id, self.user_id)
        note = "Prochaine occurrence sautée." if nxt else "Pas de prochaine occurrence."
        await interaction.response.edit_message(
            view=_reload_tasks(self.store, self.user_id, note=note, page=self.page),
        )


class _EditTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask, *, page: int = 0):
        super().__init__(style=discord.ButtonStyle.primary, label="Modifier")
        self.store = store
        self.user_id = user_id
        self.task = task
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(
            EditTaskModal(self.store, self.user_id, self.task, page=self.page),
        )


class _CancelTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask, *, page: int = 0):
        super().__init__(style=discord.ButtonStyle.danger, label="Annuler")
        self.store = store
        self.user_id = user_id
        self.task = task
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        ok = self.store.cancel(self.task.id, self.user_id)
        await interaction.response.edit_message(
            view=_reload_tasks(
                self.store, self.user_id,
                note="Tâche annulée." if ok else "Tâche introuvable.",
                page=self.page,
            ),
        )


class _CancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout annuler")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=ConfirmCancelAllTasksView(self.store, self.user_id),
        )


class _ConfirmCancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        n = self.store.cancel_all(self.user_id)
        await interaction.response.edit_message(
            view=TasksView(self.store, self.user_id, [], note=f"{n} tâche(s) annulée(s)."),
        )


class _CancelCancelAllTasksButton(discord.ui.Button):
    def __init__(self, store, user_id):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self.store = store
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=_reload_tasks(self.store, self.user_id),
        )


class ConfirmCancelAllTasksView(discord.ui.LayoutView):
    def __init__(self, store: TaskStore, user_id: int):
        super().__init__(timeout=60)
        self.add_item(discord.ui.Container(
            discord.ui.TextDisplay("## Tâches"),
            discord.ui.Separator(),
            discord.ui.TextDisplay("Annuler **toutes** tes tâches (séries incluses) ? Irréversible."),
            discord.ui.Separator(),
            discord.ui.ActionRow(
                _ConfirmCancelAllTasksButton(store, user_id),
                _CancelCancelAllTasksButton(store, user_id),
            ),
        ))


class TaskDetailView(discord.ui.LayoutView):
    """Fiche d'une tâche — actions selon statut / récurrence."""

    def __init__(
        self,
        store: TaskStore,
        user_id: int,
        task: ScheduledTask,
        *,
        page: int = 0,
        note: str = "",
    ):
        super().__init__(timeout=_VIEW_TIMEOUT)
        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {SMALL_TASK} Tâche"),
            discord.ui.Separator(),
            discord.ui.TextDisplay((task.instruction or "").strip() or "-# Sans consigne."),
            discord.ui.TextDisplay(_format_task_body(task)),
        ]
        actions: list[discord.ui.Button] = [
            _EditTaskButton(store, user_id, task, page=page),
        ]
        if task.schedule_kind != SCHEDULE_ONCE or task.status == STATUS_PAUSED:
            actions.append(_PauseTaskButton(store, user_id, task, page=page))
        if task.schedule_kind != SCHEDULE_ONCE:
            actions.append(_SkipTaskButton(store, user_id, task, page=page))
        actions += [
            _CancelTaskButton(store, user_id, task, page=page),
            _TaskBackButton(store, user_id, page),
        ]
        rows = [discord.ui.ActionRow(*actions)]
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))


class _OpenTaskButton(discord.ui.Button):
    def __init__(self, store, user_id, task: ScheduledTask, *, page: int = 0):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            emoji=discord.PartialEmoji.from_str(PENCIL),
        )
        self.store = store
        self.user_id = user_id
        self.task = task
        self.page = page

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.edit_message(
            view=TaskDetailView(self.store, self.user_id, self.task, page=self.page),
        )


class TasksView(discord.ui.LayoutView):
    """Catalogue des tâches — /taches."""

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
        pages = _task_pages(tasks)
        page = max(0, min(page, len(pages) - 1))
        shown = pages[page]
        quota_n = sum(1 for t in tasks if t.status in (TASK_PENDING, STATUS_PAUSED))
        paused_n = sum(1 for t in tasks if t.status == STATUS_PAUSED)
        subtitle = "-# Classé par prochaine exécution"
        if paused_n:
            subtitle += f" · {paused_n} en pause"
        if len(pages) > 1:
            subtitle += f" · page {page + 1}/{len(pages)}"

        children: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {SMALL_TASK} Tâches · {quota_n}/{TASK_MAX_PENDING}"),
            discord.ui.TextDisplay(subtitle),
        ]
        if not tasks:
            children.append(discord.ui.TextDisplay("-# Aucune tâche en attente."))
        else:
            for t in shown:
                children.append(discord.ui.Separator())
                children.append(_task_section(store, user_id, t, page=page))

        extra: list[discord.ui.Button] = []
        if tasks:
            extra.append(_CancelAllTasksButton(store, user_id))

        def rebuild(delta: int) -> TasksView:
            return TasksView(
                store, user_id, tasks, note=note, page=page + delta,
            )

        if len(pages) > 1:
            if page > 0:
                extra.append(_MemPageButton("Precedent", -1, rebuild))
            if page < len(pages) - 1:
                extra.append(_MemPageButton("Suivant", 1, rebuild))
        rows: list[discord.ui.ActionRow] = []
        if extra:
            rows.append(discord.ui.ActionRow(*extra[:5]))
        _append_controls(children, note=note, rows=rows)
        self.add_item(discord.ui.Container(*children))
