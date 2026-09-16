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
from common.menu_layout import (
    HubPageButton,
    HubTabButton,
    MENU_TIMEOUT,
    MariaLayout,
    apply_view,
    send_ephemeral_menu,
    sep_tight,
)
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

_VIEW_TIMEOUT = MENU_TIMEOUT
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
        config = discord.ui.TextDisplay(f"**Mode** · {mode_str}")

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
        hub: Optional["MeMemoryView"] = None,
    ):
        super().__init__()
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.display_name = display_name
        self.hub = hub
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
        if self.hub is not None:
            await self.hub.reload_from_store(interaction, note="Souvenir retenu.")
            return
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
        hub: Optional["_MemoryHub"] = None,
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
        self.hub = hub
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
        if self.hub is not None:
            await self.hub.reload_from_store(interaction, note="Souvenir modifié.")
            return
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
# Hub mémoire (CRIT : un objet, _build + apply_view)
# ---------------------------------------------------------------------------

class _MemBackButton(discord.ui.Button):
    def __init__(self, hub: "_MemoryHub"):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self._hub.scope, self._hub.user_id)
        if err and self._hub.scope == "me":
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.screen = "catalog"
        self._hub.selected = None
        self._hub.note = ""
        self._hub._build()
        await apply_view(interaction, self._hub)


class _MemConfirmButton(discord.ui.Button):
    def __init__(self, hub: "_MemoryHub", mem: Memory):
        super().__init__(style=discord.ButtonStyle.success, label="Confirmer")
        self._hub = hub
        self.mem = mem

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self._hub.scope, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        mem = await asyncio.to_thread(self._hub.store.promote_direct, self.mem.id)
        if mem:
            _upsert_vector(self._hub.vectors, mem)
        await self._hub.reload_from_store(interaction, note="Souvenir confirmé.")


class _MemRejectButton(discord.ui.Button):
    def __init__(self, hub: "_MemoryHub", mem: Memory):
        super().__init__(style=discord.ButtonStyle.danger, label="Rejeter")
        self._hub = hub
        self.mem = mem

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self._hub.scope, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        ok = await _forget_one(
            self._hub.store, self._hub.vectors, self.mem,
            scope=self._hub.scope, user_id=self._hub.user_id,
            guild_id=self._hub.guild_id,
        )
        await self._hub.reload_from_store(
            interaction, note="Souvenir rejeté." if ok else "Souvenir introuvable.",
        )


class _MemForgetButton(discord.ui.Button):
    def __init__(self, hub: "_MemoryHub", mem: Memory):
        super().__init__(style=discord.ButtonStyle.danger, label="Oublier")
        self._hub = hub
        self.mem = mem

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self._hub.scope, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.defer()
        ok = await _forget_one(
            self._hub.store, self._hub.vectors, self.mem,
            scope=self._hub.scope, user_id=self._hub.user_id,
            guild_id=self._hub.guild_id,
        )
        await self._hub.reload_from_store(
            interaction, note="Souvenir oublié." if ok else "Souvenir introuvable.",
        )


class _MemEditButton(discord.ui.Button):
    def __init__(self, hub: "_MemoryHub", mem: Memory):
        super().__init__(style=discord.ButtonStyle.primary, label="Modifier")
        self._hub = hub
        self.mem = mem

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _memory_deny(interaction, self._hub.scope, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(EditMemoryModal(
            self._hub.store, self._hub.vectors, self.mem,
            guild_id=self._hub.guild_id, user_id=self._hub.user_id,
            display_name=self._hub.display_name, scope=self._hub.scope,
            guild_name=self._hub.guild_name, can_manage=True,
            page=self._hub.page, hub=self._hub,
        ))


class _PickMemorySelect(discord.ui.Select):
    def __init__(self, hub: "_MemoryHub", items: list[Memory]):
        super().__init__(
            placeholder="Ouvrir un souvenir…",
            options=[_memory_option(m) for m in items[:_MEM_PAGE]],
        )
        self._hub = hub
        self.items = {m.id: m for m in items}

    async def callback(self, interaction: discord.Interaction) -> None:
        if self._hub.scope == "me" and interaction.user.id != self._hub.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        mem = self.items.get(self.values[0])
        if mem is None:
            return await interaction.response.send_message("Souvenir introuvable.", ephemeral=True)
        self._hub.selected = mem
        self._hub.screen = "detail"
        self._hub._build()
        await apply_view(interaction, self._hub)


class _AddPersonalMemoryButton(discord.ui.Button):
    def __init__(self, hub: "MeMemoryView"):
        super().__init__(style=discord.ButtonStyle.primary, label="Retenir…")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self._hub.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.send_modal(
            AddPersonalMemoryModal(
                self._hub.store, self._hub.vectors, self._hub.guild_id,
                self._hub.user_id, self._hub.display_name, hub=self._hub,
            )
        )


class _ResetPersonalButton(discord.ui.Button):
    def __init__(self, hub: "MeMemoryView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout oublier")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self._hub.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        self._hub.screen = "confirm_reset"
        self._hub._build()
        await apply_view(interaction, self._hub)


class _ConfirmResetMeButton(discord.ui.Button):
    def __init__(self, hub: "MeMemoryView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self._hub.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        await interaction.response.defer()
        chroma_ids = await asyncio.to_thread(self._hub.store.clear_user, self._hub.user_id)
        for mid in chroma_ids:
            self._hub.vectors.delete(mid)
        self._hub.memories = []
        self._hub.summary = "Rien de notable pour l'instant."
        self._hub.screen = "catalog"
        self._hub.note = "Mémoire perso vidée."
        self._hub._build()
        await self._hub.push(interaction)


class _CancelResetMeButton(discord.ui.Button):
    def __init__(self, hub: "MeMemoryView"):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self._hub.user_id:
            return await interaction.response.send_message("C'est pas ta mémoire.", ephemeral=True)
        self._hub.screen = "catalog"
        self._hub.note = ""
        self._hub._build()
        await apply_view(interaction, self._hub)


class _ResetServerButton(discord.ui.Button):
    def __init__(self, hub: "AllMemoryView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout oublier")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        self._hub.screen = "confirm_reset"
        self._hub._build()
        await apply_view(interaction, self._hub)


class _ConfirmResetAllButton(discord.ui.Button):
    def __init__(self, hub: "AllMemoryView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        await interaction.response.defer()
        chroma_ids = await asyncio.to_thread(self._hub.store.clear_server, self._hub.guild_id)
        for mid in chroma_ids:
            self._hub.vectors.delete(mid)
        self._hub.memories = []
        self._hub.summary = "Rien de notable pour l'instant."
        self._hub.screen = "catalog"
        self._hub.note = "Mémoire collective vidée."
        self._hub._build()
        await self._hub.push(interaction)


class _CancelResetAllButton(discord.ui.Button):
    def __init__(self, hub: "AllMemoryView"):
        super().__init__(style=discord.ButtonStyle.secondary, label="Annuler")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        if not _is_memory_mod(interaction.user):
            return await interaction.response.send_message(
                "Réservé aux modos du serveur.", ephemeral=True,
            )
        self._hub.screen = "catalog"
        self._hub.note = ""
        self._hub._build()
        await apply_view(interaction, self._hub)


class _MemoryHub(MariaLayout):
    """Catalogue + détail + confirm, même instance."""

    scope: str = "me"
    guild_name: str = ""
    can_manage: bool = False

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
        viewer_id: Optional[int] = None,
    ):
        super().__init__(viewer_id=viewer_id)
        self.display_name = display_name
        self.summary = summary
        self.memories = memories
        self.store = store
        self.vectors = vectors
        self.guild_id = guild_id
        self.user_id = user_id
        self.note = note
        self.page = page
        self.tab = "all"
        self.screen = "catalog"
        self.selected: Optional[Memory] = None
        self._build()

    def _pool(self) -> list[Memory]:
        return list(self.memories)

    def _visible(self) -> list[Memory]:
        pool = self._pool()
        if self.tab == "pending":
            return [m for m in pool if m.status == STATUS_PENDING]
        return pool

    async def reload_from_store(self, interaction: discord.Interaction, note: str = "") -> None:
        raise NotImplementedError

    def _build_detail(self) -> None:
        mem = self.selected
        if mem is None:
            self.screen = "catalog"
            self._build()
            return
        pending = mem.status == STATUS_PENDING
        meta = f"{_mem_status_label(mem)} · {_mem_conf(mem)}"
        if self.scope == "global":
            meta += f" · {_memory_cat_label(mem)}"
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay("## Souvenir"),
            discord.ui.TextDisplay(f"-# {meta}"),
            sep_tight(),
            discord.ui.TextDisplay((mem.content or "").strip() or "-# (vide)"),
        ]
        actions: list[discord.ui.Button] = []
        manage = self.can_manage or self.scope == "me"
        if manage:
            if pending:
                actions += [
                    _MemConfirmButton(self, mem),
                    _MemRejectButton(self, mem),
                    _MemEditButton(self, mem),
                ]
            else:
                actions += [
                    _MemEditButton(self, mem),
                    _MemForgetButton(self, mem),
                ]
        actions.append(_MemBackButton(self))
        if self.note:
            body += [sep_tight(), discord.ui.TextDisplay(f"-# {self.note}")]
        self.set_layout(body, discord.ui.ActionRow(*actions[:5]))

    def _build_confirm(self) -> None:
        raise NotImplementedError

    def _build_catalog(self) -> None:
        memories = self._visible()
        pages = _memory_pages(memories)
        self.page = max(0, min(self.page, len(pages) - 1))
        shown = pages[self.page]
        pending_n = sum(1 for m in self._pool() if m.status == STATUS_PENDING)
        subtitle = (
            "-# Classé par statut, puis confiance"
            + (f" · {pending_n} à confirmer" if pending_n else "")
        )
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {self._title()}"),
            discord.ui.TextDisplay(subtitle),
            sep_tight(),
            discord.ui.ActionRow(
                HubTabButton(self, "all", "Tous"),
                HubTabButton(self, "pending", f"À confirmer ({pending_n})"),
            ),
            sep_tight(),
            discord.ui.TextDisplay(self.summary),
        ]
        if memories:
            body += [
                sep_tight(),
                discord.ui.TextDisplay(_format_memory_catalog(shown)),
                discord.ui.TextDisplay(
                    f"-# Page {self.page + 1}/{len(pages)} · {len(memories)} souvenir(s)"
                ),
            ]
        else:
            body += [sep_tight(), discord.ui.TextDisplay("-# Aucun souvenir pour l'instant.")]
        rows: list[discord.ui.ActionRow] = []
        if shown and shown[0].id:
            rows.append(discord.ui.ActionRow(_PickMemorySelect(self, shown)))
        extra: list[discord.ui.Button] = list(self._extra_buttons())
        max_page = max(0, len(pages) - 1)
        if max_page > 0:
            if self.page > 0:
                extra.append(HubPageButton(self, "page", -1, "Precedent", max_page))
            if self.page < max_page:
                extra.append(HubPageButton(self, "page", 1, "Suivant", max_page))
        if extra:
            rows.append(discord.ui.ActionRow(*extra[:5]))
        if self.note:
            body += [sep_tight(), discord.ui.TextDisplay(f"-# {self.note}")]
        self.set_layout(body, *rows)

    def _title(self) -> str:
        return f"Mémoire · {self.display_name}"

    def _extra_buttons(self) -> list[discord.ui.Button]:
        return []

    def _build(self) -> None:
        if self.screen == "confirm_reset":
            self._build_confirm()
            return
        if self.screen == "detail":
            self._build_detail()
            return
        self._build_catalog()


class MeMemoryView(_MemoryHub):
    """Mémoire personnelle — /moi."""

    scope = "me"
    can_manage = True

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("viewer_id", kwargs.get("user_id"))
        super().__init__(*args, **kwargs)

    def _pool(self) -> list[Memory]:
        return [m for m in self.memories if m.category == "user" or m.user_id == self.user_id]

    def _extra_buttons(self) -> list[discord.ui.Button]:
        return [
            _AddPersonalMemoryButton(self),
            _ResetPersonalButton(self),
        ]

    def _build_confirm(self) -> None:
        self.set_layout(
            [
                discord.ui.TextDisplay(f"## Mémoire · {self.display_name}"),
                discord.ui.TextDisplay("Effacer **toute** ta mémoire perso ? Irréversible."),
            ],
            discord.ui.ActionRow(
                _ConfirmResetMeButton(self),
                _CancelResetMeButton(self),
            ),
        )

    async def reload_from_store(self, interaction: discord.Interaction, note: str = "") -> None:
        self.memories = await asyncio.to_thread(
            lambda: self.store.list_for_user(
                self.guild_id, self.user_id, limit=80,
                include_server=False, include_pending=True,
            ),
        )
        self.screen = "catalog"
        self.selected = None
        self.note = note
        self._build()
        await self.push(interaction)


class AllMemoryView(_MemoryHub):
    """Mémoire collective — /global."""

    scope = "global"

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
        viewer_id: Optional[int] = None,
    ):
        self.guild_name = guild_name
        self.can_manage = can_manage
        super().__init__(
            guild_name, summary, memories,
            store=store, vectors=vectors, guild_id=guild_id, user_id=0,
            note=note, page=page, viewer_id=viewer_id,
        )

    def _extra_buttons(self) -> list[discord.ui.Button]:
        if self.can_manage:
            return [_ResetServerButton(self)]
        return []

    def _build_confirm(self) -> None:
        self.set_layout(
            [
                discord.ui.TextDisplay(f"## Mémoire · {self.guild_name}"),
                discord.ui.TextDisplay(
                    "Effacer **toute** la mémoire collective de ce serveur ? Irréversible."
                ),
            ],
            discord.ui.ActionRow(
                _ConfirmResetAllButton(self),
                _CancelResetAllButton(self),
            ),
        )

    async def reload_from_store(self, interaction: discord.Interaction, note: str = "") -> None:
        self.memories = await asyncio.to_thread(
            lambda: self.store.list_server(self.guild_id, limit=80, include_pending=True),
        )
        self.can_manage = _is_memory_mod(interaction.user)
        self.screen = "catalog"
        self.selected = None
        self.note = note
        self._build()
        await self.push(interaction)


# Compat : anciens noms de vues de confirmation (plus utilisées en tant que LayoutView).
ConfirmResetMeView = MeMemoryView
ConfirmResetAllView = AllMemoryView
MemoryDetailView = MeMemoryView

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
    return f"**{' '.join(raw.split())}**"


def _task_meta(t: ScheduledTask) -> str:
    ts = int(t.execute_at.timestamp())
    bits: list[str] = [_task_status_label(t)]
    if t.schedule_kind != SCHEDULE_ONCE:
        bits.append(f"{REPEAT_REMINDER} {format_schedule(t)}")
    bits.append(f"<t:{ts}:R>")
    if t.deliver_dm:
        bits.append("MP")
    return " · ".join(bits)


def _task_catalog_items(hub: "TasksView", t: ScheduledTask) -> list[discord.ui.Item]:
    """Consigne en pleine largeur (elle wrappe) ; le crayon reste sur la ligne méta."""
    meta: list[discord.ui.TextDisplay] = [
        discord.ui.TextDisplay(f"-# {_task_meta(t)}"),
    ]
    if t.last_error:
        meta.append(discord.ui.TextDisplay(f"-# {_clip(t.last_error, 80)}"))
    return [
        discord.ui.TextDisplay(_task_heading(t)),
        discord.ui.Section(*meta, accessory=_OpenTaskButton(hub, t)),
    ]


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
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        super().__init__()
        self._hub = hub
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
        err = _task_deny(interaction, self._hub.user_id)
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
        self._hub.store.edit(
            self.task.id, self._hub.user_id,
            instruction=instr or None,
            execute_at=execute_at,
            time_of_day=time_of_day,
        )
        await self._hub.reload(interaction, note="Tâche modifiée.")


class _TaskBackButton(discord.ui.Button):
    def __init__(self, hub: "TasksView"):
        super().__init__(style=discord.ButtonStyle.secondary, label="Retour")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.screen = "catalog"
        self._hub.selected = None
        self._hub.note = ""
        self._hub._build()
        await apply_view(interaction, self._hub)


class _PauseTaskButton(discord.ui.Button):
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        paused = task.status == STATUS_PAUSED
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Reprendre" if paused else "Pause",
        )
        self._hub = hub
        self.task = task
        self.paused = paused

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        if self.paused:
            ok = self._hub.store.resume(self.task.id, self._hub.user_id)
            note = "Tâche reprise." if ok else "Impossible de reprendre."
        else:
            ok = self._hub.store.pause(self.task.id, self._hub.user_id)
            note = "Tâche en pause." if ok else "Impossible de mettre en pause."
        await self._hub.reload(interaction, note=note)


class _SkipTaskButton(discord.ui.Button):
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.secondary, label="Sauter")
        self._hub = hub
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        nxt = self._hub.store.skip_next(self.task.id, self._hub.user_id)
        note = "Prochaine occurrence sautée." if nxt else "Pas de prochaine occurrence."
        await self._hub.reload(interaction, note=note)


class _EditTaskButton(discord.ui.Button):
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.primary, label="Modifier")
        self._hub = hub
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        await interaction.response.send_modal(EditTaskModal(self._hub, self.task))


class _CancelTaskButton(discord.ui.Button):
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        super().__init__(style=discord.ButtonStyle.danger, label="Annuler")
        self._hub = hub
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        ok = self._hub.store.cancel(self.task.id, self._hub.user_id)
        await self._hub.reload(
            interaction,
            note="Tâche annulée." if ok else "Tâche introuvable.",
        )


class _CancelAllTasksButton(discord.ui.Button):
    def __init__(self, hub: "TasksView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Tout annuler")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.screen = "confirm_cancel_all"
        self._hub._build()
        await apply_view(interaction, self._hub)


class _ConfirmCancelAllTasksButton(discord.ui.Button):
    def __init__(self, hub: "TasksView"):
        super().__init__(style=discord.ButtonStyle.danger, label="Confirmer")
        self._hub = hub

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        n = self._hub.store.cancel_all(self._hub.user_id)
        await self._hub.reload(interaction, note=f"{n} tâche(s) annulée(s).")


class _OpenTaskButton(discord.ui.Button):
    def __init__(self, hub: "TasksView", task: ScheduledTask):
        super().__init__(
            style=discord.ButtonStyle.secondary,
            emoji=discord.PartialEmoji.from_str(PENCIL),
        )
        self._hub = hub
        self.task = task

    async def callback(self, interaction: discord.Interaction) -> None:
        err = _task_deny(interaction, self._hub.user_id)
        if err:
            return await interaction.response.send_message(err, ephemeral=True)
        self._hub.selected = self.task
        self._hub.screen = "detail"
        self._hub._build()
        await apply_view(interaction, self._hub)


class TasksView(MariaLayout):
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
        super().__init__(viewer_id=user_id)
        self.store = store
        self.user_id = user_id
        self.tasks = tasks
        self.note = note
        self.page = page
        self.screen = "catalog"
        self.selected: Optional[ScheduledTask] = None
        self._build()

    async def reload(self, interaction: discord.Interaction, note: str = "") -> None:
        self.tasks = self.store.get_user_tasks(self.user_id)
        self.screen = "catalog"
        self.selected = None
        self.note = note
        self._build()
        await self.push(interaction)

    def _build_detail(self) -> None:
        task = self.selected
        if task is None:
            self.screen = "catalog"
            self._build()
            return
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {SMALL_TASK} Tâche"),
            sep_tight(),
            discord.ui.TextDisplay((task.instruction or "").strip() or "-# Sans consigne."),
            discord.ui.TextDisplay(_format_task_body(task)),
        ]
        actions: list[discord.ui.Button] = [_EditTaskButton(self, task)]
        if task.schedule_kind != SCHEDULE_ONCE or task.status == STATUS_PAUSED:
            actions.append(_PauseTaskButton(self, task))
        if task.schedule_kind != SCHEDULE_ONCE:
            actions.append(_SkipTaskButton(self, task))
        actions += [_CancelTaskButton(self, task), _TaskBackButton(self)]
        if self.note:
            body += [sep_tight(), discord.ui.TextDisplay(f"-# {self.note}")]
        self.set_layout(body, discord.ui.ActionRow(*actions[:5]))

    def _build_confirm(self) -> None:
        self.set_layout(
            [
                discord.ui.TextDisplay("## Tâches"),
                discord.ui.TextDisplay(
                    "Annuler **toutes** tes tâches (séries incluses) ? Irréversible."
                ),
            ],
            discord.ui.ActionRow(
                _ConfirmCancelAllTasksButton(self),
                _TaskBackButton(self),
            ),
        )

    def _build_catalog(self) -> None:
        pages = _task_pages(self.tasks)
        self.page = max(0, min(self.page, len(pages) - 1))
        shown = pages[self.page]
        quota_n = sum(1 for t in self.tasks if t.status in (TASK_PENDING, STATUS_PAUSED))
        paused_n = sum(1 for t in self.tasks if t.status == STATUS_PAUSED)
        subtitle = "-# Classé par prochaine exécution"
        if paused_n:
            subtitle += f" · {paused_n} en pause"
        if len(pages) > 1:
            subtitle += f" · page {self.page + 1}/{len(pages)}"
        body: list[discord.ui.Item] = [
            discord.ui.TextDisplay(f"## {SMALL_TASK} Tâches · {quota_n}/{TASK_MAX_PENDING}"),
            discord.ui.TextDisplay(subtitle),
        ]
        if not self.tasks:
            body.append(discord.ui.TextDisplay("-# Aucune tâche en attente."))
        else:
            for t in shown:
                body.append(sep_tight())
                body.extend(_task_catalog_items(self, t))
        extra: list[discord.ui.Button] = []
        if self.tasks:
            extra.append(_CancelAllTasksButton(self))
        max_page = max(0, len(pages) - 1)
        if max_page > 0:
            if self.page > 0:
                extra.append(HubPageButton(self, "page", -1, "Precedent", max_page))
            if self.page < max_page:
                extra.append(HubPageButton(self, "page", 1, "Suivant", max_page))
        rows: list[discord.ui.ActionRow] = []
        if extra:
            rows.append(discord.ui.ActionRow(*extra[:5]))
        if self.note:
            body += [sep_tight(), discord.ui.TextDisplay(f"-# {self.note}")]
        self.set_layout(body, *rows)

    def _build(self) -> None:
        if self.screen == "confirm_cancel_all":
            self._build_confirm()
            return
        if self.screen == "detail":
            self._build_detail()
            return
        self._build_catalog()


TaskDetailView = TasksView
ConfirmCancelAllTasksView = TasksView


class TasksManageButton(
    discord.ui.DynamicItem[discord.ui.Button],
    template=r"maria:tasks:(?P<uid>\d+)",
):
    """Bouton public du widget show_tasks → hub éphémère /taches."""

    def __init__(self, uid: int) -> None:
        super().__init__(
            discord.ui.Button(
                style=discord.ButtonStyle.secondary,
                label="Gérer",
                custom_id=f"maria:tasks:{uid}",
            )
        )
        self.uid = uid

    @classmethod
    async def from_custom_id(
        cls,
        interaction: discord.Interaction,
        item: discord.ui.Button,
        match: re.Match[str],
        /,
    ):
        return cls(int(match["uid"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.uid:
            await interaction.response.send_message("C'est pas tes tâches.", ephemeral=True)
            return
        chat = interaction.client.get_cog("Chat")
        if chat is None or not hasattr(chat, "tasks"):
            await interaction.response.send_message("Indisponible.", ephemeral=True)
            return
        tasks = await asyncio.to_thread(chat.tasks.get_user_tasks, self.uid)
        view = TasksView(chat.tasks, self.uid, tasks)
        await send_ephemeral_menu(interaction, view)
