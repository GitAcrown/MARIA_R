"""Outils LLM de consultation / écriture ciblée de la mémoire long terme."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import discord

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.memory.store import (
    CATEGORY_USER,
    CONFIDENCE_STABLE,
    STATUS_ACTIVE,
    STATUS_PENDING,
    VALID_CATEGORIES,
    Memory,
    MemoryStore,
)
from common.memory.vector import VectorStore
from common.memory.worker import is_near_duplicate, sanitize_memory_content

logger = logging.getLogger("MARIA.Chat.MemoryTools")

_MAX_RESULTS = 20
_CONTENT_MAX = 120


def _canonical_user_content(display_name: str, fact: str) -> str:
    """« Pseudo : fait » — un seul fait, sans id Discord dans le texte."""
    fact = sanitize_memory_content(fact)
    if ":" in fact:
        left, _, right = fact.partition(":")
        if len(left) <= 40 and not re.search(r"\d{17,20}", left):
            fact = right.strip() or fact
    fact = re.sub(r"\s*\(\d{17,20}\)", "", fact).strip()
    name = (display_name or "?").strip() or "?"
    content = f"{name} : {fact}"
    if len(content) > _CONTENT_MAX:
        content = content[: _CONTENT_MAX - 1].rstrip() + "…"
    return content


def _find_near_duplicate(memories: list[Memory], content: str) -> Optional[Memory]:
    for m in memories:
        if is_near_duplicate(content, [m.content]):
            return m
    return None


def build_memory_tools(store: MemoryStore, vectors: VectorStore) -> list[Tool]:
    """Construit search_memory + remember_fact."""

    async def _tool_search_memory(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        guild_id = ctx.trigger_message.guild.id
        args = tc.arguments or {}
        query = (args.get("query") or "").strip()
        category = (args.get("category") or "").strip().lower() or None
        if category and category not in VALID_CATEGORIES:
            category = None
        user_id: Optional[int] = None
        raw_uid = (args.get("user_id") or "").strip()
        if raw_uid:
            try:
                user_id = int(raw_uid)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "user_id invalide"}, datetime.now(timezone.utc),
                )
            category = "user"

        memories = store.search_active(
            guild_id,
            query=query,
            category=category,
            user_id=user_id,
            limit=_MAX_RESULTS,
        )
        items = [
            {
                "id": m.id,
                "category": m.category,
                "user_id": str(m.user_id) if m.user_id else None,
                "content": m.content,
                "confidence": round(m.confidence, 2),
            }
            for m in memories
        ]
        return ToolResponseRecord(tc.id, {
            "count": len(items),
            "memories": items,
            "_llm_summary": (
                f"{len(items)} souvenir(s) trouvé(s)."
                if items
                else "Aucun souvenir correspondant."
            ),
        }, datetime.now(timezone.utc))

    async def _tool_remember_fact(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        """Écrit / met à jour un fait perso après affirmation ou confirmation en tchat."""
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        msg = ctx.trigger_message
        guild = msg.guild
        assert guild is not None
        args = tc.arguments or {}
        fact = (args.get("fact") or "").strip()
        if not fact:
            return ToolResponseRecord(
                tc.id, {"error": "fact vide"}, datetime.now(timezone.utc),
            )

        raw_uid = (args.get("user_id") or "").strip()
        if raw_uid:
            try:
                user_id = int(raw_uid)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "user_id invalide"}, datetime.now(timezone.utc),
                )
        else:
            user_id = msg.author.id

        allowed = {msg.author.id}
        for u in msg.mentions:
            if not u.bot:
                allowed.add(u.id)
        ref = msg.reference.resolved if msg.reference else None
        if isinstance(ref, discord.Message) and ref.author and not ref.author.bot:
            allowed.add(ref.author.id)
        if user_id not in allowed:
            return ToolResponseRecord(
                tc.id,
                {"error": "user_id hors conversation (auteur / mention / reply uniquement)"},
                datetime.now(timezone.utc),
            )

        bot_user = guild.me
        if bot_user is not None and user_id == bot_user.id:
            return ToolResponseRecord(
                tc.id, {"error": "Impossible de retenir un fait sur le bot"}, datetime.now(timezone.utc),
            )

        member = guild.get_member(user_id)
        display_name = (
            member.display_name if member
            else (msg.author.display_name if user_id == msg.author.id else str(user_id))
        )
        stable = bool(args.get("stable", False))
        content = _canonical_user_content(display_name, fact)

        existing = await asyncio.to_thread(
            store.list_for_users, guild.id, {user_id}, limit=40,
        )
        user_mems = [
            m for m in existing
            if m.category == CATEGORY_USER and m.user_id == user_id
        ]

        # Cible explicite (id renvoyé par search_memory) ou quasi-doublon uniquement.
        target: Optional[Memory] = None
        raw_mid = (args.get("memory_id") or "").strip()
        if raw_mid:
            cand = await asyncio.to_thread(store.get, raw_mid)
            if (
                cand is None
                or cand.category != CATEGORY_USER
                or cand.user_id != user_id
                or cand.status not in (STATUS_ACTIVE, STATUS_PENDING)
            ):
                return ToolResponseRecord(
                    tc.id,
                    {"error": "memory_id invalide ou n'appartient pas à ce membre"},
                    datetime.now(timezone.utc),
                )
            target = cand
        else:
            target = _find_near_duplicate(user_mems, content)

        if target is not None:
            was_pending = target.status != STATUS_ACTIVE
            # Remplace le contenu par le fait canonique (1 souvenir = 1 fait).
            mem = await asyncio.to_thread(
                store.update_content, target.id, content, confidence_delta=0.2,
            )
            if mem is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Mise à jour échouée"}, datetime.now(timezone.utc),
                )
            if stable:
                mem = await asyncio.to_thread(store.promote_stable, mem.id, content)
                if mem is None:
                    return ToolResponseRecord(
                        tc.id, {"error": "Promotion stable échouée"}, datetime.now(timezone.utc),
                    )
            if mem.status == STATUS_ACTIVE:
                await asyncio.to_thread(
                    vectors.upsert,
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )
            logger.info("remember_fact update %s → %s", mem.id[:8], content[:60])
            return ToolResponseRecord(tc.id, {
                "ok": True,
                "action": "updated",
                "memory_id": mem.id,
                "content": mem.content,
                "was_pending": was_pending,
                "_llm_summary": f"Souvenir mis à jour : {content}",
            }, datetime.now(timezone.utc))

        conf = CONFIDENCE_STABLE if stable else 0.55
        mem = await asyncio.to_thread(
            store.create,
            category=CATEGORY_USER,
            guild_id=guild.id,
            content=content,
            user_id=user_id,
            confidence=conf,
            status=STATUS_ACTIVE,
        )
        await asyncio.to_thread(
            vectors.upsert,
            mem.id, mem.content,
            category=mem.category, guild_id=mem.guild_id,
            user_id=mem.user_id, confidence=mem.confidence,
        )
        logger.info("remember_fact create %s → %s", mem.id[:8], content[:60])
        return ToolResponseRecord(tc.id, {
            "ok": True,
            "action": "created",
            "memory_id": mem.id,
            "content": mem.content,
            "_llm_summary": f"Souvenir retenu : {content}",
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="search_memory",
            description=(
                "Mémoire long terme (lecture seule). Les PROFILS du prompt couvrent déjà "
                "auteur + mentions — ne pas rappeler pour ça. "
                "Pour : membre/sujet ABSENT des profils, énumérer, filtrer par mot-clé. "
                "Renvoie aussi l'id de chaque souvenir (utile pour remember_fact). "
                "Pas d'écriture — pour écrire utilise remember_fact."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Mot-clé dans le contenu (ex: anniversaire, café). Vide = tout.",
                },
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Filtrer par catégorie. Omettre pour user+server+event.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Id Discord du membre (mémoires user uniquement).",
                },
            },
            optional_props=["query", "category", "user_id"],
            function=_tool_search_memory,
        ),
        Tool(
            name="remember_fact",
            description=(
                "Écrit ou met à jour TOUT DE SUITE un fait perso (1 fait = 1 souvenir). "
                "À appeler seulement quand le fait est affirmé clairement OU confirmé "
                "après une question légère de ta part — jamais sur une simple déduction non validée. "
                "Avant de dire « noté » / « j'ai retenu » : appelle cet outil. "
                "fact = fait seul (« anniversaire le 22 juillet 1999 »). "
                "stable=true pour immuable (anniv / date de naissance). "
                "memory_id optionnel = id d'un souvenir à remplacer (via search_memory) ; "
                "sinon fusion seulement si quasi-identique, sinon create."
            ),
            properties={
                "fact": {
                    "type": "string",
                    "description": (
                        "Un seul fait, sans préfixe pseudo "
                        "(ex: « anniversaire le 22 juillet 1999 »)."
                    ),
                },
                "user_id": {
                    "type": "string",
                    "description": "Id Discord concerné (défaut = auteur du FOCUS).",
                },
                "stable": {
                    "type": "boolean",
                    "description": "true si fait immuable (anniversaire, date de naissance).",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Id d'un souvenir existant à mettre à jour (optionnel).",
                },
            },
            optional_props=["user_id", "stable", "memory_id"],
            function=_tool_remember_fact,
        ),
    ]
