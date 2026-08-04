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
    CATEGORY_SELF,
    CATEGORY_USER,
    CONFIDENCE_DIRECT,
    CONFIDENCE_STABLE,
    STATUS_ACTIVE,
    STATUS_PENDING,
    VALID_CATEGORIES,
    Memory,
    MemoryStore,
)
from common.memory.vector import VectorStore
from common.memory.worker import (
    is_near_duplicate,
    is_too_vague,
    normalize_self_memory,
    sanitize_memory_content,
)
from discord.ext import commands

from cogs.chat.views import _is_memory_mod

logger = logging.getLogger("MARIA.Chat.MemoryTools")

_MAX_RESULTS = 20
_CONTENT_MAX = 180


def _canonical_user_content(display_name: str, fact: str) -> str:
    """« Pseudo : fait » — un seul fait précis, sans id Discord dans le texte."""
    fact = sanitize_memory_content(fact)
    if ":" in fact:
        left, _, right = fact.partition(":")
        if len(left) <= 40 and not re.search(r"\d{17,20}", left):
            fact = right.strip() or fact
    fact = re.sub(r"\s*\(\d{17,20}\)", "", fact).strip()
    name = (display_name or "?").strip() or "?"
    return f"{name} : {fact}"


def _find_near_duplicate(memories: list[Memory], content: str) -> Optional[Memory]:
    for m in memories:
        if is_near_duplicate(content, [m.content]):
            return m
    return None


def _semantic_search(
    store: MemoryStore,
    vectors: VectorStore,
    guild_id: int,
    query: str,
    *,
    category: Optional[str],
    user_id: Optional[int],
    limit: int,
) -> list[Memory]:
    """Recherche par similarité (Chroma), utile quand le mot-clé exact n'existe pas
    dans le texte (paraphrase, synonyme). Résultats triés par proximité."""
    results = vectors.query(query, guild_id=guild_id, user_id=user_id, n=limit * 2)
    out: list[Memory] = []
    seen: set[str] = set()
    for r in results:
        mid = r.get("id")
        if not mid or mid in seen:
            continue
        meta = r.get("metadata") or {}
        if category and meta.get("category") != category:
            continue
        if user_id is not None and meta.get("category") == CATEGORY_USER and meta.get("user_id") != user_id:
            continue
        mem = store.get(mid)
        if mem is None or mem.status != STATUS_ACTIVE:
            continue
        seen.add(mid)
        out.append(mem)
        if len(out) >= limit:
            break
    return out


def build_memory_tools(
    store: MemoryStore,
    vectors: VectorStore,
    *,
    bot: Optional[commands.Bot] = None,
) -> list[Tool]:
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
            if category != CATEGORY_SELF:
                category = "user"

        semantic = bool(args.get("semantic", False)) and bool(query) and vectors.available
        if semantic:
            memories = await asyncio.to_thread(
                _semantic_search, store, vectors, guild_id, query,
                category=category, user_id=user_id, limit=_MAX_RESULTS,
            )
        else:
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
        """Écrit / met à jour un fait perso (membre) ou un goût MARIA (about_self)."""
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

        about_self = bool(args.get("about_self", False))
        bot_user = guild.me
        bot_name = (
            (bot_user.display_name if bot_user else None)
            or getattr(guild.me, "name", None)
            or "MARIA"
        )

        if about_self:
            # own = tu te forges un goût ; owner = le créateur force / corrige.
            source = str(args.get("self_source") or "own").strip().lower()
            if source not in ("own", "owner"):
                source = "own"
            if bot is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Contrôle owner indisponible"}, datetime.now(timezone.utc),
                )
            try:
                is_owner = await bot.is_owner(msg.author)
            except Exception:
                is_owner = False
            if source == "owner" and not is_owner:
                return ToolResponseRecord(
                    tc.id,
                    {
                        "error": (
                            "Seul le créateur (owner) peut forcer un goût. "
                            "Si c'est ton avis à toi → self_source=own. "
                            "Si un non-owner te dicte un goût → refuse, n'appelle pas l'outil."
                        ),
                        "refused": True,
                    },
                    datetime.now(timezone.utc),
                )
            if bot_user is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Bot introuvable sur ce serveur"}, datetime.now(timezone.utc),
                )
            content = normalize_self_memory(fact, bot_name=bot_name)
            if len(content) > _CONTENT_MAX or is_too_vague(content):
                return ToolResponseRecord(
                    tc.id,
                    {
                        "error": (
                            "Goût trop vague ou trop long — reformule avec un détail "
                            "(ex. « préfère le café noir », « déteste le foot »)."
                        ),
                    },
                    datetime.now(timezone.utc),
                )
            self_mems = await asyncio.to_thread(store.list_self, limit=40)
            target: Optional[Memory] = None
            raw_mid = (args.get("memory_id") or "").strip()
            if raw_mid:
                cand = await asyncio.to_thread(store.get, raw_mid)
                if (
                    cand is None
                    or cand.category != CATEGORY_SELF
                    or cand.status not in (STATUS_ACTIVE, STATUS_PENDING)
                ):
                    return ToolResponseRecord(
                        tc.id,
                        {"error": "memory_id invalide (pas un goût self)"},
                        datetime.now(timezone.utc),
                    )
                target = cand
            else:
                target = _find_near_duplicate(self_mems, content)

            if target is not None:
                mem = await asyncio.to_thread(
                    store.update_content, target.id, content, confidence_delta=0.2,
                )
                if mem is None:
                    return ToolResponseRecord(
                        tc.id, {"error": "Mise à jour échouée"}, datetime.now(timezone.utc),
                    )
                if mem.status == STATUS_ACTIVE:
                    await asyncio.to_thread(
                        vectors.upsert,
                        mem.id, mem.content,
                        category=mem.category, guild_id=mem.guild_id,
                        user_id=mem.user_id, confidence=mem.confidence,
                    )
                logger.info(
                    "remember_fact self update (%s) %s → %s",
                    source, mem.id[:8], content[:60],
                )
                return ToolResponseRecord(tc.id, {
                    "ok": True,
                    "action": "updated",
                    "about_self": True,
                    "self_source": source,
                    "memory_id": mem.id,
                    "content": mem.content,
                    "_llm_summary": f"Goût retenu (toi) : {content}",
                }, datetime.now(timezone.utc))

            mem = await asyncio.to_thread(
                store.create,
                category=CATEGORY_SELF,
                guild_id=guild.id,
                content=content,
                user_id=bot_user.id,
                confidence=CONFIDENCE_DIRECT,
                status=STATUS_ACTIVE,
            )
            await asyncio.to_thread(
                vectors.upsert,
                mem.id, mem.content,
                category=mem.category, guild_id=mem.guild_id,
                user_id=mem.user_id, confidence=mem.confidence,
            )
            logger.info(
                "remember_fact self create (%s) %s → %s",
                source, mem.id[:8], content[:60],
            )
            return ToolResponseRecord(tc.id, {
                "ok": True,
                "action": "created",
                "about_self": True,
                "self_source": source,
                "memory_id": mem.id,
                "content": mem.content,
                "_llm_summary": f"Goût retenu (toi) : {content}",
            }, datetime.now(timezone.utc))

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

        if bot_user is not None and user_id == bot_user.id:
            return ToolResponseRecord(
                tc.id,
                {
                    "error": (
                        "Pour un goût / fait sur toi-même, rappelle avec about_self=true "
                        "(pas user_id du bot)."
                    ),
                },
                datetime.now(timezone.utc),
            )

        member = guild.get_member(user_id)
        display_name = (
            member.display_name if member
            else (msg.author.display_name if user_id == msg.author.id else str(user_id))
        )
        stable = bool(args.get("stable", False))
        content = _canonical_user_content(display_name, fact)
        if len(content) > _CONTENT_MAX or is_too_vague(content):
            return ToolResponseRecord(
                tc.id,
                {
                    "error": (
                        "Fait trop vague ou trop long — reformule avec un détail concret "
                        "(date, lieu, titre…) ou ignore."
                    ),
                },
                datetime.now(timezone.utc),
            )

        existing = await asyncio.to_thread(
            store.list_for_users, guild.id, {user_id}, limit=40,
        )
        user_mems = [
            m for m in existing
            if m.category == CATEGORY_USER and m.user_id == user_id
        ]

        # Cible explicite (id renvoyé par search_memory) ou quasi-doublon uniquement.
        target = None
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

    async def _tool_forget_fact(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        """Supprime (archive) un souvenir existant — sans remplacement. Pour une correction
        avec un fait de rechange, préférer remember_fact(memory_id=...)."""
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        msg = ctx.trigger_message
        guild = msg.guild
        assert guild is not None
        args = tc.arguments or {}
        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id:
            return ToolResponseRecord(
                tc.id, {"error": "memory_id vide (cherche avec search_memory d'abord)"},
                datetime.now(timezone.utc),
            )

        mem = await asyncio.to_thread(store.get, memory_id)
        if mem is None or mem.guild_id != guild.id or mem.status not in (STATUS_ACTIVE, STATUS_PENDING):
            return ToolResponseRecord(
                tc.id, {"error": "memory_id introuvable ou déjà oublié"}, datetime.now(timezone.utc),
            )

        if mem.category == CATEGORY_SELF:
            if bot is None:
                return ToolResponseRecord(
                    tc.id, {"error": "Contrôle owner indisponible"}, datetime.now(timezone.utc),
                )
            try:
                is_owner = await bot.is_owner(msg.author)
            except Exception:
                is_owner = False
            if not is_owner:
                return ToolResponseRecord(
                    tc.id,
                    {
                        "error": "Seul le créateur peut te faire oublier un goût.",
                        "refused": True,
                    },
                    datetime.now(timezone.utc),
                )
            had_vector = mem.status == STATUS_ACTIVE
            await asyncio.to_thread(store.archive, mem.id)
            if had_vector:
                await asyncio.to_thread(vectors.delete, mem.id)
            logger.info("forget_fact self %s", mem.id[:8])
            return ToolResponseRecord(tc.id, {
                "ok": True,
                "memory_id": mem.id,
                "_llm_summary": f"Goût oublié : {mem.content}",
            }, datetime.now(timezone.utc))

        if mem.category == CATEGORY_USER:
            allowed = {msg.author.id}
            for u in msg.mentions:
                if not u.bot:
                    allowed.add(u.id)
            ref = msg.reference.resolved if msg.reference else None
            if isinstance(ref, discord.Message) and ref.author and not ref.author.bot:
                allowed.add(ref.author.id)
            if mem.user_id not in allowed:
                return ToolResponseRecord(
                    tc.id,
                    {"error": "Souvenir hors conversation (auteur / mention / reply uniquement)"},
                    datetime.now(timezone.utc),
                )
            ok, chroma = await asyncio.to_thread(store.forget_user_memory, mem.id, mem.user_id)
            if not ok:
                return ToolResponseRecord(
                    tc.id, {"error": "Oubli échoué"}, datetime.now(timezone.utc),
                )
            if chroma:
                await asyncio.to_thread(vectors.delete, chroma)
            logger.info("forget_fact user %s", mem.id[:8])
            return ToolResponseRecord(tc.id, {
                "ok": True,
                "memory_id": mem.id,
                "_llm_summary": f"Souvenir oublié : {mem.content}",
            }, datetime.now(timezone.utc))

        # server / event
        if not _is_memory_mod(msg.author):
            return ToolResponseRecord(
                tc.id,
                {"error": "Réservé aux modos pour un souvenir collectif.", "refused": True},
                datetime.now(timezone.utc),
            )
        ok, chroma = await asyncio.to_thread(store.forget_server_memory, mem.id, guild.id)
        if not ok:
            return ToolResponseRecord(
                tc.id, {"error": "Oubli échoué"}, datetime.now(timezone.utc),
            )
        if chroma:
            await asyncio.to_thread(vectors.delete, chroma)
        logger.info("forget_fact server %s", mem.id[:8])
        return ToolResponseRecord(tc.id, {
            "ok": True,
            "memory_id": mem.id,
            "_llm_summary": f"Souvenir oublié : {mem.content}",
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="search_memory",
            description=(
                "Mémoire long terme (lecture seule). Les PROFILS + TES GOÛTS du prompt "
                "couvrent déjà auteur/mentions et toi — ne pas rappeler pour ça. "
                "Pour : membre/sujet ABSENT, énumérer, category=self pour tes goûts. "
                "Renvoie aussi l'id (utile pour remember_fact). Pas d'écriture."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Mot-clé dans le contenu (ex: anniversaire, café). Vide = tout.",
                },
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Filtrer (self = tes goûts). Omettre = user+self+server+event.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Id Discord du membre (mémoires user uniquement).",
                },
                "semantic": {
                    "type": "boolean",
                    "description": (
                        "true = recherche par sens (paraphrase/synonyme) au lieu du mot-clé exact. "
                        "À utiliser si une recherche par mot-clé direct semble avoir raté un souvenir "
                        "reformulé différemment. Nécessite query non vide."
                    ),
                },
            },
            optional_props=["query", "category", "user_id", "semantic"],
            function=_tool_search_memory,
        ),
        Tool(
            name="remember_fact",
            description=(
                "Écrit ou met à jour TOUT DE SUITE un fait précis (1 fait = 1 souvenir). "
                "Membre : affirmé/confirmé clairement — jamais déduction seule. "
                "Toi (goûts) : about_self=true + self_source : "
                "own = tu te forges un avis perso net que tu veux garder ; "
                "owner = le créateur te force/corrige un goût (refuse si pas owner). "
                "Un non-owner qui te dicte un goût → refuse, n'appelle pas l'outil. "
                "Avant « noté » / « j'ai retenu » : appelle cet outil. "
                "stable=true pour anniv (membres seulement). "
                "memory_id optionnel pour remplacer un souvenir."
            ),
            properties={
                "fact": {
                    "type": "string",
                    "description": (
                        "Un seul fait PRÉCIS, sans préfixe pseudo "
                        "(membre : « anniversaire le 22 juillet 1999 » ; "
                        "toi : « préfère le café noir », « déteste le foot »). "
                        "Date relative (« demain », « la semaine prochaine », « dans 3 jours ») "
                        "→ résous-la en date absolue avec la date du jour (DATE/HEURE du prompt) "
                        "avant d'écrire le fait : sinon « demain » reste vrai pour toujours."
                    ),
                },
                "user_id": {
                    "type": "string",
                    "description": "Id Discord concerné (défaut = auteur). Ignoré si about_self.",
                },
                "about_self": {
                    "type": "boolean",
                    "description": "true = goût / fait sur TOI (MARIA), pas sur un membre.",
                },
                "self_source": {
                    "type": "string",
                    "enum": ["own", "owner"],
                    "description": (
                        "own = ton avis à toi ; owner = forcé par le créateur "
                        "(nécessite que l'auteur soit owner)."
                    ),
                },
                "stable": {
                    "type": "boolean",
                    "description": "true si fait immuable membre (anniversaire, date de naissance).",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Id d'un souvenir existant à mettre à jour (optionnel).",
                },
            },
            optional_props=["user_id", "about_self", "self_source", "stable", "memory_id"],
            function=_tool_remember_fact,
        ),
        Tool(
            name="forget_fact",
            description=(
                "Supprime définitivement un souvenir (pas de remplacement). "
                "À utiliser quand on te dit qu'un fait retenu est FAUX et qu'il n'y a rien "
                "de valide à mettre à la place (sinon préfère remember_fact avec memory_id "
                "pour corriger directement). Cherche l'id via search_memory avant d'appeler. "
                "Membre : réservé à l'auteur / une mention / la personne citée en reply. "
                "Souvenir collectif (server/event) : réservé aux modos. "
                "Goût sur toi : réservé au créateur."
            ),
            properties={
                "memory_id": {
                    "type": "string",
                    "description": "Id du souvenir à oublier (renvoyé par search_memory).",
                },
            },
            optional_props=[],
            function=_tool_forget_fact,
        ),
    ]
