"""Worker mémoire — buffer hybride, flush d'extraction, decay quotidien."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

from discord.ext import tasks

from common.memory.agent import extract_memories, parse_user_id
from common.memory.store import (
    CATEGORY_EVENT,
    CATEGORY_SERVER,
    CONFIDENCE_COLLECTIVE,
    CONFIDENCE_DIRECT,
    CONFIDENCE_PENDING,
    CONFIDENCE_STABLE,
    CONFIDENCE_UPDATE_DELTA,
    STATUS_ACTIVE,
    STATUS_PENDING,
    VALID_CATEGORIES,
    MemoryStore,
)
from common.memory.vector import VectorStore

logger = logging.getLogger("MARIA.Memory.Worker")

_META_PERSON_RE = re.compile(
    r"\b(?:le\s+membre|la\s+membre|l['']utilisateur(?:\s*trice)?|la\s+personne|"
    r"le\s+user|l['']user)\s+",
    re.IGNORECASE,
)
_MEMORY_CONTENT_MAX = 180
# Discord snowflakes typiques (17–20 chiffres) dans « Name (id) » / content.
_DISCORD_ID_RE = re.compile(r"(?<!\d)(\d{17,20})(?!\d)")
_SNOWFLAKE_PARENS_RE = re.compile(r"\s*\((\d{17,20})\)")
_MENTION_NAME_ID_RE = re.compile(r"@?([^\s@<>()]+)\((\d{17,20})\)")
_NAME_ID_TAIL_RE = re.compile(r"^(.*?)\s*\((\d{17,20})\)\s*$")
# Faits trop flous pour être utiles relu plus tard → rejet worker.
_VAGUE_FACT_RE = re.compile(
    r"(?:"
    r"\.\.\.|…"
    r"|quelque\s+part|un\s+truc|des\s+trucs|je\s+sais\s+pas\s+o[uù]"
    r"|\b(souvent|parfois|un\s+jour|l['']autre\s+jour)\s*$"
    r"|\b(aime|adore|déteste|détest|kiffe)\s+(les?\s+)?(jeux?|films?|séries?|musiques?)?\s*$"
    r"|\b(joue|regarde|écoute)\s+(aux?\s+|des?\s+|à\s+la\s+)?(jeux?|films?|séries?)?\s*$"
    r"|\b(a|ont)\s+(un|des)\s+go[uû]ts?\b"
    r"|\best\s+(cool|sympa|dr[oô]le|nice|bg)\s*$"
    r"|\bgag\s+(du\s+)?serveur\b|\binside\s*joke\b\s*$"
    r"|\banniversaire\s+le\s*$"
    r"|a\s+demandé\s+la\s+m[eé]t[eé]o\s*$"
    r")",
    re.IGNORECASE,
)


def sanitize_memory_content(text: str) -> str:
    """Nettoie le texte. Ne tronque pas avec « … » (ça crée des souvenirs inutiles)."""
    content = (text or "").strip()
    if not content:
        return content
    content = _META_PERSON_RE.sub("", content)
    content = re.sub(r"\s{2,}", " ", content).strip(" .;")
    if content:
        content = content[0].upper() + content[1:]
    return content


def _clip(content: str) -> str:
    """Identité : la troncature silencieuse est interdite (précision ou rejet)."""
    return content


def is_too_vague(content: str) -> bool:
    """True si le fait n'a pas assez de détail pour être réutilisable."""
    fact = _fact_part(content)
    if not fact or len(fact) < 8:
        return True
    words = [w for w in re.split(r"\s+", fact) if w]
    if len(words) < 3:
        return True
    if fact.endswith(("…", "...")) or "…" in fact or "..." in fact:
        return True
    if _VAGUE_FACT_RE.search(fact):
        return True
    return False


def _collect_batch_user_ids(
    batch: list[BufferedMessage],
    prior: list[BufferedMessage],
) -> set[int]:
    """Auteurs, reply targets, et ids cités dans le texte du lot/prior."""
    ids: set[int] = set()
    for m in list(batch) + list(prior):
        ids.add(m.author_id)
        if m.reply_to_id is not None:
            ids.add(m.reply_to_id)
        for raw in _DISCORD_ID_RE.findall(m.content or ""):
            try:
                ids.add(int(raw))
            except ValueError:
                pass
        if m.reply_to_content:
            for raw in _DISCORD_ID_RE.findall(m.reply_to_content):
                try:
                    ids.add(int(raw))
                except ValueError:
                    pass
    return ids


def _collect_name_by_id(
    batch: list[BufferedMessage],
    prior: list[BufferedMessage],
) -> dict[int, str]:
    """Pseudo vu dans le lot pour chaque id (le lot récent écrase le prior)."""
    names: dict[int, str] = {}
    for m in list(prior) + list(batch):
        if m.author_name:
            names[m.author_id] = m.author_name
        if m.reply_to_id is not None and m.reply_to_name:
            names[m.reply_to_id] = m.reply_to_name
        for match in _MENTION_NAME_ID_RE.finditer(m.content or ""):
            try:
                names[int(match.group(2))] = match.group(1)
            except ValueError:
                pass
    return names


def _id_by_name_lower(name_by_id: dict[int, str]) -> dict[str, int]:
    """Inverse pseudo → id ; ignore les pseudos ambigus (plusieurs ids)."""
    out: dict[str, int] = {}
    ambiguous: set[str] = set()
    for uid, name in name_by_id.items():
        key = (name or "").casefold().strip()
        if not key or key in ambiguous:
            continue
        if key in out and out[key] != uid:
            ambiguous.add(key)
            out.pop(key, None)
            continue
        out[key] = uid
    return out


def _ids_in_text(text: str) -> set[int]:
    out: set[int] = set()
    for raw in _DISCORD_ID_RE.findall(text or ""):
        try:
            out.add(int(raw))
        except ValueError:
            pass
    return out


def _fact_part(content: str) -> str:
    """Retire le préfixe pseudo pour comparer uniquement le fait."""
    if ":" in content:
        return content.split(":", 1)[1].strip().lower()
    return content.strip().lower()


def is_near_duplicate(new_content: str, existing_contents: list[str]) -> bool:
    """Détecte un fait quasi identique déjà stocké (même sujet), évite les doublons."""
    new_fact = _fact_part(new_content)
    if not new_fact:
        return False
    for other in existing_contents:
        other_fact = _fact_part(other)
        if not other_fact:
            continue
        if new_fact == other_fact:
            return True
        ratio = SequenceMatcher(None, new_fact, other_fact).ratio()
        if ratio >= 0.82:
            return True
    return False


def _split_name_id(part: str) -> tuple[str, Optional[int]]:
    """« Alice (123) » → (Alice, 123) ; « Alice » → (Alice, None)."""
    part = (part or "").strip()
    m = _NAME_ID_TAIL_RE.match(part)
    if m:
        return (m.group(1) or "").strip(), int(m.group(2))
    return part, None


def normalize_user_memory(
    content: str,
    *,
    user_id: Optional[int],
    name_by_id: dict[int, str],
    lock_user_id: bool = False,
) -> tuple[Optional[int], str]:
    """Aligne user_id + content : pseudo canonique, ids seulement pour les liens ↔.

    - Fait simple → « Pseudo : fait » (jamais d'id dans le texte).
    - Lien → « A (id) ↔ B (id) : fait » avec pseudos du lot.
    - Si le préfixe nomme clairement un autre membre du lot → corrige user_id
      (sauf si lock_user_id, ex. update d'un souvenir déjà ancré).
    """
    content = sanitize_memory_content(content)
    if not content:
        return user_id, ""

    id_by_name = _id_by_name_lower(name_by_id)

    if "↔" in content:
        left, _, right = content.partition("↔")
        if ":" not in right:
            return None, ""
        right_who, _, fact = right.partition(":")
        fact = _SNOWFLAKE_PARENS_RE.sub("", fact.strip()).strip()
        if not fact:
            return None, ""
        name_a, id_a = _split_name_id(left)
        name_b, id_b = _split_name_id(right_who)

        def _resolve_side(name: str, sid: Optional[int]) -> Optional[int]:
            if sid is not None and sid in name_by_id:
                return sid
            return id_by_name.get(name.casefold()) if name else None

        id_a = _resolve_side(name_a, id_a)
        id_b = _resolve_side(name_b, id_b)
        if id_a is None or id_b is None:
            return None, ""
        label_a = name_by_id.get(id_a) or name_a
        label_b = name_by_id.get(id_b) or name_b
        if lock_user_id and user_id is not None:
            if user_id not in (id_a, id_b):
                return None, ""
            primary = user_id
        else:
            primary = user_id if user_id in (id_a, id_b) else id_a
        return primary, _clip(f"{label_a} ({id_a}) ↔ {label_b} ({id_b}) : {fact}")

    # Fait perso simple.
    if ":" in content:
        left, _, fact = content.partition(":")
        raw_name, prefix_id = _split_name_id(left)
    else:
        raw_name = ""
        prefix_id = None
        fact = content

    fact = _SNOWFLAKE_PARENS_RE.sub("", fact.strip()).strip()
    if not fact:
        return None, ""

    resolved = user_id
    if not lock_user_id:
        if prefix_id is not None and prefix_id in name_by_id:
            resolved = prefix_id
        if raw_name:
            named = id_by_name.get(raw_name.casefold())
            if named is not None:
                # Le pseudo du content prime s'il désigne un membre du lot.
                resolved = named

    if resolved is None:
        return None, ""

    canonical = name_by_id.get(resolved) or raw_name or "?"
    return resolved, _clip(f"{canonical} : {fact}")


@dataclass
class BufferedMessage:
    author_id: int
    author_name: str
    content: str
    ts: datetime
    reply_to_id: Optional[int] = None
    reply_to_name: Optional[str] = None
    reply_to_content: Optional[str] = None
    reply_is_bot: bool = False
    # True si l'humain s'adresse à MARIA (mention / reply bot / déclenche une réponse).
    addressed_to_bot: bool = False

    def format_line(self) -> str:
        """Ligne lisible pour l'agent d'extraction, avec contexte de reply si présent."""
        stamp = self.ts.strftime("%H:%M")
        head = f"[{stamp}] {self.author_name} ({self.author_id})"
        if self.addressed_to_bot:
            head += " [→ MARIA]"
        # reply_to_id peut être None quand on répond au bot — il faut quand même le marquer.
        if self.reply_to_name or self.reply_to_id is not None:
            if self.reply_is_bot or self.reply_to_id is None:
                target = self.reply_to_name or "le bot"
            else:
                target = f"{self.reply_to_name or '?'} ({self.reply_to_id})"
            snippet = (self.reply_to_content or "").replace("\n", " ").strip()
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            if snippet:
                head += f' [répond à {target}: "{snippet}"]'
            else:
                head += f" [répond à {target}]"
        return f"{head}: {self.content}"


@dataclass
class ChannelBuffer:
    messages: list[BufferedMessage] = field(default_factory=list)
    # Queue du lot précédent, renvoyée en contexte au prochain flush (sans re-create).
    overlap: list[BufferedMessage] = field(default_factory=list)
    flushing: bool = False


class MemoryWorker:
    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorStore,
        llm_client: Any,
        *,
        model: str,
        flush_messages: int = 40,
        flush_minutes: int = 30,
        buffer_cap: int = 80,
        existing_limit: int = 25,
        max_actions: int = 6,
        batch_overlap: int = 8,
        direct_flush_messages: int = 8,
        bot_user_id: Optional[int] = None,
        bot_name: str = "MARIA",
    ) -> None:
        self.store = store
        self.vectors = vectors
        self.llm_client = llm_client
        self.model = model
        self.flush_messages = flush_messages
        self.flush_minutes = flush_minutes
        self.direct_flush_messages = max(3, min(direct_flush_messages, flush_messages))
        self.buffer_cap = max(buffer_cap, flush_messages)
        self.existing_limit = existing_limit
        self.max_actions = max_actions
        self.batch_overlap = max(0, min(batch_overlap, max(0, flush_messages // 2)))
        self.bot_user_id = bot_user_id
        self.bot_name = bot_name or "MARIA"
        self._buffers: dict[tuple[int, int], ChannelBuffer] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if not self._decay_loop.is_running():
            self._decay_loop.start()
        if not self._timeout_loop.is_running():
            self._timeout_loop.start()

    async def stop(self) -> None:
        if self._decay_loop.is_running():
            self._decay_loop.cancel()
        if self._timeout_loop.is_running():
            self._timeout_loop.cancel()

    def ingest(
        self,
        *,
        guild_id: int,
        channel_id: int,
        author_id: int,
        author_name: str,
        content: str,
        reply_to_id: Optional[int] = None,
        reply_to_name: Optional[str] = None,
        reply_to_content: Optional[str] = None,
        reply_is_bot: bool = False,
        addressed_to_bot: bool = False,
    ) -> None:
        text = (content or "").strip()
        if not text:
            return
        key = (guild_id, channel_id)
        buf = self._buffers.setdefault(key, ChannelBuffer())
        addressed = bool(addressed_to_bot or reply_is_bot)
        buf.messages.append(
            BufferedMessage(
                author_id=author_id,
                author_name=author_name,
                content=text[:700],
                ts=datetime.now(timezone.utc),
                reply_to_id=reply_to_id,
                reply_to_name=reply_to_name,
                reply_to_content=(reply_to_content or "")[:240] or None,
                reply_is_bot=reply_is_bot,
                addressed_to_bot=addressed,
            )
        )
        if len(buf.messages) > self.buffer_cap:
            buf.messages = buf.messages[-self.buffer_cap :]

        should_flush = len(buf.messages) >= self.flush_messages
        if not should_flush and addressed:
            # Dialogue avec MARIA : flush plus tôt pour ancrer les faits tout de suite.
            should_flush = len(buf.messages) >= self.direct_flush_messages
        if should_flush and not buf.flushing:
            asyncio.create_task(self._flush_key(key))

    def _should_flush_timeout(self, buf: ChannelBuffer) -> bool:
        if not buf.messages or buf.flushing:
            return False
        oldest = buf.messages[0].ts
        return datetime.now(timezone.utc) - oldest >= timedelta(minutes=self.flush_minutes)

    async def _flush_key(self, key: tuple[int, int]) -> None:
        async with self._lock:
            buf = self._buffers.get(key)
            if buf is None or not buf.messages or buf.flushing:
                return
            buf.flushing = True
            batch = list(buf.messages)
            prior = list(buf.overlap)
            buf.messages.clear()

        guild_id, channel_id = key
        try:
            await self._process_batch(guild_id, batch, prior=prior)
        except Exception as e:
            logger.warning("Flush mémoire échoué (guild=%s ch=%s): %s", guild_id, channel_id, e)
        finally:
            async with self._lock:
                buf = self._buffers.get(key)
                if buf is not None:
                    # Garde la fin du lot pour le prochain flush (liaison inter-lots).
                    if self.batch_overlap > 0 and batch:
                        buf.overlap = batch[-self.batch_overlap :]
                    buf.flushing = False

    async def _process_batch(
        self,
        guild_id: int,
        batch: list[BufferedMessage],
        *,
        prior: Optional[list[BufferedMessage]] = None,
    ) -> None:
        if not batch:
            return
        prior = prior or []
        allowed_ids = _collect_batch_user_ids(batch, prior)
        name_by_id = _collect_name_by_id(batch, prior)
        if self.bot_user_id is not None:
            allowed_ids.discard(self.bot_user_id)
            name_by_id.pop(self.bot_user_id, None)
        existing = await asyncio.to_thread(
            self.store.list_for_users, guild_id, allowed_ids, limit=self.existing_limit,
        )
        batch_text = "\n".join(m.format_line() for m in batch)
        prior_text = "\n".join(m.format_line() for m in prior) if prior else ""
        direct_user_ids = {
            m.author_id for m in list(batch) + list(prior) if m.addressed_to_bot
        }
        actions = await extract_memories(
            self.llm_client,
            model=self.model,
            batch_text=batch_text,
            existing=existing,
            bot_name=self.bot_name,
            max_actions=self.max_actions,
            prior_text=prior_text,
        )
        if not actions:
            logger.info(
                "Flush mémoire : 0 action (guild=%s, msgs=%d, direct=%d)",
                guild_id, len(batch), len(direct_user_ids),
            )
            return
        existing_by_user: dict[Optional[int], list[str]] = {}
        for m in existing:
            existing_by_user.setdefault(m.user_id, []).append(m.content)
        for action in actions[: self.max_actions]:
            await self._apply_action(
                guild_id, action,
                allowed_ids=allowed_ids, name_by_id=name_by_id,
                existing_by_user=existing_by_user,
                direct_user_ids=direct_user_ids,
            )
        logger.info(
            "Flush mémoire : %d action(s) LLM (guild=%s, msgs=%d)",
            len(actions), guild_id, len(batch),
        )

    async def _apply_action(
        self,
        guild_id: int,
        action: dict,
        *,
        allowed_ids: Optional[set[int]] = None,
        name_by_id: Optional[dict[int, str]] = None,
        existing_by_user: Optional[dict[Optional[int], list[str]]] = None,
        direct_user_ids: Optional[set[int]] = None,
    ) -> None:
        kind = (action.get("action") or "").strip()
        category = (action.get("category") or "user").strip()
        if category not in VALID_CATEGORIES:
            category = "user"
        target_id = action.get("target_id") or None
        if isinstance(target_id, str) and target_id.lower() in ("null", ""):
            target_id = None
        user_id = parse_user_id(action.get("user_id"))
        raw_content = action.get("content") or ""
        names = name_by_id or {}

        # Jamais de souvenir perso (ni update) attribué au bot Discord.
        if self.bot_user_id is not None and user_id == self.bot_user_id:
            return

        if category == "user":
            user_id, content = normalize_user_memory(
                raw_content, user_id=user_id, name_by_id=names,
            )
        else:
            content = sanitize_memory_content(raw_content)
            # Collectif : pas d'ids Discord dans le texte.
            content = _SNOWFLAKE_PARENS_RE.sub("", content).strip()
            content = re.sub(r"\s{2,}", " ", content).strip(" .;")
            if category == "server":
                user_id = None

        if not content:
            return
        if len(content) > _MEMORY_CONTENT_MAX:
            logger.debug("Mémoire rejetée (trop longue): %s", content[:60])
            return
        if is_too_vague(content):
            logger.debug("Mémoire rejetée (trop vague): %s", content[:60])
            return

        # Whitelist : user_id + ids dans content doivent être dans le lot.
        allowed = allowed_ids if allowed_ids is not None else None
        if allowed is not None:
            if user_id is not None and user_id not in allowed:
                logger.debug("Mémoire rejetée (user_id hors lot): %s", user_id)
                return
            content_ids = _ids_in_text(content)
            if self.bot_user_id is not None:
                content_ids.discard(self.bot_user_id)
            unknown = content_ids - allowed
            if unknown:
                logger.debug("Mémoire rejetée (ids content hors lot): %s", unknown)
                return

        if kind == "create":
            if category == "user" and user_id is None:
                return
            # Dédup : fait quasi identique déjà en base pour cette personne/ce serveur.
            existing_contents = (existing_by_user or {}).get(
                user_id if category == "user" else None, [],
            )
            if is_near_duplicate(content, existing_contents):
                logger.debug("Mémoire rejetée (doublon proche): %s", content[:60])
                return
            # stable / direct→MARIA / collectif → actif ; passif user → pending (2 hits).
            stable = bool(action.get("stable")) and category == "user"
            collective = category in (CATEGORY_SERVER, CATEGORY_EVENT)
            from_direct = (
                category == "user"
                and user_id is not None
                and direct_user_ids is not None
                and user_id in direct_user_ids
            )
            if stable or collective or from_direct:
                if stable:
                    conf, label = CONFIDENCE_STABLE, "stable"
                elif from_direct:
                    conf, label = CONFIDENCE_DIRECT, "direct"
                else:
                    conf, label = CONFIDENCE_COLLECTIVE, "collectif"
                mem = await asyncio.to_thread(
                    self.store.create,
                    category=category,
                    guild_id=guild_id,
                    content=content,
                    user_id=user_id,
                    confidence=conf,
                    status=STATUS_ACTIVE,
                )
                await asyncio.to_thread(
                    self.vectors.upsert,
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )
                if existing_by_user is not None:
                    existing_by_user.setdefault(
                        user_id if category == "user" else None, [],
                    ).append(content)
                logger.info("Mémoire %s %s: %s", label, mem.id[:8], mem.content[:60])
                return
            mem = await asyncio.to_thread(
                self.store.create,
                category=category,
                guild_id=guild_id,
                content=content,
                user_id=user_id,
                confidence=CONFIDENCE_PENDING,
                status=STATUS_PENDING,
            )
            if existing_by_user is not None:
                existing_by_user.setdefault(user_id, []).append(content)
            logger.debug("Mémoire pending (passif) %s: %s", mem.id[:8], mem.content[:60])
            return

        if not target_id:
            return
        existing = await asyncio.to_thread(self.store.get, target_id)
        if existing is None:
            return
        if self.bot_user_id is not None and existing.user_id == self.bot_user_id:
            return
        # Souvenirs user = globaux ; server/event restent scopés au guild.
        if existing.category != "user" and existing.guild_id != guild_id:
            return

        if kind in ("update", "merge"):
            new_content = content or existing.content
            if existing.category == "user" and existing.user_id is not None:
                merge_names = dict(names)
                if existing.user_id not in merge_names:
                    hint = (existing.content or "").split(":", 1)[0]
                    hint = _SNOWFLAKE_PARENS_RE.sub("", hint).strip()
                    if hint:
                        merge_names[existing.user_id] = hint
                _, new_content = normalize_user_memory(
                    new_content,
                    user_id=existing.user_id,
                    name_by_id=merge_names,
                    lock_user_id=True,
                )
                if not new_content:
                    return
            if allowed is not None:
                content_ids = _ids_in_text(new_content)
                if self.bot_user_id is not None:
                    content_ids.discard(self.bot_user_id)
                if content_ids - allowed:
                    logger.debug("Update rejeté (ids content hors lot)")
                    return
            was_pending = existing.status == STATUS_PENDING
            # Vague check aussi sur update
            if len(new_content) > _MEMORY_CONTENT_MAX or is_too_vague(new_content):
                logger.debug("Update rejeté (vague/long): %s", new_content[:60])
                return
            mem = await asyncio.to_thread(
                self.store.update_content,
                target_id, new_content, confidence_delta=CONFIDENCE_UPDATE_DELTA,
            )
            if mem is None:
                return
            # Confirmation en parlant à MARIA → boost confiance / actif immédiat.
            from_direct = (
                mem.category == "user"
                and mem.user_id is not None
                and direct_user_ids is not None
                and mem.user_id in direct_user_ids
            )
            if from_direct and mem.confidence < CONFIDENCE_STABLE:
                mem = await asyncio.to_thread(
                    self.store.promote_direct, mem.id, new_content,
                ) or mem
            # Promotion pending → active : indexation Chroma seulement à ce moment.
            if mem.status == STATUS_ACTIVE and (was_pending or mem.chroma_id or from_direct):
                await asyncio.to_thread(
                    self.vectors.upsert,
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )
            if was_pending and mem.status == STATUS_ACTIVE:
                logger.info("Mémoire promue %s (hits=%s): %s", mem.id[:8], mem.hits, mem.content[:60])
            return

        if kind == "contradict":
            mem = await asyncio.to_thread(self.store.contradict, target_id)
            if mem is None:
                return
            if mem.status != STATUS_ACTIVE:
                await asyncio.to_thread(self.vectors.delete, target_id)
            else:
                await asyncio.to_thread(
                    self.vectors.upsert,
                    mem.id, mem.content,
                    category=mem.category, guild_id=mem.guild_id,
                    user_id=mem.user_id, confidence=mem.confidence,
                )

    @tasks.loop(hours=24)
    async def _decay_loop(self) -> None:
        try:
            archived = await asyncio.to_thread(self.store.apply_decay)
            for mid in archived:
                await asyncio.to_thread(self.vectors.delete, mid)
        except Exception as e:
            logger.warning("Decay mémoire échoué: %s", e)

    @_decay_loop.before_loop
    async def _before_decay(self) -> None:
        await asyncio.sleep(60)

    @tasks.loop(minutes=2)
    async def _timeout_loop(self) -> None:
        """Flush les buffers qui ont dépassé MEMORY_FLUSH_MINUTES."""
        keys: list[tuple[int, int]] = []
        async with self._lock:
            for key, buf in self._buffers.items():
                if self._should_flush_timeout(buf):
                    keys.append(key)
        for key in keys:
            await self._flush_key(key)

    @_timeout_loop.before_loop
    async def _before_timeout(self) -> None:
        await asyncio.sleep(30)
