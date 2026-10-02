"""Client TypeSafe / JEV — décisions structurées (adresse, intent, RAG, mémoire)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from common.ttl_cache import TTLCache

logger = logging.getLogger("llm.typesafe")

# JEV vise ~150 ms : au-delà de ce plafond, mieux vaut le comportement legacy.
REQUEST_TIMEOUT = 4.0

# Seuils figés (plan JEV).
ADDRESS_THRESHOLD = 0.65
CATEGORY_CONFIDENCE = 0.5
FORCE_CONFIDENCE = 0.5
RAG_SCORE_MIN = 1.0
RAG_CONFIDENCE_MIN = 0.4
DURABLE_THRESHOLD = 0.6

GATED_CATEGORIES = (
    "none",
    "weather",
    "football",
    "transport",
    "media_topic",
    "images_search",
    "summary",
    "layout",
    "server_stats",
    "youtube",
    "web",
)


@dataclass(frozen=True)
class IntentDecision:
    force_level: str  # none | hint | require_web
    category: str  # gated category or "none"
    force_confidence: float = 0.0
    category_confidence: float = 0.0
    from_jev: bool = False


@dataclass(frozen=True)
class RelevanceHit:
    index: int
    score: float
    confidence: float


class MariaTypeSafeClient:
    """Wrapper optionnel autour d'AsyncTypeSafeClient.

    Sans clé API : `enabled` est False et toutes les méthodes échouent en
    fail-open (le caller garde le comportement legacy).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "jev-latest",
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._client: Any = None
        # blob de message → tâche d'intent (en vol ou terminée), cf. prefetch_intent.
        self._intent_tasks = TTLCache(ttl=300, maxsize=256)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def _ensure(self) -> Any:
        if not self.enabled:
            return None
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            self._client = AsyncTypeSafeClient(api_key=self._api_key, model=self._model)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug("TypeSafe aclose: %s", e)
            self._client = None

    async def system_one(self, state: Any, questions: dict) -> Any | None:
        """Un appel JEV, borné à REQUEST_TIMEOUT s : au-delà on retombe sur le legacy
        (fail-open) plutôt que de retarder la réponse du bot."""
        client = await self._ensure()
        if client is None:
            return None
        try:
            return await asyncio.wait_for(
                client.system_one(state, questions), timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("TypeSafe system_one : timeout (%.1fs), fallback legacy", REQUEST_TIMEOUT)
            return None
        except Exception as e:
            logger.warning("TypeSafe system_one échoué: %s", e)
            return None

    async def is_addressed_to_bot(
        self,
        message: str,
        *,
        bot_name: str,
    ) -> bool:
        """True si l'auteur s'adresse au bot. Fail-open (True) sans JEV / erreur."""
        if not self.enabled:
            return True
        from typesafe_sdk import Noul

        name = (bot_name or "Maria").strip() or "Maria"
        result = await self.system_one(
            {"bot_name": name, "message": (message or "").strip()},
            {
                "addressed_to_bot": Noul(
                    instructions=(
                        "Does the author address the bot named `bot_name`, "
                        "asking or expecting a reply from it?"
                    ),
                    criteria={
                        "true": "Direct address, question, or request to the bot",
                        "false": (
                            "Talking about the bot to someone else, "
                            "or mere name drop with no expectation of a reply"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return True
        try:
            noul = float(result.nouls["addressed_to_bot"].noul)
        except (KeyError, AttributeError, TypeError, ValueError):
            return True
        return noul >= ADDRESS_THRESHOLD

    def prefetch_intent(self, text: str) -> None:
        """Lance l'appel d'intent en tâche de fond (sans l'attendre). Le `resolve_intent`
        suivant sur le même texte réutilise cette tâche : l'appel JEV se déroule en
        parallèle du RAG / des profils au lieu de s'enchaîner derrière eux."""
        blob = (text or "").strip()
        if not self.enabled or not blob or self._intent_tasks.get(blob) is not None:
            return
        try:
            self._intent_tasks.set(blob, asyncio.ensure_future(self._resolve_intent_uncached(blob)))
        except RuntimeError:  # pas de boucle active : le resolve classique prendra le relais
            pass

    async def resolve_intent(self, text: str) -> IntentDecision | None:
        """Choice force_level + primary_category. None = fallback regex.

        Résultat partagé par texte (TTL 5 min) : tâche déjà lancée par `prefetch_intent`,
        régénération / édition du même message, etc. Un échec n'est jamais mémorisé."""
        if not self.enabled:
            return None
        blob = (text or "").strip()
        if not blob:
            return IntentDecision(
                force_level="none",
                category="none",
                from_jev=True,
            )
        task = self._intent_tasks.get(blob)
        if task is None:
            task = asyncio.ensure_future(self._resolve_intent_uncached(blob))
            self._intent_tasks.set(blob, task)
        # shield : l'annulation d'un appelant ne doit pas tuer la tâche partagée.
        result = await asyncio.shield(task)
        if result is None:
            self._intent_tasks.discard(blob)
        return result

    async def _resolve_intent_uncached(self, blob: str) -> IntentDecision | None:
        from typesafe_sdk import Choice

        result = await self.system_one(
            {"message": blob},
            {
                "force_level": Choice(
                    instructions=(
                        "How should a Discord bot handle tools for this user message? "
                        "`none` = chat/opinion, no tool needed. "
                        "`hint` = up-to-date fact may help but do not force a tool. "
                        "`require_web` = user explicitly wants a search or linked page read."
                    ),
                    criteria={
                        "none": "Casual chat, opinion, joke, or self-knowledge",
                        "hint": "Live fact (weather, score, news) may help; soft suggestion",
                        "require_web": "Explicit search request or external URL to read",
                    },
                ),
                "primary_category": Choice(
                    instructions=(
                        "Which specialized tool category best fits this message? "
                        "`none` if none of the others apply."
                    ),
                    criteria={
                        "none": "No specialized gated tool",
                        "weather": "Weather forecast or conditions",
                        "football": "Football / soccer scores or fixtures",
                        "transport": "Transit, trains, metro, itinerary",
                        "media_topic": "Movie, series, game, or music lookup",
                        "images_search": "User wants images/photos shown",
                        "summary": "Summarize or recap the channel",
                        "layout": "Recipe, multi-step tuto, dense comparative fiche/widget",
                        "server_stats": "Discord server or channel statistics",
                        "youtube": "YouTube video content / subtitles",
                        "web": "Read a specific web page (not a vague search)",
                    },
                ),
            },
        )
        if result is None:
            return None
        try:
            force_ans = result.choices["force_level"]
            cat_ans = result.choices["primary_category"]
            force_level = str(force_ans.choice or "none")
            category = str(cat_ans.choice or "none")
            force_conf = float(getattr(force_ans, "confidence", 0.0) or 0.0)
            cat_conf = float(getattr(cat_ans, "confidence", 0.0) or 0.0)
        except (KeyError, AttributeError, TypeError, ValueError):
            return None

        if force_level not in ("none", "hint", "require_web"):
            force_level = "none"
        if category not in GATED_CATEGORIES:
            category = "none"

        return IntentDecision(
            force_level=force_level,
            category=category,
            force_confidence=force_conf,
            category_confidence=cat_conf,
            from_jev=True,
        )

    async def score_relevance(
        self,
        query: str,
        contents: Sequence[str],
    ) -> list[RelevanceHit] | None:
        """Score chaque candidat. None = garder le ranking legacy."""
        if not self.enabled or not contents:
            return None
        from typesafe_sdk import Score

        q = (query or "").strip()
        if not q:
            return None

        questions: dict = {}
        state: dict[str, Any] = {"query": q}
        for i, content in enumerate(contents):
            key = f"c{i}"
            state[key] = (content or "").strip()[:500]
            questions[key] = Score(
                instructions=(
                    "How useful is memory candidate `" + key + "` for answering `query`? "
                    "Ignore mere lexical overlap; judge whether the fact helps this question."
                ),
                criteria=["off-topic", "weakly useful", "directly useful"],
            )

        result = await self.system_one(state, questions)
        if result is None:
            return None

        hits: list[RelevanceHit] = []
        for i in range(len(contents)):
            key = f"c{i}"
            try:
                ans = result.scores[key]
                hits.append(
                    RelevanceHit(
                        index=i,
                        score=float(ans.score),
                        confidence=float(getattr(ans, "confidence", 0.0) or 0.0),
                    )
                )
            except (KeyError, AttributeError, TypeError, ValueError):
                # Candidat illisible → on le garde (fail-open partiel).
                hits.append(RelevanceHit(index=i, score=RAG_SCORE_MIN, confidence=1.0))
        return hits

    async def is_durable_fact(
        self,
        *,
        action_content: str,
        batch_excerpt: str = "",
    ) -> bool:
        """True si le fait mérite d'être stocké. Fail-open (True) sans JEV / erreur."""
        if not self.enabled:
            return True
        from typesafe_sdk import Noul

        content = (action_content or "").strip()
        if not content:
            return False

        result = await self.system_one(
            {
                "fact": content[:600],
                "batch_excerpt": (batch_excerpt or "")[:800],
            },
            {
                "durable_fact": Noul(
                    instructions=(
                        "Is `fact` a durable identity / preference / relationship / event "
                        "worth storing long-term (city, job, birthday, lasting taste, named gag), "
                        "given optional `batch_excerpt` context?"
                    ),
                    criteria={
                        "true": (
                            "Stable fact reusable later: identity, preference with a concrete "
                            "object, relationship, dated event"
                        ),
                        "false": (
                            "One-off chat, vague vibe, transient request, or noise "
                            "(asked for weather, empty gag, unfinished thought)"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return True
        try:
            noul = float(result.nouls["durable_fact"].noul)
        except (KeyError, AttributeError, TypeError, ValueError):
            return True
        return noul >= DURABLE_THRESHOLD

    async def filter_durable_actions(
        self,
        actions: Sequence[dict],
        *,
        batch_excerpt: str = "",
    ) -> list[dict]:
        """Filtre les actions d'extraction. Fail-open = liste inchangée."""
        if not self.enabled or not actions:
            return list(actions)

        from typesafe_sdk import Noul

        questions: dict = {}
        state: dict[str, Any] = {"batch_excerpt": (batch_excerpt or "")[:800]}
        for i, action in enumerate(actions):
            key = f"a{i}"
            state[key] = (action.get("content") or "")[:600]
            questions[key] = Noul(
                instructions=(
                    "Is proposed memory `" + key + "` a durable fact worth storing "
                    "(identity, lasting preference, relationship, dated event), "
                    "given `batch_excerpt`?"
                ),
                criteria={
                    "true": "Stable reusable fact",
                    "false": "One-off chat, vague, or noise",
                },
            )

        result = await self.system_one(state, questions)
        if result is None:
            return list(actions)

        kept: list[dict] = []
        for i, action in enumerate(actions):
            key = f"a{i}"
            try:
                noul = float(result.nouls[key].noul)
            except (KeyError, AttributeError, TypeError, ValueError):
                kept.append(action)
                continue
            if noul >= DURABLE_THRESHOLD:
                kept.append(action)
            else:
                logger.info(
                    "JEV a rejeté un souvenir non durable (noul=%.2f): %s",
                    noul,
                    (action.get("content") or "")[:80],
                )
        return kept
