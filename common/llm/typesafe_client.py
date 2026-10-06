"""Client TypeSafe / JEV."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from common.ttl_cache import TTLCache

logger = logging.getLogger("llm.typesafe")

REQUEST_TIMEOUT = 4.0

ADDRESS_THRESHOLD = 0.65
CATEGORY_CONFIDENCE = 0.5
FORCE_CONFIDENCE = 0.5
RAG_SCORE_MIN = 1.0
RAG_CONFIDENCE_MIN = 0.4
DURABLE_THRESHOLD = 0.5
FOLLOWUP_CONFIDENCE = 0.48
# React « à froid » : plus strict que la mention — évite le spam d'emojis.
REACTION_CONFIDENCE = 0.68
BANDWAGON_CONFIDENCE = 0.55
# React JEV (mention / follow-up) sous ce seuil → traité comme ignore.
REACT_VERDICT_CONFIDENCE = 0.62


def _heuristic_tab(message: str, labels: Sequence[str]) -> int | None:
    """Repli lexical (demain, jour de la semaine…) si JEV est off ou KO."""
    import re

    msg = (message or "").casefold()
    if not msg or len(labels) < 2:
        return None
    tokens = set(re.findall(r"[a-z0-9àâäéèêëïîôùûüçœæ]{3,}", msg))
    best_i: int | None = None
    best = 0.0
    for i, lab in enumerate(labels):
        low = lab.casefold()
        score = 0.0
        if "demain" in msg and "demain" in low:
            score += 5.0
        if "aujourd" in msg and "aujourd" in low:
            score += 5.0
        if "semaine" in msg and "semaine" in low:
            score += 4.0
        lab_tok = set(re.findall(r"[a-z0-9àâäéèêëïîôùûüçœæ]{3,}", low))
        score += 2.0 * len(tokens & lab_tok)
        if score > best:
            best = score
            best_i = i
    return best_i if best >= 2.0 else None


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
    "tasks",
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
    """Wrapper AsyncTypeSafeClient. Sans clé : enabled=False, méthodes en fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "jev-latest",
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model = model
        self._client: Any = None
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
        """Appel JEV borné à REQUEST_TIMEOUT ; None = timeout / erreur."""
        client = await self._ensure()
        if client is None:
            return None
        try:
            return await asyncio.wait_for(
                client.system_one(state, questions), timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("TypeSafe system_one : timeout (%.1fs)", REQUEST_TIMEOUT)
            return None
        except Exception as e:
            logger.warning("TypeSafe system_one échoué: %s", e)
            return None

    async def healthcheck(self) -> tuple[bool, str]:
        """Petit appel JEV pour vérifier la clé. (ok, détail)."""
        if not self.enabled:
            return False, "clé absente"
        try:
            from typesafe_sdk import Noul
        except ImportError as e:
            return False, f"SDK manquant ({e})"
        result = await self.system_one(
            {"ping": "ok"},
            {
                "alive": Noul(
                    instructions="Is the value of `ping` equal to the string ok?",
                    criteria={"true": "yes", "false": "no"},
                ),
            },
        )
        if result is None:
            return False, "timeout ou erreur (voir logs)"
        try:
            noul = float(result.nouls["alive"].noul)
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            return False, f"réponse illisible ({e})"
        return True, f"ok · noul={noul:.2f} · modèle {self._model}"

    async def is_addressed_to_bot(
        self,
        message: str,
        *,
        bot_name: str,
    ) -> bool:
        """True si l'auteur s'adresse au bot. Sans JEV / erreur → True."""
        decision = await self.classify_bot_mention(message, bot_name=bot_name)
        return decision == "respond"

    async def classify_bot_mention(
        self,
        message: str,
        *,
        bot_name: str,
    ) -> str:
        """Mention / nom du bot : respond | react | ignore.

        Sans JEV / erreur → respond (fail-open comme avant).
        """
        if not self.enabled:
            return "respond"
        from typesafe_sdk import Choice

        name = (bot_name or "Maria").strip() or "Maria"
        snippet = (message or "").strip()
        # Déjà fenêtré côté chat ; garde-fou si appelé ailleurs avec un pavé.
        if len(snippet) > 500:
            m = re.search(
                rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])",
                snippet,
                flags=re.IGNORECASE,
            )
            if m is not None:
                before = 500 // 3
                start = max(0, m.start() - before)
                end = min(len(snippet), start + 500)
                start = max(0, end - 500)
                snippet = snippet[start:end]
            else:
                snippet = snippet[:500]
        result = await self.system_one(
            {"bot_name": name, "message": snippet},
            {
                "mention": Choice(
                    instructions=(
                        "The bot `bot_name` was named or mentioned in `message` "
                        "(may be a long preamble with her name mid/end — the `…` ellipsis "
                        "means truncated context). "
                        "How should she handle it in a casual Discord group chat?"
                    ),
                    criteria={
                        "respond": (
                            "Direct address: question, request, greeting to her, "
                            "opinion asked of her, or clear expectation of a written reply "
                            "— even if her name comes after a long setup"
                        ),
                        "react": (
                            "Worth a light emoji ack without words: joke she is in, "
                            "talking about her with room for a vibe reaction, "
                            "group banter she can nod to — NOT asking her to answer"
                        ),
                        "ignore": (
                            "Passive name-drop only: listed among other members, "
                            "roll call, invite list, tags list, or talking about her "
                            "to someone else with no reason to acknowledge"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return "respond"
        try:
            ans = result.choices["mention"]
            choice = str(ans.choice or "respond")
            conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        except (KeyError, AttributeError, TypeError, ValueError):
            return "respond"
        if choice not in ("respond", "react", "ignore"):
            return "respond"
        if choice == "react" and conf < REACT_VERDICT_CONFIDENCE:
            # React forcé = spam : conf basse → silence.
            return "ignore"
        if conf < CATEGORY_CONFIDENCE:
            # Incertain : respond léger si le modèle penche vraiment, sinon silence
            # (plus de react par défaut — trop bruyant).
            if choice == "respond" and conf >= 0.35:
                return "respond"
            return "ignore"
        return choice

    def prefetch_intent(self, text: str) -> None:
        """Démarre resolve_intent en arrière-plan (même clé = même tâche)."""
        blob = (text or "").strip()
        if not self.enabled or not blob or self._intent_tasks.get(blob) is not None:
            return
        try:
            self._intent_tasks.set(blob, asyncio.ensure_future(self._resolve_intent_uncached(blob)))
        except RuntimeError:
            pass

    async def resolve_intent(self, text: str) -> IntentDecision | None:
        """force_level + primary_category. None = fallback regex. Cache TTL 5 min."""
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
                        "tasks": "Schedule, list, edit, pause, or cancel a reminder/task",
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
        """Score chaque candidat. None = ranking legacy."""
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
                hits.append(RelevanceHit(index=i, score=RAG_SCORE_MIN, confidence=1.0))
        return hits

    async def is_durable_fact(
        self,
        *,
        action_content: str,
        batch_excerpt: str = "",
    ) -> bool:
        """True si le fait est durable. Sans JEV / erreur → True."""
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

    async def batch_may_contain_durable(
        self,
        batch_excerpt: str,
        *,
        prior_excerpt: str = "",
    ) -> bool:
        """True si le lot mérite un extract GPT. Sans JEV / erreur → True (fail-open)."""
        if not self.enabled:
            return True
        excerpt = (batch_excerpt or "").strip()
        if not excerpt:
            return False
        from typesafe_sdk import Noul

        result = await self.system_one(
            {
                "batch_excerpt": excerpt[:1200],
                "prior_excerpt": (prior_excerpt or "")[:400],
            },
            {
                "has_durable": Noul(
                    instructions=(
                        "Does `batch_excerpt` (optional `prior_excerpt` for context) contain "
                        "at least one durable fact worth long-term memory "
                        "(identity, lasting preference, relationship, dated event)?"
                    ),
                    criteria={
                        "true": (
                            "At least one stable reusable fact is stated or clearly implied"
                        ),
                        "false": (
                            "Only banter, one-off requests, vibes, or noise — nothing to store"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return True
        try:
            noul = float(result.nouls["has_durable"].noul)
        except (KeyError, AttributeError, TypeError, ValueError):
            return True
        return noul >= DURABLE_THRESHOLD

    async def filter_durable_actions(
        self,
        actions: Sequence[dict],
        *,
        batch_excerpt: str = "",
    ) -> list[dict]:
        """Filtre les actions d'extraction. Sans JEV → liste inchangée."""
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

    async def classify_followup(
        self,
        message: str,
        *,
        bot_last: str,
        chain_depth: int = 0,
        attention: float = 0.5,
        fatigue: float = 0.0,
        is_addressee: bool = True,
    ) -> str:
        """Suite après une réponse de MARIA : respond / react / ignore.

        Sans JEV, ou en cas d'erreur ou de doute : ignore.
        `chain_depth` / `attention` / `fatigue` (0–1) renforcent la paresse.
        `is_addressee=False` : membre autre que le destinataire → paresse forte.
        """
        text = (message or "").strip()
        if not self.enabled or not text:
            return "ignore"
        from typesafe_sdk import Choice

        # Paresse douce : la fatigue / la chaîne comptent, pas un premier follow-up à froid.
        lazy = min(
            1.0,
            max(
                0.0,
                0.18 * max(0, chain_depth)
                + 0.45 * fatigue
                + 0.20 * (1.0 - attention)
                + (0.35 if not is_addressee else 0.0),
            ),
        )
        min_conf = FOLLOWUP_CONFIDENCE + 0.12 * lazy
        react_min = REACT_VERDICT_CONFIDENCE + 0.10 * lazy

        result = await self.system_one(
            {
                "bot_last": (bot_last or "")[:400],
                "message": text[:400],
                "laziness": f"{lazy:.2f}",
                "chain_depth": str(max(0, int(chain_depth))),
                "addressee": "yes" if is_addressee else "no",
            },
            {
                "followup": Choice(
                    instructions=(
                        "`bot_last` is what the bot MARIA just said in the channel. "
                        "`message` is a member follow-up without naming her. "
                        "`addressee` is yes if this member is who she just answered; "
                        "no = another member chiming in — prefer ignore, then react, almost never respond. "
                        "`laziness` (0–1) and `chain_depth` say how much she should stay out: "
                        "high laziness → prefer ignore, then react, rarely respond. "
                        "On the FIRST follow-up from the addressee (chain_depth 0), be responsive: "
                        "pushback, insistence, or disagreement with her answer needs words. "
                        "How would a chill friend handle it?"
                    ),
                    criteria={
                        "respond": (
                            "The ADDRESSEE continues with MARIA and needs words: question, "
                            "request, another tab/facet, OR short insistence / contradiction "
                            "(« si », « non », « fais-le », « allez », disagreeing with her refusal). "
                            "Not a bystander commenting to the room."
                        ),
                        "react": (
                            "Closing ack only (thanks, ok, lol, nice), a GIF/image/sticker "
                            "or a bare media link with no text — emoji enough, "
                            "no argument and no ask"
                        ),
                        "ignore": (
                            "Unrelated to her answer, another member talking to the room, "
                            "aimed at someone else, or nothing to answer"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return "ignore"
        try:
            ans = result.choices["followup"]
            choice = str(ans.choice or "ignore")
            conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        except (KeyError, AttributeError, TypeError, ValueError):
            return "ignore"
        if choice == "respond" and conf >= min_conf:
            return "respond"
        if choice == "react" and conf >= react_min:
            return "react"
        return "ignore"

    async def pick_tab(
        self,
        message: str,
        labels: Sequence[str],
        *,
        current: int = 0,
    ) -> int | None:
        """Quel onglet correspond au message. None = aucun / hors sujet."""
        clean = [str(lab).strip() for lab in labels if str(lab).strip()]
        if len(clean) < 2:
            return None
        if not self.enabled:
            return _heuristic_tab(message, clean)
        from typesafe_sdk import Choice

        criteria: dict[str, str] = {
            "none": "The message is not asking to show one of these tabs",
        }
        state: dict[str, Any] = {
            "message": (message or "").strip()[:400],
            "current": clean[current] if 0 <= current < len(clean) else clean[0],
        }
        for i, lab in enumerate(clean[:12]):
            key = f"t{i}"
            state[key] = lab[:80]
            criteria[key] = f"Show tab `{lab[:80]}`"

        result = await self.system_one(
            state,
            {
                "tab": Choice(
                    instructions=(
                        "`message` follows a Discord card that already has these tabs. "
                        "Which tab answers `message`? Prefer a different tab than "
                        "`current` when the user asks for another day/result. "
                        "Pick `none` if they need new data not on the card."
                    ),
                    criteria=criteria,
                ),
            },
        )
        if result is None:
            return _heuristic_tab(message, clean)
        try:
            ans = result.choices["tab"]
            choice = str(ans.choice or "none")
            conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        except (KeyError, AttributeError, TypeError, ValueError):
            return _heuristic_tab(message, clean)
        if choice == "none" or conf < 0.45:
            return None
        if choice.startswith("t") and choice[1:].isdigit():
            idx = int(choice[1:])
            if 0 <= idx < len(clean):
                return idx
        return None

    async def pick_reaction(
        self,
        message: str,
        candidates: Sequence[Any],
    ) -> Any | None:
        """Choice parmi des EmojiCandidate + `none`. None = pas de réaction."""
        if not candidates:
            return None
        if not self.enabled:
            return None
        from typesafe_sdk import Choice

        # Clés stables e0..eN (noms Discord peuvent coller / se répéter).
        criteria: dict[str, str] = {
            "none": "No reaction fits; stay silent",
        }
        state: dict[str, Any] = {"message": (message or "").strip()[:500]}
        by_key: dict[str, Any] = {}
        for i, cand in enumerate(candidates[:8]):
            key = f"e{i}"
            by_key[key] = cand
            samples = " | ".join((cand.samples or [])[:3])
            hint = getattr(cand, "hint", "")
            if samples:
                usage = f"members react with it on: {samples[:220]}"
            elif hint:
                usage = f"generally used for: {hint}"
            else:
                usage = "no usage samples yet"
            state[f"{key}_usage"] = usage[:400]
            state[f"{key}_name"] = cand.name
            kind = "Emoji" if getattr(cand, "is_unicode", False) else "Custom emoji"
            criteria[key] = f"{kind} `{cand.name}` — {usage}"

        result = await self.system_one(
            state,
            {
                "reaction": Choice(
                    instructions=(
                        "Which reaction emoji fits `message`, matching how members of "
                        "this server actually use it (see each option's usage)? "
                        "Pick `none` if nothing fits."
                    ),
                    criteria=criteria,
                ),
            },
        )
        if result is None:
            return None
        try:
            ans = result.choices["reaction"]
            choice = str(ans.choice or "none")
            conf = float(getattr(ans, "confidence", 0.0) or 0.0)
        except (KeyError, AttributeError, TypeError, ValueError):
            return None
        if choice == "none" or conf < REACTION_CONFIDENCE:
            return None
        return by_key.get(choice)

    @staticmethod
    def reaction_social_threshold(
        human_react_count: int,
        *,
        fatigue: float = 0.0,
        attention: float = 0.0,
        cold: float = 0.70,
    ) -> float:
        """Seuil Noul : ↓ avec emojis humains, ↑ si fatigue ou peu d'attention sur l'auteur."""
        n = max(0, min(int(human_react_count), 4))
        fat = max(0.0, min(1.0, fatigue))
        att = max(0.0, min(1.0, attention))
        # À froid, l'attention pèse fort ; en pile-on, un peu moins.
        cold_att_penalty = (1.0 - att) * (0.30 if n == 0 else 0.14)
        return min(
            0.95,
            max(0.35, cold - 0.08 * n + 0.10 * fat + cold_att_penalty),
        )

    async def should_ambient_react(
        self,
        message: str,
        *,
        threshold: float = 0.70,
    ) -> bool:
        """React à froid (0 emoji humain). True seulement si clairement naturel."""
        text = (message or "").strip()
        if not self.enabled or not text:
            return False
        from typesafe_sdk import Noul

        result = await self.system_one(
            {"message": text[:500]},
            {
                "ambient": Noul(
                    instructions=(
                        "In a casual Discord group chat, would a friend who was JUST "
                        "talking with this person naturally add ONE emoji reaction to "
                        "`message` without writing anything? Be selective: only clear "
                        "jokes, hype, or vibe — not every message, not strangers."
                    ),
                    criteria={
                        "true": "Clear joke, hype, wholesome beat, or vibe worth a silent emoji from someone already in the vibe",
                        "false": "Ordinary chat, question, request, stranger energy, or nothing special to ack",
                    },
                ),
            },
        )
        if result is None:
            return False
        try:
            noul = float(result.nouls["ambient"].noul)
        except (KeyError, AttributeError, TypeError, ValueError):
            return False
        return noul >= threshold

    async def should_join_reaction(
        self,
        *,
        message: str,
        emoji_name: str,
        human_count: int,
        author_name: str = "",
        threshold: float | None = None,
    ) -> bool:
        """Pile-on : d'autres membres ont déjà mis cet emoji. True = MARIA le remet aussi."""
        if not self.enabled or human_count < 1:
            return False
        from typesafe_sdk import Noul

        cut = BANDWAGON_CONFIDENCE if threshold is None else float(threshold)
        result = await self.system_one(
            {
                "message": (message or "").strip()[:500],
                "emoji": (emoji_name or "").strip()[:80],
                "human_count": human_count,
                "author": (author_name or "").strip()[:80],
            },
            {
                "join": Noul(
                    instructions=(
                        "Human members (`human_count`) already reacted with "
                        "emoji `emoji` on `author`'s message `message`. "
                        "Would MARIA — a friend in this Discord group chat — naturally "
                        "add the SAME emoji too?"
                    ),
                    criteria={
                        "true": (
                            "The emoji fits the message; piling on is natural for a "
                            "friend who is loosely concerned or agrees (funny, shared "
                            "reaction, group vibe). Not private between two people."
                        ),
                        "false": (
                            "Private/targeted, does not fit, spammy, or MARIA has no "
                            "reason to care; staying out is better"
                        ),
                    },
                ),
            },
        )
        if result is None:
            return False
        try:
            noul = float(result.nouls["join"].noul)
        except (KeyError, AttributeError, TypeError, ValueError):
            return False
        return noul >= cut
