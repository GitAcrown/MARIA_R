"""Contexte de conversation — fenêtre tokens / âge / compte."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

DEFAULT_WINDOW = 8192
DEFAULT_AGE = timedelta(hours=2)
SESSION_MSG_SHARE = 0.60
SESSION_SUMMARY_CHARS = 1200
CONTEXT_KEEP_RECENT = 4
CONTEXT_CHATTER_MAX_WORDS = 3
CONTEXT_CHATTER_MAX_CHARS = 20
TIME_GAP_MARKER = timedelta(minutes=20)
_NAME_ID_RE = re.compile(r"(?<![\w])([^\s()\[\]:|]{1,40}) \((\d{15,20})\)")


@dataclass
class ContentComponent:
    type: Literal["text", "image_url"]
    data: dict
    token_count: int = 0

    def to_payload(self) -> dict:
        return self.data


class TextComponent(ContentComponent):
    def __init__(self, text: str):
        super().__init__(
            type="text",
            data={"type": "text", "text": text},
            token_count=len(TOKENIZER.encode(text)),
        )


class ImageComponent(ContentComponent):
    def __init__(self, url: str, detail: Literal["low", "high", "auto"] = "auto"):
        super().__init__(
            type="image_url",
            data={"type": "image_url", "image_url": {"url": url, "detail": detail}},
            token_count=85 if detail == "low" else 250,
        )


class MetadataComponent(ContentComponent):
    def __init__(self, title: str, **meta):
        text = f"<{title.upper()}"
        if meta:
            text += " " + " ".join(f"{k}={v}" for k, v in meta.items())
        text += ">"
        super().__init__(
            type="text",
            data={"type": "text", "text": text},
            token_count=len(TOKENIZER.encode(text)),
        )


@dataclass
class MessageRecord:
    role: Literal["user", "assistant", "developer", "tool"]
    components: list[ContentComponent]
    created_at: datetime
    name: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def token_count(self) -> int:
        return sum(c.token_count for c in self.components)

    @property
    def full_text(self) -> str:
        return "".join(
            c.data.get("text", "")
            for c in self.components
            if c.type == "text" and "text" in c.data
        )

    def to_payload(self) -> dict:
        p = {
            "role": self.role,
            "content": [c.to_payload() for c in self.components],
        }
        if self.name:
            p["name"] = self.name
        return p


@dataclass
class ToolCallRecord:
    id: str
    function_name: str
    arguments: dict

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


class AssistantRecord(MessageRecord):
    def __init__(
        self,
        components: list[ContentComponent],
        created_at: datetime,
        tool_calls: Optional[list["ToolCallRecord"]] = None,
        finish_reason: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            role="assistant",
            components=components,
            created_at=created_at,
            **kwargs,
        )
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason

    def to_payload(self) -> dict:
        if self.tool_calls:
            return {
                "role": "assistant",
                "tool_calls": [t.to_payload() for t in self.tool_calls],
                "content": None,
            }
        return super().to_payload()


class ToolResponseRecord(MessageRecord):
    def __init__(
        self,
        tool_call_id: str,
        response_data: dict,
        created_at: datetime,
        **kwargs,
    ):
        summary = response_data.get("_llm_summary")
        content = summary if summary else json.dumps(response_data, ensure_ascii=False)
        super().__init__(
            role="tool",
            components=[TextComponent(content)],
            created_at=created_at,
            **kwargs,
        )
        self.tool_call_id = tool_call_id
        self.response_data = response_data

    def compact_for_history(self, *, cap: int = 160) -> None:
        """Ne garde qu'une ligne : le tour suivant n'a plus le JSON complet."""
        data = self.response_data or {}
        if data.get("_history_compacted"):
            return
        tool = str(data.get("_tool") or "outil")
        summary = (data.get("_llm_summary") or "").strip()
        if not summary:
            err = data.get("error")
            if err:
                summary = f"{tool}: {err}"
            else:
                bits = _short_result_bits(data)
                summary = f"{tool}: {bits}" if bits else f"{tool}: résultat utilisé."
        summary = re.sub(r"\s+", " ", summary).strip()
        if len(summary) > cap:
            summary = summary[: cap - 1].rstrip() + "…"
        self.response_data = {
            "_tool": tool,
            "_llm_summary": summary,
            "_history_compacted": True,
        }
        self.components = [TextComponent(summary)]

    def to_payload(self) -> dict:
        summary = self.response_data.get("_llm_summary")
        content = summary if summary else json.dumps(self.response_data, ensure_ascii=False)
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": self.tool_call_id,
        }


def _short_result_bits(data: dict, *, cap: int = 120) -> str:
    """Morceaux courts d'un résultat d'outil (titres), pas le corps."""
    bits: list[str] = []

    def walk(obj, depth: int = 0) -> None:
        if sum(len(b) for b in bits) >= cap or depth > 3:
            return
        if isinstance(obj, str):
            text = obj.strip()
            if 2 < len(text) <= 80:
                bits.append(text)
        elif isinstance(obj, dict):
            for key, val in obj.items():
                if str(key).startswith("_"):
                    continue
                walk(val, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:4]:
                walk(item, depth + 1)

    walk(data)
    return " · ".join(bits)[:cap]


def _is_context_chatter(message: "MessageRecord") -> bool:
    """True pour un [contexte] trop court pour le résumé de session."""
    if not (message.metadata or {}).get("context_only"):
        return False
    text = (message.full_text or "").strip()
    if not text:
        return True
    if any(mark in text for mark in ("[EMBED]", "[LAYOUT]", "[VIDEO:")):
        return False
    bodies: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("[Cité"):
            continue
        if line.startswith("[contexte]"):
            idx = line.find(": ")
            bodies.append(line[idx + 2 :].strip() if idx >= 0 else "")
            continue
        return False
    speech = " ".join(b for b in bodies if b).strip()
    if not speech:
        return True
    words = speech.split()
    return len(words) <= CONTEXT_CHATTER_MAX_WORDS and len(speech) <= CONTEXT_CHATTER_MAX_CHARS


def _counts_as_gap(message: "MessageRecord") -> bool:
    if message.role == "tool" or getattr(message, "tool_calls", None):
        return False
    if message.role == "user" and getattr(message, "name", None) == "system":
        return False
    if message.role not in ("user", "assistant"):
        return False
    return not _is_context_chatter(message)


def _is_conversational(message: "MessageRecord") -> bool:
    if message.role == "assistant":
        return not getattr(message, "tool_calls", None)
    return message.role == "user" and getattr(message, "name", None) != "system"


def _format_pause(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min plus tard"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest:02d} plus tard" if rest else f"{hours} h plus tard"


def _gap_marker(omitted: int, pause: Optional[timedelta]) -> str:
    bits: list[str] = []
    if omitted > 0:
        bits.append(f"{omitted} message{'s' if omitted > 1 else ''} omis")
    if pause is not None and pause >= TIME_GAP_MARKER:
        bits.append(_format_pause(pause))
    return f"[… {' · '.join(bits)}]" if bits else ""


def _rewrite_text_parts(content, fn) -> list:
    """Copie les dicts texte avant mutation (`data` est partagé avec le contexte)."""
    if not isinstance(content, list):
        return content
    out: list = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            new = fn(part.get("text", ""))
            if new != part.get("text"):
                part = {**part, "text": new}
        out.append(part)
    return out


class ConversationContext:
    def __init__(
        self,
        developer_prompt: str,
        *,
        context_window: int = DEFAULT_WINDOW,
        context_age: timedelta = DEFAULT_AGE,
        max_messages: int = 0,
    ):
        self.developer_prompt = developer_prompt
        self.context_window = context_window
        self.context_age = context_age
        self.max_messages = max_messages  # 0 = illimité
        self._messages: list[MessageRecord] = []
        self._needs_trim = False
        self.session_summary: str = ""

    def add_message(self, msg: MessageRecord) -> None:
        self._messages.append(msg)
        self._needs_trim = True

    def add_user_message(
        self,
        components: list[ContentComponent],
        name: str = "user",
        discord_message=None,
        **meta,
    ) -> MessageRecord:
        r = MessageRecord(
            role="user",
            components=components,
            created_at=datetime.now(timezone.utc),
            name=name,
            metadata=meta,
        )
        if discord_message is not None:
            r.metadata["discord_message"] = discord_message
        self.add_message(r)
        return r

    def add_assistant_message(
        self,
        components: list[ContentComponent],
        tool_calls: Optional[list[ToolCallRecord]] = None,
        finish_reason: Optional[str] = None,
        **meta,
    ) -> AssistantRecord:
        r = AssistantRecord(
            components=components,
            created_at=datetime.now(timezone.utc),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            metadata=meta,
        )
        self.add_message(r)
        return r

    def get_recent_messages(self, count: int) -> list[MessageRecord]:
        return self._messages[-count:] if count > 0 else []

    def get_messages(self) -> list[MessageRecord]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._needs_trim = False
        self.session_summary = ""

    def truncate_from(self, predicate) -> list["MessageRecord"]:
        """Retire le premier message matchant `predicate` et tout ce qui suit."""
        idx = None
        for i, m in enumerate(self._messages):
            if predicate(m):
                idx = i
                break
        if idx is None:
            return []
        removed = self._messages[idx:]
        self._messages = self._messages[:idx]
        return removed

    _SYSTEM_NOTE_MIN_AGE = timedelta(minutes=15)

    def trim(self) -> None:
        """Évince par âge, budget tokens, puis plafond de messages."""
        now = datetime.now(timezone.utc)

        def _keep_by_age(m: "MessageRecord") -> bool:
            age = now - m.created_at
            if age < self.context_age:
                return True
            if m.role == "user" and getattr(m, "name", None) == "system":
                return age < self._SYSTEM_NOTE_MIN_AGE
            return False

        aged_out = [m for m in self._messages if not _keep_by_age(m)]
        self._messages = [m for m in self._messages if _keep_by_age(m)]
        dev_tokens = len(TOKENIZER.encode(self.developer_prompt)) if self.developer_prompt else 0
        effective_window = max(self.context_window - dev_tokens, 0)
        if self.session_summary:
            msg_window = max(int(effective_window * SESSION_MSG_SHARE), 0)
        else:
            msg_window = effective_window
        priority_signal, recent_ctx, old_ctx = self._split_context_priority(self._messages)
        if self.context_window <= 0:
            selected = list(self._messages)
        else:
            selected = self._take_newest(priority_signal, msg_window)
            spent = sum(m.token_count for m in selected)
            for pool in (recent_ctx, old_ctx):
                room = msg_window - spent
                if room <= 0:
                    break
                extra = self._take_newest(pool, room, keep_oversize=False)
                selected.extend(extra)
                spent += sum(m.token_count for m in extra)
        selected_ids = {id(m) for m in selected}
        kept = [m for m in self._messages if id(m) in selected_ids]
        kept = self._cap_messages(kept, old_ctx)
        self._record_gaps(self._messages, kept)
        kept_ids = {id(m) for m in kept}
        evicted = aged_out + [m for m in self._messages if id(m) not in kept_ids]
        self._fold_evicted(evicted)
        self._messages = self._sanitize_tool_pairs(kept)
        self._needs_trim = False

    def compact_tool_history(self) -> None:
        """Résume les résultats d'outils déjà consommés, avant le prochain tour."""
        for m in self._messages:
            compact = getattr(m, "compact_for_history", None)
            if m.role == "tool" and callable(compact):
                compact()

    def history_mix(self) -> tuple[int, int, int, int]:
        """(messages, tokens [contexte], tokens outils, tokens [SYSTEM])."""
        ctx = tool = system = 0
        for m in self._messages:
            tokens = m.token_count
            if m.role == "tool" or getattr(m, "tool_calls", None):
                tool += tokens
            elif (m.metadata or {}).get("context_only"):
                ctx += tokens
            elif m.role == "user" and getattr(m, "name", None) == "system":
                system += tokens
            elif (m.full_text or "").lstrip().startswith("[SYSTEM]"):
                system += tokens
        return len(self._messages), ctx, tool, system

    @staticmethod
    def _record_gaps(before: list["MessageRecord"], kept: list["MessageRecord"]) -> None:
        """Cumule `gap_before` sur le message gardé qui suit un trou (hors tête de fenêtre)."""
        kept_ids = {id(m) for m in kept}
        missing = 0
        seen_kept = False
        for m in before:
            if id(m) in kept_ids:
                if missing and seen_kept:
                    m.metadata["gap_before"] = m.metadata.get("gap_before", 0) + missing
                missing = 0
                seen_kept = True
            else:
                missing += m.metadata.get("gap_before", 0)
                if _counts_as_gap(m):
                    missing += 1

    @staticmethod
    def _is_context_only(message: "MessageRecord") -> bool:
        return bool((message.metadata or {}).get("context_only"))

    @classmethod
    def _split_context_priority(
        cls, messages: list["MessageRecord"]
    ) -> tuple[list["MessageRecord"], list["MessageRecord"], list["MessageRecord"]]:
        """(signal, [contexte] récent, [contexte] ancien)."""
        ctx = [m for m in messages if cls._is_context_only(m)]
        recent_ids = {id(m) for m in ctx[-CONTEXT_KEEP_RECENT:]}
        signal: list[MessageRecord] = []
        recent: list[MessageRecord] = []
        old: list[MessageRecord] = []
        for m in messages:
            if not cls._is_context_only(m):
                signal.append(m)
            elif id(m) in recent_ids:
                recent.append(m)
            else:
                old.append(m)
        return signal, recent, old

    def _take_newest(
        self,
        pool: list["MessageRecord"],
        budget: int,
        *,
        keep_oversize: bool = True,
    ) -> list["MessageRecord"]:
        """Plus récents de `pool` dans `budget` tokens."""
        if not pool:
            return []
        total = 0
        chosen: list[MessageRecord] = []
        for m in reversed(pool):
            if (
                self.context_window > 0
                and total + m.token_count > budget
                and (chosen or not keep_oversize)
            ):
                break
            chosen.append(m)
            total += m.token_count
        chosen.reverse()
        return chosen

    def _cap_messages(
        self,
        kept: list["MessageRecord"],
        expendable: list["MessageRecord"],
    ) -> list["MessageRecord"]:
        if self.max_messages <= 0 or len(kept) <= self.max_messages:
            return kept
        overflow = len(kept) - self.max_messages
        expendable_ids = {id(m) for m in expendable}
        drop: set[int] = set()
        for m in kept:
            if overflow <= 0:
                break
            if id(m) in expendable_ids:
                drop.add(id(m))
                overflow -= 1
        for m in kept:
            if overflow <= 0:
                break
            if id(m) not in drop:
                drop.add(id(m))
                overflow -= 1
        return [m for m in kept if id(m) not in drop]

    def _fold_evicted(self, messages: list["MessageRecord"]) -> None:
        bits: list[str] = []
        for m in messages:
            if m.role == "tool" or getattr(m, "tool_calls", None):
                continue
            if m.role == "user" and getattr(m, "name", None) == "system":
                continue
            if (m.metadata or {}).get("context_only"):
                continue
            text = (m.full_text or "").strip()
            if not text or text.startswith("[SYSTEM]"):
                continue
            text = re.sub(r"\s+", " ", text)
            if len(text) > 180:
                text = text[:180].rstrip() + "…"
            bits.append(text)
        if not bits:
            return
        blob = " · ".join(bits)
        prev = (self.session_summary or "").strip()
        merged = f"{prev} | {blob}" if prev else blob
        if len(merged) > SESSION_SUMMARY_CHARS:
            merged = merged[-SESSION_SUMMARY_CHARS:].lstrip(" |")
        self.session_summary = merged

    @staticmethod
    def _sanitize_tool_pairs(messages: list) -> list:
        required_ids: set[str] = set()
        present_ids: set[str] = set()
        for m in messages:
            if m.role == "assistant":
                for tc in getattr(m, "tool_calls", []) or []:
                    required_ids.add(tc.id)
            elif m.role == "tool":
                tid = getattr(m, "tool_call_id", None)
                if tid:
                    present_ids.add(tid)

        orphan_calls = required_ids - present_ids
        orphan_responses = present_ids - required_ids

        if not orphan_calls and not orphan_responses:
            return messages

        clean: list = []
        skip_tool_ids: set[str] = orphan_responses.copy()
        for m in messages:
            if m.role == "assistant" and any(
                tc.id in orphan_calls for tc in (getattr(m, "tool_calls", []) or [])
            ):
                for tc in getattr(m, "tool_calls", []) or []:
                    skip_tool_ids.add(tc.id)
                continue
            if m.role == "tool" and getattr(m, "tool_call_id", None) in skip_tool_ids:
                continue
            clean.append(m)
        return clean

    def prepare_payload(self) -> list[dict]:
        if self._needs_trim:
            self.trim()
        dev = MessageRecord(
            role="developer",
            components=[TextComponent(self.developer_prompt)],
            created_at=datetime.now(timezone.utc),
        )
        payload = [dev.to_payload()]
        seen_ids: set[str] = set()
        prev_t: Optional[datetime] = None
        carried_gap = 0
        for m in self._messages:
            p = m.to_payload()
            carried_gap += m.metadata.get("gap_before", 0)

            if m.role == "user" and getattr(m, "name", None) != "system":
                pause = (m.created_at - prev_t) if prev_t is not None else None
                marker = _gap_marker(carried_gap, pause)
                carried_gap = 0

                def _dedupe(text: str) -> str:
                    def _sub(mo: re.Match) -> str:
                        if mo.group(2) in seen_ids:
                            return mo.group(1)
                        seen_ids.add(mo.group(2))
                        return mo.group(0)
                    return _NAME_ID_RE.sub(_sub, text)

                p["content"] = _rewrite_text_parts(p["content"], _dedupe)
                if marker and isinstance(p["content"], list):
                    p["content"] = [{"type": "text", "text": marker}] + p["content"]
            elif (
                m.role == "assistant"
                and not getattr(m, "tool_calls", None)
                and m.metadata.get("reply_to")
                and isinstance(p.get("content"), list)
            ):
                tag = f"[à {m.metadata['reply_to']}] "
                done = False

                def _prefix(text: str) -> str:
                    nonlocal done
                    if done or text.startswith("<EMPTY"):
                        return text
                    done = True
                    return tag + text

                p["content"] = _rewrite_text_parts(p["content"], _prefix)

            if _is_conversational(m):
                prev_t = m.created_at
            payload.append(p)
        return payload

    def get_stats(self) -> dict:
        total = sum(m.token_count for m in self._messages)
        return {
            "total_messages": len(self._messages),
            "total_tokens": total,
            "window_usage_pct": (total / self.context_window * 100) if self.context_window else 0,
            "context_window": self.context_window,
        }

    def filter_images(self) -> None:
        for m in self._messages:
            m.components = [c for c in m.components if c.type != "image_url"]
