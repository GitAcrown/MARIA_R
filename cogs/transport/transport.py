"""Cog Transports — PRIM (Île-de-France) + SNCF (trains nationaux).

Un outil :
- arrêt IDF → prochains passages (PRIM)
- ligne IDF → trafic
- gare hors IDF / « train à … » → départs SNCF

Clés .env : PRIM_API_KEY, SNCF_API_KEY
(PRIM : prim.iledefrance-mobilites.fr — SNCF : numerique.sncf.com/startup/api)
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import discord
import requests
from discord.ext import commands

from common.discord_ui import layout_with_commentary
from common.dyn_widgets import make_tabbed_view, register_tabs, unregister_tabs
from common.emojis import TRAIN
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.timezones import PARIS_TZ
from common.widgets import register_widget, unregister_widget

logger = logging.getLogger("MARIA.Transport")

_PLACE_TTL = 6 * 3600
_MAX_ROWS = 10
_MODE_RANK = {
    "métro": 0, "metro": 0,
    "rer": 1,
    "train": 2, "transilien": 2, "localtrain": 2,
    "tramway": 3, "tram": 3,
    "bus": 4,
}
_SNCF_HINT = re.compile(
    r"\b(gare|tgv|ter|intercit|lyria|eurostar|train|ouigo)\b",
    re.I,
)
_IDF_HINT = re.compile(r"\b(m[ée]tro|rer|tram|bus)\b", re.I)
_ACCESSIBILITY = re.compile(r"\b(ascenseur|escalier|escalator)\b", re.I)


@dataclass(frozen=True)
class _Backend:
    name: str
    label: str
    base: str
    coverage: str
    key: str
    auth: str  # "header" | "basic"


def _parse_navitia_dt(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(tzinfo=PARIS_TZ)
    except (ValueError, TypeError):
        return None


def _wait_label(dt: datetime, *, now: datetime) -> str:
    secs = (dt - now).total_seconds()
    if secs < 45:
        return "imminent"
    mins = int(round(secs / 60))
    if mins >= 60:
        return dt.strftime("%H:%M")
    return f"{max(1, mins)} min"


def _mode_rank(mode: str) -> int:
    return _MODE_RANK.get((mode or "").lower(), 5)


def _line_code(info: dict) -> str:
    return (info.get("code") or info.get("label") or info.get("headsign") or "?").strip() or "?"


def _physical_mode(info: dict) -> str:
    return (
        info.get("commercial_mode")
        or info.get("physical_mode")
        or info.get("network")
        or ""
    ).strip()


def _fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", " ").replace("’", " ").replace("-", " ")
    return " ".join(s.split())


def _place_queries(query: str) -> list[str]:
    q = " ".join((query or "").split())
    if not q:
        return []
    out = [q]
    elided = re.sub(r"\bl ([aeiouh])", r"l'\1", q, flags=re.I)
    if elided not in out:
        out.append(elided)
    hyphen = re.sub(r"\s+", "-", elided)
    if hyphen not in out:
        out.append(hyphen)
    return out[:3]


def _place_name_bonus(query: str, name: str) -> int:
    q = _fold(query)
    n = _fold((name or "").split("(")[0])
    if not q or not n:
        return 0
    if q == n:
        return 80
    if n.startswith(q) or q.startswith(n):
        extra = len(set(n.split()) - set(q.split()))
        return 40 - 8 * extra
    return 0


def _source_label(data: dict) -> str:
    return data.get("source") or "Île-de-France Mobilités"


def _uri(ident: str) -> str:
    return quote(ident or "", safe=":")


_SKIP_CHANNELS = {"pids", "cbiv"}
_TITLE_CHANNELS = {"title", "titre", "notification"}
_NOTE_LIMIT = 1600


def _plain_text(raw: str, *, breaks: bool = False) -> str:
    text = raw or ""
    if breaks:
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</p\s*>", "\n", text)
        text = re.sub(r"(?i)<p[^>]*>", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        lines = [" ".join(ln.split()) for ln in text.splitlines()]
        compact: list[str] = []
        for ln in lines:
            if ln:
                compact.append(ln)
            elif compact and compact[-1]:
                compact.append("")
        return "\n".join(compact).strip()
    text = re.sub(r"(?i)<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("\n"), cut.rfind(". "))
    if boundary < limit * 0.5:
        boundary = cut.rfind(" ")
    if boundary <= 0:
        boundary = limit
    return cut[:boundary].rstrip(" .,;:—-") + "…"


def _disruption_copy(messages: list) -> str:
    title = ""
    body = ""
    body_len = 0
    for m in messages or []:
        ch = m.get("channel") or {}
        types = {str(t).lower() for t in (ch.get("types") or [])}
        name = (ch.get("name") or "").strip().lower()
        if types & _SKIP_CHANNELS or name in _SKIP_CHANNELS:
            continue
        text = _plain_text(m.get("text") or "", breaks=True)
        if not text:
            continue
        is_title = bool(types & {"title"}) or name in _TITLE_CHANNELS
        if is_title:
            if not title:
                title = " ".join(text.split())
            continue
        if len(text) > body_len:
            body_len = len(text)
            body = text
    if not body:
        return title
    body = _clip(body, _NOTE_LIMIT)
    if title and title.casefold() not in body[: len(title) + 40].casefold():
        return f"**{title}**\n{body}"
    return body


_LINE_PREFIX = re.compile(
    r"^(RER\s*\S+|M[ée]tro\s*\S+|Ligne\s*\S+|Tram\s*\S+|Bus\s*\S+)\s*[:–\-]\s*",
    re.I,
)


def _note_label(note: str) -> str:
    first = (note.split("\n", 1)[0] if note else "").replace("*", "").strip()
    first = " ".join(first.split())
    first = _LINE_PREFIX.sub("", first).strip()
    if not first:
        return "Info"
    if len(first) <= 90:
        return first
    return _clip(first, 90)


def traffic_tab_labels(payload: dict) -> list[str]:
    labels: list[str] = []
    seen: dict[str, int] = {}
    for note in payload.get("notes") or []:
        lab = _note_label(note)
        n = seen.get(lab, 0) + 1
        seen[lab] = n
        labels.append(lab if n == 1 else f"{_clip(lab, 80)} {n}")
    return labels


def traffic_tab_body(payload: dict, index: int) -> discord.ui.Item:
    notes = payload.get("notes") or []
    note = notes[index] if 0 <= index < len(notes) else ""
    data = dict(payload)
    data["notes"] = [note] if note else []
    return _traffic_container(data)


def build_transport_view(data: dict, commentary: str = "") -> Optional[discord.ui.LayoutView]:
    if not isinstance(data, dict) or "error" in data:
        return None
    kind = data.get("kind")
    try:
        if kind == "departures":
            container = _departures_container(data)
        elif kind == "traffic":
            notes = data.get("notes") or []
            if len(notes) >= 2:
                view = make_tabbed_view(
                    "transport_traffic",
                    {
                        "kind": "traffic",
                        "title": data.get("title"),
                        "status": data.get("status"),
                        "notes": notes,
                        "source": data.get("source"),
                    },
                    commentary,
                    0,
                )
                if view is not None:
                    return view
            container = _traffic_container(data)
        else:
            return None
    except Exception as e:
        logger.error("build_transport_view: %s", e, exc_info=True)
        return None
    return layout_with_commentary(container, commentary)


def _departures_container(data: dict) -> discord.ui.Container:
    stop = data.get("stop") or "Arrêt"
    rows = data.get("rows") or []
    realtime = bool(data.get("realtime"))
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## {TRAIN} {stop}"),
        discord.ui.TextDisplay("-# Prochains passages"),
        discord.ui.Separator(),
    ]
    if not rows:
        children.append(discord.ui.TextDisplay("-# Aucun passage proche."))
    else:
        lines = []
        for row in rows:
            code = row.get("code") or "?"
            dest = row.get("direction") or ""
            waits = " · ".join(row.get("waits") or [])
            mode = row.get("mode") or ""
            head = f"**{code}**"
            if mode and mode.lower() not in code.lower():
                head = f"{head} · {mode}"
            bit = f"{head}  {dest}" if dest else head
            if waits:
                bit = f"{bit}  ·  {waits}"
            lines.append(bit)
        children.append(discord.ui.TextDisplay("\n".join(lines)))
    stamp = "temps réel" if realtime else "horaire théorique"
    children += [
        discord.ui.Separator(),
        discord.ui.TextDisplay(f"-# {stamp} · {_source_label(data)}"),
    ]
    return discord.ui.Container(*children)


def _traffic_container(data: dict) -> discord.ui.Container:
    title = data.get("title") or "Trafic"
    status = data.get("status") or ""
    notes = data.get("notes") or []
    children: list[discord.ui.Item] = [
        discord.ui.TextDisplay(f"## {TRAIN} {title}"),
    ]
    if status:
        children.append(discord.ui.TextDisplay(status))
    children.append(discord.ui.Separator())
    if notes:
        for i, n in enumerate(notes):
            if i:
                children.append(discord.ui.Separator(spacing=discord.SeparatorSpacing.small))
            children.append(discord.ui.TextDisplay(n))
    else:
        children.append(discord.ui.TextDisplay("Trafic normal."))
    children += [
        discord.ui.Separator(),
        discord.ui.TextDisplay(f"-# {_source_label(data)}"),
    ]
    return discord.ui.Container(*children)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class Transport(commands.Cog):

    def __init__(self, bot: commands.Bot):
        cfg = getattr(bot, "config", {}) or {}
        self.bot = bot
        self._prim = _Backend(
            name="prim",
            label="Île-de-France Mobilités",
            base="https://prim.iledefrance-mobilites.fr/marketplace/v2/navitia",
            coverage="",
            key=(cfg.get("PRIM_API_KEY") or "").strip(),
            auth="header",
        )
        self._sncf = _Backend(
            name="sncf",
            label="SNCF",
            base="https://api.sncf.com/v1",
            coverage="/coverage/sncf",
            key=(cfg.get("SNCF_API_KEY") or "").strip(),
            auth="basic",
        )
        self._place_cache: dict[str, tuple[float, dict]] = {}

    def _get(self, backend: _Backend, path: str, params: Optional[dict] = None) -> dict:
        if not backend.key:
            env = "PRIM_API_KEY" if backend.name == "prim" else "SNCF_API_KEY"
            return {"error": f"Clé API manquante ({env} dans .env)"}
        url = f"{backend.base}{path}"
        kwargs: dict = {
            "params": params or {},
            "headers": {"Accept": "application/json"},
            "timeout": 12,
        }
        if backend.auth == "basic":
            kwargs["auth"] = (backend.key, "")
        else:
            kwargs["headers"]["apiKey"] = backend.key
        try:
            r = requests.get(url, **kwargs)
        except requests.RequestException as e:
            return {"error": f"Réseau {backend.label} : {e}"}
        if r.status_code == 401:
            return {"error": f"Clé {backend.label} invalide"}
        if r.status_code == 429:
            return {"error": f"Quota {backend.label} atteint, réessaie plus tard"}
        if not r.ok:
            logger.warning("PRIM/SNCF %s %s", r.status_code, url)
            return {"error": f"Erreur {backend.label} {r.status_code}"}
        try:
            payload = r.json()
        except ValueError:
            return {"error": f"Réponse {backend.label} illisible"}
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return {"error": err["message"]}
        return payload

    def _purge_place_cache(self, now: float) -> None:
        expired = [k for k, (ts, _) in self._place_cache.items() if now - ts >= _PLACE_TTL]
        for k in expired:
            del self._place_cache[k]

    def _search_place(self, backend: _Backend, query: str, *, allow_admin: bool = False) -> dict:
        key = f"{backend.name}:{allow_admin}:{query.strip().lower()}"
        cached = self._place_cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < _PLACE_TTL:
            return cached[1]
        self._purge_place_cache(now)
        best = None
        best_score = -10**9
        last_error = None
        for q in _place_queries(query):
            payload = self._get(
                backend,
                f"{backend.coverage}/places",
                {"q": q, "count": 8},
            )
            if "error" in payload:
                last_error = payload
                continue
            for place in payload.get("places") or []:
                kind = place.get("embedded_type") or ""
                embedded = (
                    place.get("stop_area")
                    or place.get("stop_point")
                    or (place.get("administrative_region") if allow_admin else None)
                )
                if not embedded:
                    continue
                name = embedded.get("name") or place.get("name") or query
                score = int(place.get("quality") or 0)
                kind_bonus = {"stop_area": 20, "stop_point": 10, "administrative_region": 5}.get(kind, 0)
                rank = score * 10 + kind_bonus + _place_name_bonus(query, name)
                if rank > best_score:
                    best_score = rank
                    best = {
                        "id": embedded.get("id") or place.get("id"),
                        "name": name,
                        "kind": kind or "stop_area",
                    }
        if not best or not best.get("id"):
            return last_error or {"error": f"Lieu introuvable ({backend.label}) : {query!r}"}
        self._place_cache[key] = (now, best)
        return best

    def _search_line(self, query: str) -> dict:
        payload = self._get(
            self._prim,
            f"{self._prim.coverage}/pt_objects",
            {"q": query, "type[]": "line", "count": 8},
        )
        if "error" in payload:
            return payload
        objects = payload.get("pt_objects") or []
        best = None
        best_score = -10**9
        q = query.strip().lower()
        tokens = re.findall(r"[a-z0-9]+", q)
        for obj in objects:
            line = obj.get("line") or {}
            lid = line.get("id")
            if not lid:
                continue
            code = (line.get("code") or "").strip().lower()
            name = (line.get("name") or "").lower()
            modes = line.get("physical_modes") or []
            mode = ((modes[0].get("name") if modes else "") or "").lower()
            score = int(obj.get("quality") or 0)
            if "remplacement" in name:
                score -= 80
            if re.search(r"\brer\b", q) and "rer" in mode:
                score += 40
            if re.search(r"\bm[ée]tro\b", q) and mode in ("métro", "metro"):
                score += 40
            if re.search(r"\btram", q) and "tram" in mode:
                score += 40
            if code and code in tokens:
                score += 25
            if score > best_score:
                best_score = score
                best = line
        if not best:
            return {"error": f"Ligne introuvable : {query!r}"}
        modes = best.get("physical_modes") or []
        mode = ((modes[0].get("name") if modes else "") or "").strip()
        return {
            "id": best["id"],
            "name": best.get("name") or best.get("code") or query,
            "code": best.get("code") or "",
            "mode": mode,
        }

    def _fetch_departures(self, backend: _Backend, stop_id: str, kind: str) -> dict:
        segment = "stop_areas" if kind != "stop_point" else "stop_points"
        return self._get(
            backend,
            f"{backend.coverage}/{segment}/{_uri(stop_id)}/departures",
            {"count": 30, "data_freshness": "realtime"},
        )

    def _group_departures(self, raw: dict, line_filter: str = "") -> tuple[list[dict], bool]:
        now = datetime.now(PARIS_TZ)
        want = line_filter.strip().lower()
        grouped: dict[tuple[str, str], dict] = {}
        realtime = False
        for dep in raw.get("departures") or []:
            info = dep.get("display_informations") or {}
            code = _line_code(info)
            mode = _physical_mode(info)
            direction = (info.get("direction") or info.get("headsign") or "").strip()
            if want and want not in code.lower() and want not in (info.get("label") or "").lower():
                if want not in mode.lower() and want not in direction.lower():
                    continue
            sdt = dep.get("stop_date_time") or {}
            dt = _parse_navitia_dt(sdt.get("departure_date_time") or "")
            if dt is None or dt < now - timedelta(minutes=1):
                continue
            if (sdt.get("data_freshness") or "") == "realtime":
                realtime = True
            key = (code, direction)
            slot = grouped.get(key)
            if slot is None:
                grouped[key] = {
                    "code": code,
                    "mode": mode,
                    "direction": direction,
                    "rank": _mode_rank(mode),
                    "times": [dt],
                }
            elif len(slot["times"]) < 2:
                slot["times"].append(dt)
        rows = sorted(grouped.values(), key=lambda r: (r["rank"], r["times"][0]))
        out = []
        for row in rows[:_MAX_ROWS]:
            out.append({
                "code": row["code"],
                "mode": row["mode"],
                "direction": row["direction"],
                "waits": [_wait_label(t, now=now) for t in row["times"]],
            })
        return out, realtime

    def _departures_on(self, backend: _Backend, stop: str, line_filter: str = "") -> dict:
        found = self._search_place(backend, stop)
        if "error" in found:
            return found
        raw = self._fetch_departures(backend, found["id"], found.get("kind") or "stop_area")
        if "error" in raw:
            return raw
        rows, realtime = self._group_departures(raw, line_filter)
        return {
            "kind": "departures",
            "stop": found["name"],
            "rows": rows,
            "realtime": realtime,
            "source": backend.label,
        }

    def _departures_payload(self, stop: str, line_filter: str = "") -> dict:
        blob = f"{stop} {line_filter}"
        prefer_sncf = bool(_SNCF_HINT.search(blob) and not _IDF_HINT.search(blob))
        first, second = (self._sncf, self._prim) if prefer_sncf else (self._prim, self._sncf)
        data = self._departures_on(first, stop, line_filter)
        if "error" not in data and data.get("rows"):
            return data
        fallback = self._departures_on(second, stop, line_filter)
        if "error" not in fallback:
            return fallback
        return data if "error" not in data else fallback

    def _disruption_texts(self, payload: dict, *, line_id: str = "") -> list[str]:
        primary: list[str] = []
        secondary: list[str] = []
        seen: set[str] = set()
        for d in payload.get("disruptions") or []:
            status = (d.get("status") or "").lower()
            if status and status not in ("active", "future"):
                continue
            line_hit = False
            for pt in d.get("impacted_objects") or []:
                pto = pt.get("pt_object") or pt
                pt_id = pto.get("id") or ""
                kind = (pto.get("embedded_type") or "").lower()
                if kind == "line":
                    line_hit = True
                if line_id and (line_id in pt_id or pt_id in line_id):
                    line_hit = True
            if line_id and d.get("impacted_objects") and not line_hit:
                continue
            msg = _disruption_copy(d.get("messages") or [])
            if not msg:
                msg = _plain_text(d.get("cause") or "")
            if not msg:
                sev = d.get("severity") or {}
                msg = _plain_text(sev.get("name") or "")
            if not msg or msg.lower() in seen:
                continue
            if _ACCESSIBILITY.search(msg):
                continue
            seen.add(msg.lower())
            if line_hit or not line_id:
                primary.append(msg)
            else:
                secondary.append(msg)
        notes = primary or secondary
        return notes[:6]

    def _traffic_payload(self, line: str) -> dict:
        found = self._search_line(line)
        if "error" in found:
            return found
        raw = self._get(
            self._prim,
            f"{self._prim.coverage}/lines/{_uri(found['id'])}/line_reports",
            {"count": 20},
        )
        if "error" in raw:
            return raw
        notes = self._disruption_texts(raw, line_id=found["id"])
        title = found["name"]
        if found.get("code") and found["code"] not in title:
            title = f"{found.get('mode') or 'Ligne'} {found['code']}".strip()
        return {
            "kind": "traffic",
            "title": title,
            "status": "Perturbé" if notes else "Trafic normal",
            "notes": notes,
            "source": self._prim.label,
        }

    def _global_traffic_payload(self) -> dict:
        raw = self._get(
            self._prim,
            f"{self._prim.coverage}/line_reports/line_reports",
            {"count": 20},
        )
        if "error" in raw:
            return raw
        notes = self._disruption_texts(raw)
        if not notes:
            return {
                "kind": "traffic",
                "title": "Trafic IDF",
                "status": "Rien de notable",
                "notes": [],
                "source": self._prim.label,
            }
        return {
            "kind": "traffic",
            "title": "Trafic IDF",
            "status": f"{len(notes)} perturbation(s)",
            "notes": notes[:8],
            "source": self._prim.label,
        }

    def _llm_summary(self, data: dict) -> str:
        kind = data.get("kind")
        if kind == "departures":
            stop = data.get("stop") or "l'arrêt"
            rows = data.get("rows") or []
            if not rows:
                return f"Aucun passage proche à {stop}. Widget affiché."
            bits = []
            for row in rows[:5]:
                waits = ", ".join(row.get("waits") or [])
                bits.append(f"{row.get('code')} {row.get('direction')} ({waits})")
            return f"Prochains passages à {stop} : {'; '.join(bits)}. Widget affiché."
        title = data.get("title") or "Trafic"
        status = data.get("status") or ""
        notes = data.get("notes") or []
        extra = f" {notes[0]}" if notes else ""
        return f"{title} : {status}.{extra} Widget affiché."

    async def _tool_transport(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        args = tc.arguments or {}
        stop = (args.get("stop") or "").strip()
        line = (args.get("line") or "").strip()
        if stop:
            data = await asyncio.to_thread(self._departures_payload, stop, line)
        elif line:
            data = await asyncio.to_thread(self._traffic_payload, line)
        else:
            data = await asyncio.to_thread(self._global_traffic_payload)
        if "error" in data:
            return ToolResponseRecord(tc.id, {"error": data["error"]}, datetime.now(timezone.utc))
        return ToolResponseRecord(tc.id, {
            "_tool": "get_transport",
            "_llm_summary": self._llm_summary(data),
            **data,
        }, datetime.now(timezone.utc))

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="get_transport",
                description=(
                    "Transports : Île-de-France (PRIM) + trains SNCF (TGV, TER, Intercités). "
                    "stop = arrêt/gare → prochains passages (IDF d'abord, SNCF si gare/train). "
                    "line seule (RER B, 11…) → trafic IDF. "
                    "Ni stop ni line → trafic global IDF. "
                    "Arrêt métro habituel en mémoire si « mon métro »."
                ),
                properties={
                    "stop": {
                        "type": "string",
                        "description": "Arrêt ou gare (République, Gare de Lyon, Part-Dieu)",
                    },
                    "line": {
                        "type": "string",
                        "description": "Ligne IDF (RER B, 11, T3a) — filtre ou trafic",
                    },
                },
                optional_props=["stop", "line"],
                function=self._tool_transport,
            ),
        ]


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(Transport(bot))
    register_widget("get_transport", build_transport_view)
    register_tabs("transport_traffic", traffic_tab_labels, traffic_tab_body)


async def teardown(bot: commands.Bot) -> None:
    unregister_widget("get_transport")
    unregister_tabs("transport_traffic")
