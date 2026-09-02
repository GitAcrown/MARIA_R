"""Compteur fun pour la vue stats serveur — un running gag par guild, 72 h.

Toutes les 72 h, un appel LLM (mémoire collective du serveur) choisit une chose
à compter dans les messages (phrase ou regex). Les hits sont bufferisés comme
l'activité : rien n'est écrit sur le chemin d'un `on_message`. Tant que le
seuil n'est pas atteint, le classement n'apparaît pas (on garde « les plus
bavards »). Une fois révélé, il remplace ce bloc jusqu'à expiration.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger("MARIA.FunStat")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "funstat.db"

KIND_PHRASE = "phrase"
KIND_REGEX = "regex"
VALID_KINDS = (KIND_PHRASE, KIND_REGEX)

PERIOD_HOURS = 72
MIN_HITS = 4
MIN_USERS = 2
MIN_MEMORIES = 3
HISTORY_KEEP = 8
PATTERN_MAX = 80
TITLE_MAX = 60
UNIT_MAX = 16
MESSAGE_SCAN_MAX = 400

_UNSAFE_RE = re.compile(
    r"(\.\*){2,}|(\+\+)|(\*\*)|\{\d{3,}|\(\?R|\(\?1\)",
)
_TOO_COMMON = frozenset({
    "oui", "non", "ok", "lol", "mdr", "ptdr", "pk", "pq", "mdp", "stp", "svp",
    "wsh", "le", "la", "les", "un", "une", "des", "je", "tu", "il", "on", "est",
    "pas", "que", "qui", "quoi", "a", "à", "et", "ou", "de", "du", "en", "ce",
    "ça", "ca", "ya", "y'a", "mdrr", "lmao", "bruh", "tf", "gg", "ez",
})

Matcher = Callable[[str], bool]


@dataclass
class FunCampaign:
    guild_id: int
    title: str
    unit: str
    kind: str
    pattern: str
    created_at: datetime
    expires_at: datetime
    revealed: bool


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
                guild_id   INTEGER PRIMARY KEY,
                title      TEXT NOT NULL,
                unit       TEXT NOT NULL,
                kind       TEXT NOT NULL,
                pattern    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revealed   INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hits (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                count    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id   INTEGER NOT NULL,
                pattern    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_funstat_hist_guild ON history(guild_id, id)"
        )


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_dt(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_to_campaign(r: sqlite3.Row) -> FunCampaign:
    return FunCampaign(
        guild_id=int(r["guild_id"]),
        title=r["title"],
        unit=r["unit"],
        kind=r["kind"],
        pattern=r["pattern"],
        created_at=_parse_dt(r["created_at"]),
        expires_at=_parse_dt(r["expires_at"]),
        revealed=bool(r["revealed"]),
    )


def compile_matcher(kind: str, pattern: str) -> Optional[Matcher]:
    """None si le motif est vide, trop large, ou dangereux à compiler."""
    pattern = (pattern or "").strip()
    if not pattern or len(pattern) > PATTERN_MAX:
        return None
    if kind == KIND_PHRASE:
        folded = pattern.casefold()
        if folded in _TOO_COMMON or len(folded) < 2:
            return None
        if len(folded) <= 4 and " " not in folded:
            try:
                rx = re.compile(r"\b" + re.escape(pattern) + r"\b", re.IGNORECASE)
            except re.error:
                return None
            return lambda text, _rx=rx: _rx.search(text[:MESSAGE_SCAN_MAX]) is not None
        return lambda text, _p=folded: _p in text[:MESSAGE_SCAN_MAX].casefold()
    if kind != KIND_REGEX:
        return None
    if _UNSAFE_RE.search(pattern):
        return None
    try:
        rx = re.compile(pattern, re.IGNORECASE | re.UNICODE)
    except re.error:
        return None
    if rx.search(""):
        return None
    return lambda text, _rx=rx: _rx.search(text[:MESSAGE_SCAN_MAX]) is not None


_PROPOSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fun_stat_campaign",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Titre du classement, style « Ont dit « j'arrive » » "
                        "ou « Ont parlé de la salle ». Court, fun, tutoiement du groupe."
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": "Unité courte du compteur, en général « fois ».",
                },
                "kind": {
                    "type": "string",
                    "enum": [KIND_PHRASE, KIND_REGEX],
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "phrase : texte à trouver (casse ignorée). "
                        "regex : motif Python, IGNORECASE, pour quelques variantes "
                        "(j'arrive|je arrive). Jamais un motif qui matche tout."
                    ),
                },
            },
            "required": ["title", "unit", "kind", "pattern"],
            "additionalProperties": False,
        },
    },
}

_PROPOSE_SYSTEM = """Tu inventes UN compteur fun pour un petit Discord entre potes.
Il sera scanné sur chaque message pendant 72 h, puis affiché comme classement (top membres).

Choisis un running gag / catchphrase / tics de langage VRAIMENT ancré dans la MÉMOIRE fournie.
Doit arriver souvent assez pour avoir quelques hits en 2-3 jours, mais pas un mot-outil.

INTERDIT :
- prénom / pseudo / id d'un membre (pas de classement « qui a dit Bob »)
- le nom du bot, les pings, oui/non/lol/mdr/ok/wsh et mots de 1-2 lettres
- sujet sensible, harcèlement, doxx, sexe impliquant un mineur
- motif trop large (.*, .+ , un seul point)

Préfère kind=phrase. regex seulement pour 2-3 variantes d'une même locution.
Titre = libellé de classement, pas une phrase complète. unit = « fois » sauf raison."""


async def propose_campaign(
    llm_client,
    *,
    model: str,
    guild_name: str,
    memories: list[str],
    recent_patterns: list[str],
) -> Optional[tuple[str, str, str, str]]:
    """Demande au LLM un (title, unit, kind, pattern). None si rien de solide."""
    if len(memories) < MIN_MEMORIES:
        return None
    mem_block = "\n".join(f"- {m}" for m in memories[:25])
    avoid = ", ".join(recent_patterns[:HISTORY_KEEP]) or "(aucun)"
    try:
        completion = await llm_client.chat(
            [
                {"role": "system", "content": _PROPOSE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Serveur : {guild_name}\n"
                        f"Déjà utilisés récemment (à éviter) : {avoid}\n\n"
                        f"MÉMOIRE COLLECTIVE :\n{mem_block}"
                    ),
                },
            ],
            model=model,
            response_format=_PROPOSE_SCHEMA,
            max_tokens=250,
        )
    except Exception as e:
        logger.warning("Proposition fun-stat échouée (%s) : %s", guild_name, e)
        return None
    choice = completion.choices[0] if completion.choices else None
    raw = (choice.message.content if choice else None) or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Fun-stat JSON illisible pour %s : %r", guild_name, raw[:200])
        return None
    title = str(data.get("title") or "").strip()[:TITLE_MAX]
    unit = str(data.get("unit") or "fois").strip()[:UNIT_MAX] or "fois"
    kind = str(data.get("kind") or KIND_PHRASE).strip().lower()
    pattern = str(data.get("pattern") or "").strip()[:PATTERN_MAX]
    if kind not in VALID_KINDS or not title or compile_matcher(kind, pattern) is None:
        logger.info("Fun-stat rejetée pour %s (kind=%s pattern=%r)", guild_name, kind, pattern)
        return None
    return title, unit, kind, pattern


class FunStatTracker:
    """`observe` est synchrone et pur mémoire. `flush` / `start_campaign` / lectures
    font de l'I/O — flush via `asyncio.to_thread`, start depuis la boucle de rotation."""

    def __init__(self):
        _init_db()
        self._campaigns: dict[int, FunCampaign] = {}
        self._matchers: dict[int, Matcher] = {}
        self._hits: dict[int, Counter[int]] = {}
        self._buffer: Counter[tuple[int, int]] = Counter()
        self._load()

    def _load(self) -> None:
        now = datetime.now(timezone.utc)
        with _db() as conn:
            rows = conn.execute("SELECT * FROM campaigns").fetchall()
            hit_rows = conn.execute("SELECT guild_id, user_id, count FROM hits").fetchall()
        hits: dict[int, Counter[int]] = {}
        for r in hit_rows:
            hits.setdefault(int(r["guild_id"]), Counter())[int(r["user_id"])] = int(r["count"])
        for r in rows:
            camp = _row_to_campaign(r)
            if camp.expires_at <= now:
                continue
            matcher = compile_matcher(camp.kind, camp.pattern)
            if matcher is None:
                continue
            self._campaigns[camp.guild_id] = camp
            self._matchers[camp.guild_id] = matcher
            self._hits[camp.guild_id] = hits.get(camp.guild_id, Counter())

    def needs_roll(self, guild_id: int) -> bool:
        camp = self._campaigns.get(guild_id)
        if camp is None:
            return True
        return camp.expires_at <= datetime.now(timezone.utc)

    def recent_patterns(self, guild_id: int, *, limit: int = HISTORY_KEEP) -> list[str]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT pattern FROM history WHERE guild_id = ? ORDER BY id DESC LIMIT ?",
                (guild_id, limit),
            ).fetchall()
        return [r["pattern"] for r in rows]

    def start_campaign(
        self, guild_id: int, *, title: str, unit: str, kind: str, pattern: str,
        period_hours: int = PERIOD_HOURS,
    ) -> Optional[FunCampaign]:
        matcher = compile_matcher(kind, pattern)
        if matcher is None:
            return None
        now = datetime.now(timezone.utc)
        camp = FunCampaign(
            guild_id=guild_id,
            title=title.strip()[:TITLE_MAX],
            unit=(unit or "fois").strip()[:UNIT_MAX] or "fois",
            kind=kind,
            pattern=pattern.strip()[:PATTERN_MAX],
            created_at=now,
            expires_at=now + timedelta(hours=period_hours),
            revealed=False,
        )
        with _db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO campaigns
                    (guild_id, title, unit, kind, pattern, created_at, expires_at, revealed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    guild_id, camp.title, camp.unit, camp.kind, camp.pattern,
                    camp.created_at.isoformat(), camp.expires_at.isoformat(),
                ),
            )
            conn.execute("DELETE FROM hits WHERE guild_id = ?", (guild_id,))
            conn.execute(
                "INSERT INTO history (guild_id, pattern, created_at) VALUES (?, ?, ?)",
                (guild_id, camp.pattern, camp.created_at.isoformat()),
            )
            conn.execute(
                """
                DELETE FROM history
                WHERE guild_id = ? AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM history WHERE guild_id = ? ORDER BY id DESC LIMIT ?
                    )
                )
                """,
                (guild_id, guild_id, HISTORY_KEEP),
            )
        # Drops les hits encore dans le buffer pour cette guild (ancienne campagne).
        self._buffer = Counter({
            k: n for k, n in self._buffer.items() if k[0] != guild_id
        })
        self._campaigns[guild_id] = camp
        self._matchers[guild_id] = matcher
        self._hits[guild_id] = Counter()
        logger.info(
            "Fun-stat #%s : « %s » [%s] %r jusqu'à %s",
            guild_id, camp.title, camp.kind, camp.pattern, camp.expires_at.isoformat(),
        )
        return camp

    def observe(self, guild_id: int, user_id: int, text: str) -> None:
        camp = self._campaigns.get(guild_id)
        if camp is None or camp.expires_at <= datetime.now(timezone.utc):
            return
        matcher = self._matchers.get(guild_id)
        if matcher is None or not text:
            return
        if not matcher(text):
            return
        self._buffer[(guild_id, user_id)] += 1
        self._hits.setdefault(guild_id, Counter())[user_id] += 1
        if not camp.revealed:
            counts = self._hits[guild_id]
            if sum(counts.values()) >= MIN_HITS and len(counts) >= MIN_USERS:
                camp.revealed = True

    def flush(self) -> None:
        items = list(self._buffer.items())
        self._buffer.clear()
        reveal_ids = [gid for gid, c in self._campaigns.items() if c.revealed]
        if not items and not reveal_ids:
            return
        try:
            with _db() as conn:
                for (guild_id, user_id), n in items:
                    conn.execute(
                        """
                        INSERT INTO hits (guild_id, user_id, count)
                        VALUES (?, ?, ?)
                        ON CONFLICT(guild_id, user_id)
                        DO UPDATE SET count = count + excluded.count
                        """,
                        (guild_id, user_id, n),
                    )
                for gid in reveal_ids:
                    conn.execute(
                        "UPDATE campaigns SET revealed = 1 WHERE guild_id = ? AND revealed = 0",
                        (gid,),
                    )
        except sqlite3.Error as e:
            logger.error("Flush fun-stat échoué : %s", e, exc_info=True)
            for key, n in items:
                self._buffer[key] += n

    def visible_ranking(
        self, guild_id: int, *, limit: int = 3,
    ) -> Optional[tuple[str, str, list[tuple[int, int]]]]:
        """Titre, unité, top (user_id, count) si la campagne est révélée et encore vive."""
        camp = self._campaigns.get(guild_id)
        if camp is None or not camp.revealed:
            return None
        if camp.expires_at <= datetime.now(timezone.utc):
            return None
        counts = self._hits.get(guild_id) or Counter()
        top = counts.most_common(limit)
        if not top:
            return None
        return camp.title, camp.unit, top

    def peek(self, guild_id: int) -> Optional[dict]:
        """Aperçu debug (commande owner) — pas d'I/O."""
        camp = self._campaigns.get(guild_id)
        if camp is None:
            return None
        counts = self._hits.get(guild_id) or Counter()
        return {
            "title": camp.title,
            "unit": camp.unit,
            "kind": camp.kind,
            "pattern": camp.pattern,
            "revealed": camp.revealed,
            "expires_at": camp.expires_at,
            "total": int(sum(counts.values())),
            "users": len(counts),
        }
