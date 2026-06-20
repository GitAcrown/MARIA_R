"""Conscience temporelle — note d'actualité courante injectée dans le prompt.

Principe :
- Un résumé des actualités du jour est stocké en base (table `global_config`).
- Il est rafraîchi automatiquement toutes les ~20h au démarrage et via une tâche
  quotidienne, grâce à Brave News API (même clé que le cog Web).
- Si aucune actu fraîche n'est disponible, une instruction de fallback est renvoyée
  pour que le LLM raisonne depuis la date.
- L'admin peut forcer une valeur manuelle ou déclencher un refresh via /chatbot actu.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from common.dataio import CogData, DictTableBuilder

logger = logging.getLogger("MARIA.Awareness")

STALE_AFTER = timedelta(hours=20)
MAX_CTX_LEN = 900

# Fallback injecté quand aucune actu fraîche n'est disponible
_FALLBACK = (
    "Utilise la date pour déduire les grands événements en cours "
    "(compétitions sportives, sorties, actualités…). "
    "Pour tout fait précis ou résultat → appelle l'outil approprié."
)


# ---------------------------------------------------------------------------
# Helpers HTTP (synchrones → exécutés dans un thread)
# ---------------------------------------------------------------------------

def _brave_news(api_key: str, query: str, n: int = 6) -> list[dict]:
    """Appel direct Brave News Search API."""
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/news/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": min(n * 2, 20), "search_lang": "fr", "country": "FR"},
            timeout=8,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        logger.warning(f"Brave news ({query!r}): {e}")
        return []


def _build_ctx(results: list[dict], date_str: str) -> str:
    """Construit une note compacte à partir des résultats Brave."""
    lines: list[str] = []
    seen: set[str] = set()
    for r in results:
        title = (r.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        desc = (r.get("description") or "").strip()
        line = f"- {title}"
        if desc:
            line += f" : {desc[:120]}"
        lines.append(line)
        if len(lines) >= 7:
            break
    if not lines:
        return ""
    return f"ACTU DU {date_str} :\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class CurrentAwareness:
    """Note de contexte temporel/actualités partagée entre tous les salons."""

    _TABLE = "global_config"

    def __init__(self, data: CogData, brave_key: str = "") -> None:
        self._data = data
        self._brave_key = brave_key
        # Initialise la table globale (INSERT OR IGNORE → sans écraser les données)
        self._data.set_builders(
            "global",
            DictTableBuilder(self._TABLE, {
                "current_ctx":         "",
                "current_ctx_updated": "",
            }),
        )

    # -- Accès DB ------------------------------------------------------------

    def _db(self):
        return self._data.get("global").settings(self._TABLE)

    # -- Lecture -------------------------------------------------------------

    def get(self) -> str:
        """Retourne la note si fraîche (< STALE_AFTER), vide sinon."""
        db = self._db()
        updated_str = db.get("current_ctx_updated", "")
        if not updated_str:
            return ""
        try:
            updated = datetime.fromisoformat(updated_str)
            if datetime.now(timezone.utc) - updated < STALE_AFTER:
                return db.get("current_ctx", "")
        except ValueError:
            pass
        return ""

    def peek(self) -> tuple[str, str]:
        """(texte brut, horodatage ISO) — pour l'affichage admin."""
        db = self._db()
        return (
            db.get("current_ctx", "") or "",
            db.get("current_ctx_updated", "") or "",
        )

    def is_stale(self) -> bool:
        db = self._db()
        updated_str = db.get("current_ctx_updated", "")
        if not updated_str:
            return True
        try:
            updated = datetime.fromisoformat(updated_str)
            return datetime.now(timezone.utc) - updated >= STALE_AFTER
        except ValueError:
            return True

    # -- Écriture ------------------------------------------------------------

    def set(self, text: str) -> None:
        """Définit manuellement la note (admin)."""
        db = self._db()
        db.set("current_ctx", text[:MAX_CTX_LEN])
        db.set("current_ctx_updated", datetime.now(timezone.utc).isoformat())

    def clear(self) -> None:
        """Supprime la note (force le fallback)."""
        db = self._db()
        db.set("current_ctx", "")
        db.set("current_ctx_updated", "")

    # -- Refresh automatique -------------------------------------------------

    async def refresh(self) -> Optional[str]:
        """Rafraîchit depuis Brave News. Retourne la note générée, ou None si échec/pas de clé."""
        if not self._brave_key:
            logger.debug("Awareness refresh skipped: pas de clé Brave")
            return None

        date_str = datetime.now().strftime("%d/%m/%Y")

        # Deux requêtes parallèles : sport + actu générale
        sport_task = asyncio.to_thread(_brave_news, self._brave_key, "actualités sport du jour", 5)
        world_task = asyncio.to_thread(_brave_news, self._brave_key, "actualités monde du jour", 4)
        sport_res, world_res = await asyncio.gather(sport_task, world_task)

        all_results = sport_res + world_res
        note = _build_ctx(all_results, date_str)
        if not note:
            logger.warning("Awareness refresh: aucun résultat Brave")
            return None

        self.set(note)
        logger.info(f"Awareness refreshed ({len(note)} car.)")
        return note

    # -- Injection prompt ----------------------------------------------------

    def prompt_hint(self) -> str:
        """Retourne le bloc à injecter dans le developer prompt."""
        ctx = self.get()
        return ctx if ctx else _FALLBACK
