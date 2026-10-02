"""Cache TTL minimal, thread-safe et borné — pour les appels d'API externes.

Les outils réseau tournent dans `asyncio.to_thread` : plusieurs threads peuvent
lire/écrire en parallèle, d'où le verrou. Seuls les succès doivent être mis en
cache (l'appelant décide) pour ne jamais figer une erreur transitoire.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Hashable, Optional

_MISSING = object()


class TTLCache:
    def __init__(self, ttl: float, *, maxsize: int = 256):
        self.ttl = float(ttl)
        self.maxsize = max(1, int(maxsize))
        self._data: dict[Hashable, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key, _MISSING)
            if entry is _MISSING:
                return default
            ts, value = entry  # type: ignore[misc]
            if now - ts >= self.ttl:
                del self._data[key]
                return default
            return value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._data) >= self.maxsize and key not in self._data:
                self._evict(now)
            self._data[key] = (now, value)

    def _evict(self, now: float) -> None:
        """Purge les expirés ; si c'est encore plein, retire le plus ancien."""
        expired = [k for k, (ts, _) in self._data.items() if now - ts >= self.ttl]
        for k in expired:
            del self._data[k]
        if len(self._data) >= self.maxsize:
            oldest: Optional[Hashable] = min(self._data, key=lambda k: self._data[k][0])
            del self._data[oldest]

    def discard(self, key: Hashable) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
