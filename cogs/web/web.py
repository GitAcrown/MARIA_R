"""Cog Web — recherche et lecture de pages avec crawler."""

import asyncio
import logging
import re
import time
import html
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import discord
from bs4 import BeautifulSoup
from ddgs import DDGS
from discord.ext import commands

from common.discord_ui import layout_with_commentary
from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.widgets import register_widget, unregister_widget

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

logger = logging.getLogger("MARIA.Web")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
DIFFICULT_DOMAINS = {"twitter.com", "x.com", "facebook.com", "instagram.com", "reddit.com", "medium.com", "linkedin.com"}
SEARCH_CACHE_SEC = 300
PAGE_CACHE_HOURS = 12
CHUNK_SIZE = 2000
_MAX_GALLERY_IMAGES = 4


# ---------------------------------------------------------------------------
# Builder LayoutView — galerie d'images
# ---------------------------------------------------------------------------

def build_image_view(data: dict, commentary: str = ""):
    """Construit une galerie d'images (MediaGallery) depuis le résultat search_images."""
    if not isinstance(data, dict) or "error" in data:
        return None
    images = data.get("images") or []
    if not images:
        return None

    gallery = discord.ui.MediaGallery()
    added = 0
    for img in images[:_MAX_GALLERY_IMAGES]:
        url = (img.get("image_url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        try:
            gallery.add_item(media=url, description=(img.get("title") or "")[:256] or None)
            added += 1
        except Exception:
            continue
    if added == 0:
        return None

    return layout_with_commentary(gallery, commentary)

class Web(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._brave_api_key: str = getattr(bot, "config", {}).get("BRAVE_API_KEY", "") or ""
        self._search_cache: dict[str, tuple[list, float]] = {}
        self._page_cache: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Crawler
    # ------------------------------------------------------------------

    def _crawl_page(self, url: str) -> str:
        """Fetch + extraction en cascade (trafilatura > readability > bs4)."""
        domain = urlparse(url).netloc
        if any(d in domain for d in DIFFICULT_DOMAINS):
            return ""

        if url in self._page_cache:
            content, ts = self._page_cache[url]
            if time.time() - ts < PAGE_CACHE_HOURS * 3600:
                return content

        try:
            r = requests.get(url, headers=HEADERS, timeout=(5, 15), allow_redirects=True)
            if r.status_code != 200:
                return ""
            html_content = r.text
        except Exception:
            return ""

        text = ""
        if TRAFILATURA_AVAILABLE:
            try:
                text = trafilatura.extract(
                    html_content, url=url,
                    include_comments=False, include_tables=False,
                    favor_recall=True,
                ) or ""
            except Exception:
                pass

        if (not text or len(text.strip()) < 200) and READABILITY_AVAILABLE:
            try:
                doc = Document(html_content)
                if doc.summary():
                    soup = BeautifulSoup(doc.summary(), "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
            except Exception:
                pass

        if not text or len(text.strip()) < 200:
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            for sel in [".ad", ".cookie", ".sidebar", ".comments", ".share"]:
                for el in soup.select(sel):
                    el.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main:
                text = main.get_text(separator="\n", strip=True)

        if text:
            text = html.unescape(text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text).strip()

        if text and len(text) > 200:
            self._page_cache[url] = (text, time.time())
        return text

    def _chunk_text(self, text: str, size: int = CHUNK_SIZE) -> list[str]:
        chunks = []
        paras = [p for p in re.split(r"\n\n+", text) if len(p.strip()) > 30]
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 2 > size and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur.strip():
            chunks.append(cur.strip())
        return [c for c in chunks if len(c) > 100]

    # ------------------------------------------------------------------
    # Moteurs de recherche (synchrones, exécutés dans un thread)
    # ------------------------------------------------------------------

    def _brave_search(self, query: str, lang: str = "fr", n: int = 4) -> list[dict]:
        """Brave Search API — résultats web de qualité avec fraîcheur."""
        if not self._brave_api_key:
            return []
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_api_key,
        }
        params = {
            "q": query,
            "count": min(n + 2, 10),
            "search_lang": lang,
            "country": "FR",
            "extra_snippets": "1",
        }
        logger.info(f"Brave web search: {query!r}")
        try:
            r = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            raw = data.get("web", {}).get("results", [])
            results: list[dict] = []
            seen: set[str] = set()
            for item in raw:
                url = item.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                snippets = item.get("extra_snippets", [])
                body = item.get("description", "")
                if snippets:
                    body = body + " " + " ".join(snippets[:2])
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "body": body.strip(),
                    "date": item.get("age") or item.get("page_age") or "",
                })
                if len(results) >= n:
                    break
            logger.info(f"Brave web search: {len(results)} résultat(s) pour {query!r}")
            return results
        except Exception as e:
            logger.warning(f"Brave web search failed ({query!r}): {e}")
            return []

    def _brave_news_search(self, query: str, lang: str = "fr", n: int = 4) -> list[dict]:
        """Brave News API — actualités récentes."""
        if not self._brave_api_key:
            return []
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_api_key,
        }
        params = {"q": query, "count": min(n * 2, 20), "search_lang": lang, "country": "FR"}
        logger.info(f"Brave news search: {query!r}")
        try:
            r = requests.get(
                "https://api.search.brave.com/res/v1/news/search",
                headers=headers,
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            raw = data.get("results", [])
            results: list[dict] = []
            seen: set[str] = set()
            for item in raw:
                url = item.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "body": item.get("description", ""),
                    "date": item.get("age", ""),
                    "source": item.get("meta_url", {}).get("hostname", ""),
                })
                if len(results) >= n:
                    break
            logger.info(f"Brave news search: {len(results)} résultat(s) pour {query!r}")
            return results
        except Exception as e:
            logger.warning(f"Brave news search failed ({query!r}): {e}")
            return []

    def _brave_image_search(self, query: str, lang: str = "fr", n: int = 4) -> list[dict]:
        """Brave Images API — recherche d'images."""
        if not self._brave_api_key:
            return []
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_api_key,
        }
        params = {
            "q": query[:400],
            "count": min(n + 2, 20),
            "search_lang": lang,
            "country": "FR",
            "safesearch": "strict",  # l'API Images n'accepte que "off" ou "strict"
        }
        logger.info(f"Brave image search: {query!r}")
        try:
            r = requests.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers=headers,
                params=params,
                timeout=10,
            )
            if r.status_code != 200:
                logger.warning(f"Brave image search HTTP {r.status_code} ({query!r}): {r.text[:300]}")
                return []
            data = r.json()
            raw = data.get("results", [])
            results: list[dict] = []
            seen: set[str] = set()
            for item in raw:
                img_url = (item.get("properties") or {}).get("url", "")
                if not img_url or img_url in seen:
                    continue
                seen.add(img_url)
                results.append({
                    "title":     item.get("title", ""),
                    "image_url": img_url,
                    "source":    item.get("url", ""),
                })
                if len(results) >= n:
                    break
            logger.info(f"Brave image search: {len(results)} résultat(s) pour {query!r}")
            return results
        except Exception as e:
            logger.warning(f"Brave image search failed ({query!r}): {e}")
            return []

    def _ddg_search(self, query: str, lang: str = "fr", n: int = 4) -> list[dict]:
        """DuckDuckGo — fallback si pas de clé Brave."""
        logger.info(f"DDG search: {query!r}")
        results: list[dict] = []
        seen: set[str] = set()
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query=query, region=f"{lang}-{lang}", max_results=max(n, 8)))
            for r in raw[:6]:
                url = r.get("href", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                entry: dict = {
                    "title": r.get("title", ""),
                    "url": url,
                    "body": r.get("body", ""),
                    "date": r.get("date", "") or "",
                }
                results.append(entry)
                if len(results) <= 2:
                    excerpt = self._crawl_page(url)
                    if excerpt:
                        entry["excerpt"] = excerpt[:600] + ("..." if len(excerpt) > 600 else "")
        except Exception as e:
            logger.warning(f"DDG text search failed ({query!r}): {e}")

        if len(results) < n:
            try:
                with DDGS() as ddgs:
                    news_raw = list(ddgs.news(query=query, region=f"{lang}-{lang}", max_results=max(n, 8)))
                for r in news_raw:
                    url = r.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "body": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", ""),
                    })
                    if len(results) >= n:
                        break
            except Exception as e:
                logger.warning(f"DDG news search failed ({query!r}): {e}")

        logger.info(f"DDG search: {len(results)} résultat(s) pour {query!r}")
        return results[:n]

    def _search(self, query: str, lang: str = "fr", n: int = 4) -> list[dict]:
        """Recherche avec cache. Brave si clé dispo, sinon DDG."""
        key = f"{lang}:{query.strip().lower()}"
        if key in self._search_cache:
            res, ts = self._search_cache[key]
            if time.time() - ts < SEARCH_CACHE_SEC:
                return res[:n]

        if self._brave_api_key:
            results = self._brave_search(query, lang, n)
            # News Brave uniquement si le web search est vraiment vide (économie de crédits)
            if len(results) < 2:
                news = self._brave_news_search(query, lang, n)
                seen = {r["url"] for r in results}
                for item in news:
                    if item["url"] not in seen:
                        results.append(item)
                        if len(results) >= n:
                            break
            # Fallback DDG si Brave échoue (quota dépassé, erreur réseau…)
            if not results:
                results = self._ddg_search(query, lang, n)
        else:
            results = self._ddg_search(query, lang, n)

        self._search_cache[key] = (results, time.time())
        return results[:n]

    @staticmethod
    def _as_sources(results: list[dict]) -> list[dict]:
        """Normalise les hits pour le LLM et le footer Discord (ids stables, pas d'URL inventée)."""
        out: list[dict] = []
        for i, item in enumerate(results, start=1):
            snippet = (item.get("body") or item.get("excerpt") or "").strip()
            if len(snippet) > 280:
                snippet = snippet[:279] + "…"
            out.append({
                "id": f"s{i}",
                "title": (item.get("title") or "").strip(),
                "url": item.get("url") or "",
                "snippet": snippet,
                "date": (item.get("date") or "").strip(),
            })
        return out

    # ------------------------------------------------------------------
    # Tool handlers (async — exécutent le I/O bloquant dans un thread)
    # ------------------------------------------------------------------

    async def _tool_search(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        q = tc.arguments.get("query", "").strip()
        lang = tc.arguments.get("lang", "fr") or "fr"
        if not q:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))
        res = await asyncio.to_thread(self._search, q, lang, 4)
        if not res:
            return ToolResponseRecord(
                tc.id,
                {
                    "error": "Aucun résultat",
                    "query": q,
                    "reformulate_hint": (
                        "Reformule avec un nom propre, une date, un lieu ou un terme plus précis, "
                        "puis rappelle search_web."
                    ),
                },
                datetime.now(timezone.utc),
            )
        sources = self._as_sources(res)
        return ToolResponseRecord(
            tc.id,
            {
                "query": q,
                "results": sources,
                "note": (
                    "Cite uniquement ces ids (s1, s2…). "
                    "Si les extraits sont minces, read_web_page sur l'URL. "
                    "N'invente aucune URL."
                ),
            },
            datetime.now(timezone.utc),
        )

    async def _tool_images(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        q = tc.arguments.get("query", "").strip()
        lang = tc.arguments.get("lang", "fr")
        try:
            count = int(tc.arguments.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        count = max(1, min(count, _MAX_GALLERY_IMAGES))
        if not q:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))
        if not self._brave_api_key:
            return ToolResponseRecord(tc.id, {"error": "Recherche d'images indisponible (clé Brave manquante)"}, datetime.now(timezone.utc))
        res = await asyncio.to_thread(self._brave_image_search, q, lang, count)
        if not res:
            return ToolResponseRecord(tc.id, {"error": "Aucune image trouvée"}, datetime.now(timezone.utc))
        return ToolResponseRecord(
            tc.id,
            {
                "_tool":        "search_images",
                "_llm_summary": f"Galerie de {len(res)} image(s) affichée pour « {q} ».",
                "query":        q,
                "images":       res,
            },
            datetime.now(timezone.utc),
        )

    async def _tool_read(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        url = tc.arguments.get("url", "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return ToolResponseRecord(tc.id, {"error": "URL invalide"}, datetime.now(timezone.utc))
        host = (urlparse(url).netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host in {"youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com"}:
            return ToolResponseRecord(
                tc.id,
                {"error": "C'est une vidéo YouTube — appelle read_youtube avec cette URL."},
                datetime.now(timezone.utc),
            )
        raw_chunk = tc.arguments.get("chunk")
        try:
            chunk_idx = 0 if raw_chunk is None else int(raw_chunk)
        except (TypeError, ValueError):
            chunk_idx = 0
        content = await asyncio.to_thread(self._crawl_page, url)
        if not content:
            domain = urlparse(url).netloc
            return ToolResponseRecord(
                tc.id,
                {"error": f"Impossible de lire {domain} (blocage ou format non supporté).", "url": url},
                datetime.now(timezone.utc),
            )
        chunks = self._chunk_text(content)
        if not chunks:
            chunks = [content[:CHUNK_SIZE]]
        chunk_idx = max(0, min(chunk_idx, len(chunks) - 1))
        next_chunk = chunk_idx + 1 if chunk_idx + 1 < len(chunks) else None
        return ToolResponseRecord(
            tc.id,
            {
                "id": "p1",
                "url": url,
                "content": chunks[chunk_idx],
                "chunk": chunk_idx,
                "total_chunks": len(chunks),
                "next_chunk": next_chunk,
            },
            datetime.now(timezone.utc),
        )

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="search_web",
                description=(
                    "Recherche web pour l'actualité, les faits du monde réel, les dates, "
                    "chiffres, définitions ou toute info potentiellement obsolète. "
                    "PAS pour un avis, une blague, ou ce qui s'est dit dans le salon. "
                    "Les résultats portent des ids (s1, s2…) : cite-les, n'invente pas d'URL."
                ),
                properties={
                    "query": {"type": "string", "description": "Requête précise (noms, date, lieu)"},
                    "lang": {"type": "string", "description": "Code langue (défaut: fr)"},
                },
                optional_props=["lang"],
                function=self._tool_search,
            ),
            Tool(
                name="read_web_page",
                description=(
                    "Lit le contenu d'une URL déjà obtenue (search_web ou lien du message). "
                    "Si total_chunks > 1, rappelle avec chunk=next_chunk pour la suite. "
                    "Pas pour YouTube (read_youtube)."
                ),
                properties={
                    "url": {"type": "string", "description": "URL complète https://…"},
                    "chunk": {
                        "type": "integer",
                        "description": "Index du morceau à lire (0 = début).",
                    },
                },
                optional_props=["chunk"],
                function=self._tool_read,
            ),
            Tool(
                name="search_images",
                description=(
                    "Recherche des images sur le web via Brave et les affiche dans une galerie. "
                    "Utile pour illustrer une réponse, trouver une photo, un logo, un personnage, etc. "
                    "Choisis count selon le besoin/demande : 1 par défaut, plus si on veut un aperçu (max 4)."
                ),
                properties={
                    "query": {"type": "string", "description": "Requête de recherche d'images"},
                    "lang":  {"type": "string", "description": "Code langue (défaut: fr)"},
                    "count": {
                        "type":        "integer",
                        "description": "Nombre d'images à afficher, entre 1 et 4 (défaut 1).",
                        "minimum":     1,
                        "maximum":     _MAX_GALLERY_IMAGES,
                    },
                },
                function=self._tool_images,
            ),
        ]


async def setup(bot):
    await bot.add_cog(Web(bot))
    register_widget("search_images", build_image_view)


async def teardown(bot):
    unregister_widget("search_images")
