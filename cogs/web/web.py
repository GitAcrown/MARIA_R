"""Cog Web — recherche et lecture de pages avec crawler."""

import asyncio
import logging
import re
import time
import html
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from discord.ext import commands

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

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

logger = logging.getLogger("web")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
DIFFICULT_DOMAINS = {"twitter.com", "x.com", "facebook.com", "instagram.com", "reddit.com", "medium.com", "linkedin.com"}
SEARCH_CACHE_SEC = 300
PAGE_CACHE_HOURS = 12
CHUNK_SIZE = 2000
SCREENSHOT_API = "https://image.thum.io/get/width/1280/crop/900/"


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
                results.append({"title": item.get("title", ""), "url": url, "body": body.strip()})
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
                entry: dict = {"title": r.get("title", ""), "url": url, "body": r.get("body", "")}
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

    # ------------------------------------------------------------------
    # Tool handlers (async — exécutent le I/O bloquant dans un thread)
    # ------------------------------------------------------------------

    async def _tool_search(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        q = tc.arguments.get("query", "").strip()
        lang = tc.arguments.get("lang", "fr")
        if not q:
            return ToolResponseRecord(tc.id, {"error": "Requête manquante"}, datetime.now(timezone.utc))
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, self._search, q, lang, 4)
        if not res:
            return ToolResponseRecord(tc.id, {"error": "Aucun résultat"}, datetime.now(timezone.utc))
        return ToolResponseRecord(
            tc.id,
            {"query": q, "results": res, "note": "Utilise read_web_page sur une URL pour plus de détails."},
            datetime.now(timezone.utc),
        )

    def _screenshot_url(self, url: str) -> str | None:
        """Vérifie que thum.io répond bien avant de renvoyer l'URL screenshot."""
        screenshot_url = SCREENSHOT_API + url
        try:
            r = requests.head(screenshot_url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                return screenshot_url
        except Exception as e:
            logger.warning(f"Screenshot head check failed ({url}): {e}")
        return None

    async def _tool_screenshot(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        url = tc.arguments.get("url", "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return ToolResponseRecord(tc.id, {"error": "URL invalide"}, datetime.now(timezone.utc))
        loop = asyncio.get_event_loop()
        screenshot_url = await loop.run_in_executor(None, self._screenshot_url, url)
        if not screenshot_url:
            return ToolResponseRecord(
                tc.id,
                {"error": f"Impossible de capturer {urlparse(url).netloc}"},
                datetime.now(timezone.utc),
            )
        logger.info(f"Screenshot: {url}")
        return ToolResponseRecord(
            tc.id,
            {"screenshot_url": screenshot_url, "source_url": url},
            datetime.now(timezone.utc),
        )

    async def _tool_read(self, tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        url = tc.arguments.get("url", "").strip()
        if not url or not url.startswith(("http://", "https://")):
            return ToolResponseRecord(tc.id, {"error": "URL invalide"}, datetime.now(timezone.utc))
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, self._crawl_page, url)
        if not content:
            domain = urlparse(url).netloc
            return ToolResponseRecord(
                tc.id,
                {"error": f"Impossible de lire {domain} (blocage ou format non supporté).", "url": url},
                datetime.now(timezone.utc),
            )
        chunks = self._chunk_text(content)
        return ToolResponseRecord(
            tc.id,
            {"url": url, "content": chunks[0] if chunks else content[:2000], "total_chunks": len(chunks)},
            datetime.now(timezone.utc),
        )

    @property
    def GLOBAL_TOOLS(self) -> list:
        return [
            Tool(
                name="search_web",
                description="Recherche web. À utiliser pour l'actualité, les événements récents, les faits du monde réel, ou toute info potentiellement obsolète dans tes données d'entraînement.",
                properties={
                    "query": {"type": "string", "description": "Requête précise"},
                    "lang": {"type": "string", "description": "Code langue (défaut: fr)"},
                },
                function=self._tool_search,
            ),
            Tool(
                name="read_web_page",
                description="Lit le contenu d'une URL. Si les extraits de search_web sont insuffisants.",
                properties={"url": {"type": "string", "description": "URL complète"}},
                function=self._tool_read,
            ),
            Tool(
                name="screenshot_page",
                description="Prend une capture d'écran d'une page web et l'affiche. Utile pour voir le rendu visuel d'un site.",
                properties={"url": {"type": "string", "description": "URL complète de la page à capturer"}},
                function=self._tool_screenshot,
            ),
        ]


async def setup(bot):
    await bot.add_cog(Web(bot))
